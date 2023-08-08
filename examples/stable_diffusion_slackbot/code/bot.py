import io 
import os
from typing import Optional

from fastapi import Request
from modal import Image, Secret, SharedVolume, Stub, web_endpoint

stub = Stub("oxen-stable-diffusion-bot")
volume = SharedVolume().persist("stable-diff-model-vol")

CACHE_PATH = "/root/model_cache"

@stub.function(
    gpu="T4",
    image=(Image.debian_slim()
        .pip_install("diffusers", "transformers", "scipy", "ftfy", "accelerate", "torch", "slack-sdk")),
    shared_volumes={CACHE_PATH: volume},
    secret=Secret.from_name("huggingface-token"),
)

async def run_stable_diffusion(prompt: str, channel_name: Optional[str] = None):
    from diffusers import StableDiffusionPipeline
    from torch import float16

    pipe = StableDiffusionPipeline.from_pretrained(
        "bartuso/ox_2",
        use_auth_token=os.environ["HUGGINGFACE-TOKEN"],
        torch_dtype=float16,
        cache_dir=CACHE_PATH,
        device_map="auto"
    )

    image = pipe(prompt, num_inference_steps=50).images[0]

    # Convert PIL image to PNG byte array 
    with io.BytesIO() as buf:
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
    
    if channel_name: 
        post_image_to_slack.call(prompt, channel_name, img_bytes)
    
    return img_bytes

@stub.function()
@web_endpoint(method="POST")
async def entrypoint(request: Request):
    body = await request.form()
    prompt = body["text"]
    run_stable_diffusion.spawn(prompt, body["channel_name"])
    return f"Running stable diffusion for {prompt}."

@stub.function(
    image=Image.debian_slim().pip_install("slack-sdk"),
    secret=Secret.from_name("slack-secret"),
)
def post_image_to_slack(title: str, channel_name: str, image_bytes: bytes):
    import slack_sdk

    client = slack_sdk.WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    client.files_upload(channels=channel_name, title=title, content=image_bytes)

# Testing 
@stub.local_entrypoint()
def run(
    prompt: str = "an image of the oxenai ox eating cereal",
    output_dir: str= "/tmp/stable-diffusion"
):
    os.makedirs(output_dir, exist_ok=True)
    img_bytes = run_stable_diffusion.call(prompt)
    output_path = os.path.join(output_dir, "output.png")
    with open(output_path, "wb") as f:
        f.write(img_bytes)
    print(f"Wrote data to {output_path}")