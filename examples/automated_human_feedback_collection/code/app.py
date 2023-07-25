import torch
from diffusers import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt
import os
import gradio as gr
import shortuuid
from oxen import RemoteRepo

# Config Oxen
repo = RemoteRepo("ba/active-learning-test")
repo.checkout("dev")

# Config model training
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")

IMAGE_DEST="images"

def generate_images(prompt):
    images = pipe(prompt, guidance_scale=7.5, num_images_per_prompt=4).images
    return images

def get_rating(label):
    if label == '👍':
        return 'good'
    elif label == '👎':
        return 'bad'
    else:
        raise ValueError(f"Unexpected rating label: {label}")

def save_image_to_repo(img_prompt, img, button):
    # Download locally 
    filename = f"{shortuuid.uuid()}.png"
    img.save(filename)
    print(button)
    rating = get_rating(button)
    row = {"prompt": img_prompt, "path": f"{IMAGE_DEST}/{filename}", "rating": rating}
    # Remotely stage to repo 
    try:  
        repo.add(filename, IMAGE_DEST)
        repo.add_df_row("train.csv",  row)
        repo.commit(f"Add from RLHF: {img_prompt}")
    except Exception as e:
        print(e)
        # Unstage local changes 
        repo.remove(filename)
        repo.restore_df("train.csv")
    os.remove(filename)


with gr.Blocks() as demo:
    prompt = gr.components.Textbox(label="Enter Prompt")
    generate = gr.Button("Generate candidate images")
    images = {}
    upvote_buttons = {}
    downvote_buttons = {}
    with gr.Row():
        for i in range(1,5):
            with gr.Column(min_width=0, scale=1):
                images[i] = gr.components.Image(label=f"Candidate Image {i}", type='pil').style(full_width=False)
                with gr.Row(min_width=0, scale=1):
                    upvote_buttons[i] = gr.Button(value="👍", min_width=10)
                    downvote_buttons[i] = gr.Button(value="👎", min_width=10)
    
    generate.click(generate_images, inputs=prompt, outputs=list(images.values()))
    for i in range(1,5):
        upvote_buttons[i].click(save_image_to_repo, inputs=[prompt, images[i], upvote_buttons[i]])
        downvote_buttons[i].click(save_image_to_repo, inputs=[prompt, images[i], downvote_buttons[i]])

demo.launch(share=True)