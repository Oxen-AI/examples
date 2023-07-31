import torch
from diffusers import StableDiffusionPipeline
from PIL import Image 
import gradio as gr 

# Run inference with the base stable diffusion model 
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda") # If using CUDA for GPU

def generate_images(prompt):
    images = pipe(prompt, guidance_scale=7.5, num_images_per_prompt=4).images
    return images 

with gr.Blocks() as demo: 
	prompt = gr.components.Textbox(label="Enter Prompt")
	generate = gr.Button("Generate candidate images") 
	images = {}
	with gr.Row():
		for i in range(1,5):
			with gr.Column():
				images[i] = gr.components.Image(label=f"Candidate Image {i}", type='pil')
	generate.click(generate_images, inputs=prompt, outputs=list(images.values()))

demo.launch(share=True)