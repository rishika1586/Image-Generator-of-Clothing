import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Load the Stable Diffusion model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use GPU for faster generation

# Define a prompt for clothing generation
prompt = "A futuristic cyberpunk-style leather jacket with neon blue highlights, detailed fabric texture, highly realistic, studio lighting"

# Generate image
image = pipe(prompt).images[0]

# Save and show the output
image.save("generated_clothing.png")
image.show()