import base64
import re

import requests
from langfuse.decorators import observe

from models import *
from utils import create_message, find_flag, download_and_extract_zip

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe()
def generate_image_and_save(prompt, file_name):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    batch_size = 1
    json_data = {
        'prompt': prompt,
        'negative_prompt': '',
        'batch_size': batch_size,
        'n_iter': 1,
        'steps': 50,
        "sampler_name": "Euler a",
        'cfg_scale': 7,
        'width': 512,
        'height': 512,
        "hr_scale": 2,
        "hr_upscaler": "Lanczos",
        "denoising_strength": 0.7
    }
    base_url = "https://players-cap-various-su.trycloudflare.com"
    urls = []
    response = requests.post('http://127.0.0.1:7860/sdapi/v1/txt2img', headers=headers, json=json_data)
    images = response.json()["images"]
    for idx in range(batch_size):
        if idx == 0 and batch_size > 1:
            continue
        # Decode the base64 string and save it as an image file
        file_name = f"{file_name}_{idx}.png"
        with open(os.path.join("assets", file_name), "wb") as file:
            file.write(base64.b64decode(images[idx]))
        urls.append(f"{base_url}/{file_name}")
    return urls


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    model = MistralP7B(debug=True)

    data_url = f"{domain}/data/{mykey}/robotid.json"

    response = requests.get(data_url)
    data = response.json()["description"]

    system_message = """
Convert user input into a structured prompt for Stable Diffusion 1.5 that generates a detailed illustration 
with a robot as the main focus. Follow these steps:

- Main Subject (Robot): Prioritize creating a futuristic or sci-fi robot as per the user’s details. 
Use descriptors like "cybernetic," "mechanical," "android," or "robotic character."

- Art Style: Choose an illustrative style with terms like "concept art," "digital art," "highly detailed," 
"cyberpunk," or "futuristic." Adjust the style based on any specifics from the user, 
such as "retro-futuristic" or "steampunk."

- Background and Environment: If a setting is given (e.g., "futuristic city," "industrial scene"), 
add terms like "neon-lit," "dark sci-fi cityscape," or "laboratory environment." 
Incorporate descriptors for mood and lighting like "moody lighting," "bright neon glow," or "soft shadows."

- Robot Details and Features: Highlight any specific features mentioned, such as "metallic textures," 
"glowing lights," or "weathered, worn details." Use phrases like "mechanical limbs," 
"intricate circuitry," "sleek design," "rusted metal," or "polished armor" to enhance specific traits.

- Composition: Keep the composition simple based on user direction (e.g., "close-up," "full body," "portrait view") 
and emphasize high detail for a sharp, vivid visual.

Sample Output:
"illustration, futuristic robot, [user details], cybernetic, android, digital art, concept art, cyberpunk, sci-fi, 
[user-specified setting], intricate details, dramatic lighting, neon glow, metallic, 
glowing circuitry, [composition type: full-body, close-up, etc.], cinematic, high detail"
"""
    generated_prompt = model.chat([
        create_message("system", system_message),
        create_message("user", data)
    ])
    # generated_prompt = model.chat([
    #     create_message("system", system_message),
    #     create_message("user", data),
    #     create_message("assistant", generated_prompt),
    #     create_message("user", "Translate this prompt in English language")
    # ])
    print(generated_prompt)
    urls = generate_image_and_save(generated_prompt, "S02E03")
    for url in urls:
        data = dict(answer=url, apikey=mykey, task="robotid")
        answer_result = requests.post(f"{domain}/report", json=data)
        print(answer_result.text)
        flag = find_flag(answer_result.text)
        print(flag)


if __name__ == '__main__':
    main()
