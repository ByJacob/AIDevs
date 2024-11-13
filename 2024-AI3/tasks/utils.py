import base64
import os
import re
import zipfile
from io import BytesIO

import requests

from PIL import Image


def find_flag(text):
    pattern = r"\{\{FLG:.*?\}\}"
    # Find all matches
    matches = re.findall(pattern, text)

    # Print matches
    return matches


def create_message(role, text):
    """Creates a dictionary for a chat message with a role and text."""
    if role not in ['system', 'user', 'assistant', 'tool']:
        raise ValueError("Role must be one of 'system', 'user', or 'assistant'")
    return {"role": role, "content": text}


def create_message_with_image(role, text, images_path, images_format):
    if images_format is None:
        raise NotImplementedError(f"Model cannot use chat with image")
    content = [dict(type="text", text=text)]
    for image_path in images_path:
        base64_image = resize_to_best_format_base64(image_path, images_format)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"{base64_image}",
                "detail": "high"
            },
        })
    return create_message(role, content)


def download_and_extract_zip(extract_dir, url):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    zip_path = os.path.join(extract_dir, url.split('/')[-1])
    # Check if the file is already downloaded
    if not os.path.exists(zip_path):
        # Download the file
        response = requests.get(url)
        with open(zip_path, "wb") as file:
            file.write(response.content)
        print("Extracting the contents of the zip file...")
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(extract_dir)


def resize_to_best_format_base64(image_path, formats):
    # Define available formats

    supported_formats = {"PNG", "JPEG", "JPG", "WEBP", "GIF"}

    file_extension = os.path.splitext(image_path)[-1].lower()
    if file_extension not in [".png", ".jpeg", ".jpg", ".webp", ".gif"]:
        raise Exception(f"Unsupported file format. We currently support {', '.join(supported_formats)}")

    # Load the image
    with Image.open(image_path) as img:

        if img.format not in supported_formats:
            raise Exception(f"Unsupported file format. We currently support {', '.join(supported_formats)}")

        original_format = img.format
        original_width, original_height = img.size
        original_aspect_ratio = original_width / original_height

        # Find the best matching format based on aspect ratio
        if len(formats) > 0:
            best_format = min(formats, key=lambda x: abs((x[0] / x[1]) - original_aspect_ratio))
        else:
            best_format = (original_width, original_height)

        # Resize the image to the best matching format
        resized_image = img.resize(best_format, Image.Resampling.LANCZOS)

        # Convert the resized image to base64
        buffered = BytesIO()
        resized_image.save(buffered, format=original_format)
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Format the base64 string with data URI prefix
        mime_type = f"image/{original_format.lower()}"
        return f"data:{mime_type};base64,{base64_image}"
