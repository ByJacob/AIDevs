import base64
import json
import os
import re
import zipfile
from io import BytesIO

import requests

from PIL import Image
from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams


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


def create_message_with_image(role, text, images_path, images_format, detail="high"):
    if images_format is None:
        raise NotImplementedError(f"Model cannot use chat with image")
    content = [dict(type="text", text=text)]
    for image_path in images_path:
        base64_image = resize_to_best_format_base64(image_path, images_format)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"{base64_image}",
                "detail": detail
            },
        })
    return create_message(role, content)


def download_file(extract_dir, url):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    filename = os.path.join(extract_dir, url.split('/')[-1])
    # Check if the file is already downloaded
    if not os.path.exists(filename):
        # Download the file
        response = requests.get(url)
        with open(filename, "wb") as file:
            file.write(response.content)
        return True
    return False


def download_and_extract_zip(extract_dir, url):
    if download_file(extract_dir, url):
        zip_path = os.path.join(extract_dir, url.split('/')[-1])
        extract_zip(zip_path, extract_dir)


def extract_zip(zip_path, extract_dir, pwd=None):
    if not os.path.exists(extract_dir):
        print("Extracting the contents of the zip file...")
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(extract_dir, pwd=bytes(str(pwd), 'utf-8'))


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


def strip_text(text, left, right):
    # Find the position of the first '{' and the last '}'
    left = text.find('{')
    right = text.rfind('}')

    # Return the content between the braces if both are found
    if left != -1 and right != -1:
        return text[left + 1:right]
    return None


def extract_md_link(text):
    return strip_text(text, '(', ')')


def extract_json(text):
    return_str = strip_text(text, '{', '}')
    if return_str is None:
        return None
    try:
        _ = json.loads(f"{{{return_str}}}")
        return f"{{{return_str}}}"
    except:
        return None


def init_qdrant(model, qdrant_collection_name):
    client = QdrantClient(
        host=os.getenv('AIDEVS3_QDRANT_HOST'),
        port=6333,
        api_key=os.getenv('AIDEVS3_QDRANT_KEY')
    )
    all_collections = list(map(lambda x: x.name, client.get_collections().collections))
    if qdrant_collection_name not in all_collections:
        result = model.embedding("PING")
        pass
        client.create_collection(
            qdrant_collection_name,
            vectors_config=VectorParams(
                size=len(result.embedding),
                distance=Distance.COSINE,
            ),
        )
    return client
