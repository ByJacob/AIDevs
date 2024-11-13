import glob
import re

import requests
from langfuse.decorators import observe

from models import *
from utils import create_message, create_message_with_image, download_and_extract_zip

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    model = OpenAi4o(debug=True)
    pass
    images = []
    properties = []
    for image_idx in range(4):
        image_path = os.path.join("assets", f"S02E02-{image_idx+1}.png")
        images.append(image_path)
    message_text = """
Analyze the map fragment provided. Extract the names of the streets and the places on it. 
Based on this information, analyze what city in Poland this map fragment is from.
The city you are looking for must have a historic granary and a historic fortress.
"""
    message = create_message_with_image("user", message_text, images, model.formats)
    result = model.chat([message], temperature=0.1)


if __name__ == '__main__':
    main()
