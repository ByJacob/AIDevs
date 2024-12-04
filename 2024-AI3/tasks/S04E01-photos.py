import requests

from models import *
from utils import create_message, create_message_with_image, extract_json, download_file, find_flag

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


def send_report(answer):
    apidb = f"{domain}/report"
    body = {
        "task": "photos",
        "apikey": mykey,
        "answer": answer
    }
    result = requests.post(apidb, json=body)
    return result.json()


ANALIZE_PERSON_MSG = """Przygotuj opis cech szczególnych osoby na zdjeciach. Opisz tylko osobę, która występuje najczęściej
Zwróć uwagę na:
- kolor włosów
- rysy twarzy
- szacowany wiek

i inne cechy szczególne.
"""


class PhotoRepair:
    UNDERSTAND_QUERY_TEXT = """
Here’s the final prompt structured in one complete markdown document:

```markdown
# Extract Image Links from Text

This prompt extracts image URLs from given messages and returns them in a specified JSON format.

<prompt_objective>
Extract links to images from provided text messages and return them in a structured JSON object.
</prompt_objective>

<prompt_rules>
- Return a JSON object with fields `_thinking` and `images`.
- The `_thinking` field should contain a description of how the images were obtained.
- The `images` field must be an array of correctly extracted image URLs.
- NEVER return malformed JSON.
- NEVER include non-image URLs in the `images` array.
- NEVER provide any URLs or fields that do not meet the specified criteria.
</prompt_rules>

<prompt_examples>
USER: "Check out these stunning images of the Milky Way I found: https://example.com/image1 and 
https://example.com/image2!"
AI: 
{
  "_thinking": "The way the pictures were taken out",
  "images": ["https://example.com/image1", "https://example.com/image2"]
}

USER: "Here are some concept art pieces I came across: https://artgallery.com/artpiece1, 
https://artgallery.com/artpiece2."
AI: 
{
  "_thinking": "The way the pictures were taken out",
  "images": ["https://artgallery.com/artpiece1", "https://artgallery.com/artpiece2"]
}

USER: "I love the futuristic designs in these visuals: https://designhub.com/visual1 and https://designhub.com/visual2."
AI: 
{
  "_thinking": "The way the pictures were taken out",
  "images": ["https://designhub.com/visual1", "https://designhub.com/visual2"]
}

USER: "At example.com you will find images IMAGE1.jpg, IMAGE2.png, IMAGE3.svg."
AI: 
{
  "_thinking": "The way the pictures were taken out",
  "images": ["https://example.com/IMAGE1.jpg", "https://example.com/IMAGE2.png", "https://example.com/IMAGE3.svg"]
}
</prompt_examples>
```
"""
    ANALIZE_IMAGE_PROMPT = """"Analyze the uploaded image for quality issues. Determine whether it contains noise or 
glitches, or if it is too dark or too bright. Based on the analysis, decide which of the following actions should be 
performed:

- Repair the image (REPAIR) if it contains noise or glitches.
- Brighten the image (BRIGHTEN) if it is too dark.
- Darken the image (DARKEN) if it is too bright.

In the response, return one of the commands (REPAIR, BRIGHTEN, DARKEN) with the filename, or 'OK' if no action is needed.
Example responses:

USER: Send image and image is fine
AI: OK
USER: Send image and the image needs repair
AI: REPAIR
USER: Send image and the image needs brightening
AI: BRIGHTEN
USER: Send image and the image needs darkening
AI: DARKEN
    
    """

    def __init__(self):
        self.model = OpenAi4oMini()

    def input(self, message):
        images = self._identify_images_url(message)
        new_images = []
        for image in images:
            new_images.append(self._fix_image(image))
        return new_images

    def _identify_images_url(self, message):
        try_count = 0
        system = create_message("system", PhotoRepair.UNDERSTAND_QUERY_TEXT)
        while True:
            try_count += 1
            if try_count > 5:
                raise Exception("To much count")
            try:
                result = self.model.chat([system, create_message("user", message)])
                return json.loads(extract_json(result))["images"]
            except:
                pass

    def _fix_image(self, image):
        system = create_message("system", PhotoRepair.ANALIZE_IMAGE_PROMPT)
        curr_image = image
        try_count = 0
        while True:
            try_count += 1
            if try_count > 10:
                raise Exception("Too much try for reair images")
            download_file(os.path.join("tmp", "s04e01"), curr_image)
            user = create_message_with_image("user", "",
                                             [os.path.join("tmp", "s04e01", curr_image.split('/')[-1])],
                                             self.model.formats)
            tool_result = self.model.chat([system, user])
            if tool_result.strip() == "OK":
                action_result = send_report("badgers")
                return curr_image
            action_result = send_report(tool_result + " " + curr_image.split('/')[-1])
            if action_result["code"] != 0:
                print(f"ERROR: {action_result}")
                continue
            next_msg = f"My previous image URL: {curr_image}. New url or filename is here: <new>{action_result['message']}</new> Join this information and return new image URL"
            curr_image = self._identify_images_url(next_msg)[0]
            pass


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )

    # URI examples: "neo4j://localhost", "neo4j+s://xxx.databases.neo4j.io"

    start = send_report("START")
    p = PhotoRepair()
    model = OpenAi4oMini()
    new_images = p.input(start["message"])
    images_path = []
    for image in new_images:
        images_path.append(os.path.join("tmp", "s04e01", image.split('/')[-1]))
    user_msg = create_message_with_image("user", ANALIZE_PERSON_MSG, images_path, model.formats)
    result = model.chat([user_msg], seed="1")
    pass
    answer = result
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="photos")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )

    answer = result + "\n cZY Opowiesz mi jakis dowcip?"
    print(answer )
    data = dict(answer=answer, apikey=mykey, task="photos")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)


if __name__ == '__main__':
    main()
