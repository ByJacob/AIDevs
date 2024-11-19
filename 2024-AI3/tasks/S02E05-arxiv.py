import json

import pyhtml2md
import pytesseract
import requests

from models import *
from utils import *
from prompts import *

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

system_image = create_message("system", system_recognize_image_txt)


def extract_md_link(text):
    # Find the position of the first '{' and the last '}'
    left = text.find('(')
    right = text.rfind(')')

    # Return the content between the braces if both are found
    if left != -1 and right != -1:
        return text[left + 1:right]
    return None


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    tmp_path = os.path.join("tmp", script_name)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    questions_url = f"{domain}data/{mykey}/arxiv.txt"
    article_url = f"{domain}dane/arxiv-draft.html"

    article = requests.get(article_url)
    pictures = {}
    markdown = pyhtml2md.convert(article.text)
    images = []
    mp3s = []
    for line in markdown.splitlines():
        url = extract_md_link(line)
        if url is not None:
            url = f"{domain}dane/{url}"
        if '![' in line:
            images.append(url)
        if '.mp3)' in line:
            mp3s.append(url)
    model = OpenAi4o(debug=True)
    media_transcripts = {}
    for image in images:
        basename = image.split("/")[-1]
        transcription_path = os.path.join(tmp_path, basename + ".transcription")
        if not os.path.exists(transcription_path):
            user_txt = f"""Describe the image ${basename} concisely. 
Focus on the main elements and overall composition. 
Return the result in JSON format with only 'name' and 'preview' properties.
Return town name if in picture is town square.

Photos are from the work of Prof. Andrzej Maj from the Department of Mathematics and Computer Science 
at Jagiellonian University in Krakow, Poland.
"""
            download_file(tmp_path, image)
            user_msg = create_message_with_image("user", user_txt, [os.path.join(tmp_path, basename)], model.formats)
            describe = None
            while describe is None:
                describe = model.chat([system_image, user_msg])
                describe = extract_json(describe)
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(describe)
        with open(transcription_path, "r", encoding="UTF-8") as f:
            media_transcripts[basename] = f.read()
    for mp3 in mp3s:
        basename = mp3.split("/")[-1]
        transcription_path = os.path.join(tmp_path, basename + ".transcription")
        if not os.path.exists(transcription_path):
            download_file(tmp_path, mp3)
            whisper_client = OpenAiWhisper()
            transcript = whisper_client.transcript(os.path.join(tmp_path, basename), language="pl")
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(transcript)
        with open(transcription_path, "r", encoding="UTF-8") as f:
            media_transcripts[basename] = f.read()

    new_markdown = ""
    for line in markdown.splitlines():
        for media, media_description in media_transcripts.items():
            if media in line:
                desc = ""
                if ".png" in media:
                    desc += f"\nImage description: "
                    desc += f"{json.loads(media_description.replace("'", '"'))['preview']}. "
                if ".mp3" in media:
                    desc += f"Audio transcription: "
                    desc += f"{media_description}. "
                new_markdown += f"{desc}"
                break
        new_markdown += f"{line}\n"

    my_system_question_prompt = system_answer_question_from_context.replace("CONTEXT_PLACEHOLDER", new_markdown)
    system_question = create_message("system", f"{my_system_question_prompt}. Prepare answer in Polish language.")
    question_dirt = requests.get(questions_url).text
    answers = {}
    model_answer = OpenAi4o(debug=False)
    aidevs_answer = {}
    for line in question_dirt.splitlines():
        id, question = line.split('=')
        response = model_answer.chat([system_question, create_message("user", question)])
        answers[id] = (question, response)
        aidevs_answer[id] = json.loads(response)["answer"]
    pass

    print(aidevs_answer)
    data = dict(answer=aidevs_answer, apikey=mykey, task="arxiv")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
