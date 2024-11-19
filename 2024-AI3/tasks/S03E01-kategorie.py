import glob
import re

import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    download_url = f"{domain}/dane/pliki_z_fabryki.zip"
    directory_path = os.path.join("tmp", "s03e01")
    facts_path = os.path.join(directory_path, "facts")
    download_and_extract_zip(directory_path, download_url)

    # Iterate through files in the directory

    facts = []

    for file_name in os.listdir(facts_path):
        with open(os.path.join(facts_path, file_name), "r", encoding="UTF-8") as f:
            content = f.read()
            if "entry deleted" not in content:
                facts.append(content)
    pass

    system_msg = f"""<document> 
{"###".join(facts)}
</document> 

Analyze the provided document and the content sent by the user. Based on these, generate a list of 
keywords in Polish that include: the person's name, his profession, his skills, the programming languages he/she uses, what the person did, the special elements of the event.
In count words, add the programming languages spoken by the person mentioned.
Include places where human fingerprints presences were found.
Keywords should be clear, precise, contextually related to both the document and the content, and as concise as possible. 
As concise as possible. Include important phrases and technical terms where appropriate. 

Example:

<document> Jan Kowalski jest programistą pythona i javy</document>
USER: Widziano na skraju lasu Jana Kowalskiego
AI: las, Jan, Kowalski, programista, python, java

<document> Jan Kowalski jest nauczycielem</document>
USER: Widziano na skraju lasu Jana Kowalskiego
AI: las, Jan, Kowalski, nauczyciel

Response format: comma-delimited word list, example key1, key2, key3,..."""
        # Check if the file has the correct extension

    model = OpenAi4oMini(debug=False)
    system_msg = create_message("system", system_msg)
    answer = {}
    for file in os.listdir(directory_path):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(directory_path, file), "r", encoding="UTF-8") as f:
            content = f.read()
        sektor = file.split("-")[-1].split(".")[0]
        result = model.chat([system_msg, create_message('user', content)])
        answer[file] = result + ", " + sektor.replace("_", " ")
        pass

    print(answer)
    data = dict(answer=answer, apikey=mykey, task="dokumenty")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
