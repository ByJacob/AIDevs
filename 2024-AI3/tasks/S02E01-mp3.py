import glob
import re

import requests
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message, download_and_extract_zip

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    download_url = f"{domain}/dane/przesluchania.zip"
    directory_path = os.path.join("tmp", "s02e01")
    download_and_extract_zip(directory_path, download_url)
    m4a_files = glob.glob(os.path.join(directory_path, "**", "*.m4a"), recursive=True)
    for file in m4a_files:
        absolute_path = os.path.abspath(file)
        transcription_path = absolute_path.split(".")[0] + ".txt"
        if not os.path.exists(transcription_path):
            whisper_client = OpenAiWhisper()
            transcript = whisper_client.transcript(absolute_path)
            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(transcript)
    txt_files = glob.glob(os.path.join(directory_path, "**", "*.txt"), recursive=True)

    context = ""
    for txt in txt_files:
        name = os.path.basename(txt)
        with open(txt, "r", encoding="utf-8") as f:
            context += f"## {name}\n{f.read()}"

    system_message_txt = """You are a journalist. 
Based on the information of people in context, write all the information you know about the university and the department where Andrzej Maj teaches/works. 
Write the information out as a list
"""
    model = BielikP11B(debug=True)
    messages = [
        create_message("system", system_message_txt),
        create_message("user", f"<context>\n{context}\n</context> ")
    ]
    response = model.chat(messages)
    messages.append(create_message("assistant", response))
    messages.append(create_message("user", """Based on this information are you able to provide the name of the university? 
Justify your choice before giving the name
Return the name and department of the university in polish language in the form <university>...</university>.    
"""))
    response = model.chat(messages)
    pattern = r"<university>(.*?)</university>"
    # Search for the pattern in the input string
    match = re.search(pattern, response)
    university_name = match.group(1)
    model2 = OpenAi35Turbo(debug=True)
    model2_system_message = """
Zwracasz nazwę ulicy dla miejsca, które poda użytkownik. Nie generuj pełnych zdań. Zwracaj bardzo krótkie odpowiedzi.
Przykłady:

USER: Pałac Kultury i Nauki, Warszawa  
ASSISTANT: Plac Defilad  

USER: Sukiennice, Kraków  
ASSISTANT: Rynek Główny  

USER: Muzeum Narodowe, Warszawa  
ASSISTANT: Aleje Jerozolimskie  

USER: Centrum Nauki Kopernik, Warszawa  
ASSISTANT: Wybrzeże Kościuszkowskie  

USER: Zamek Królewski, Warszawa  
ASSISTANT: Plac Zamkowy  

USER: Wawel, Kraków  
ASSISTANT: Wawel  

USER: Opera Narodowa, Warszawa  
ASSISTANT: Plac Teatralny  
"""
    street = model2.chat([
        create_message("system", model2_system_message),
        create_message("user", university_name)
    ])
    data = dict(answer=street, apikey=mykey, task="mp3")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
