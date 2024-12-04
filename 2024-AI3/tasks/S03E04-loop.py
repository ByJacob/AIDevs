import glob
import json
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


def send_people(person, extra="people"):
    apidb = f"{domain}/{extra}"
    body = {
        "apikey": mykey,
        "query": person.replace("Ł", "L")
    }
    result = requests.post(apidb, json=body)
    return result.json()

def send_places(place):
    return send_people(place, extra="places")



@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )

    system_msg_text = """
You return only names from messages from users

USER: Pije kawę z Janem Kowalskim i Elizą Przybysz
AI: Jan Eliza

USER: Widziałem Przemka Kosa na przystanku, rozmawiał z Adamem Burym i Staśkiem Grzędą
AI: Przemek Adam Stanisław
"""
    system_msg = create_message('system', system_msg_text)
    model = OpenAi4oMini(debug=True)
    barbara_txt = requests.get(f"{domain}/dane/barbara.txt")
    messages = [system_msg, create_message('user', barbara_txt.text)]
    response = model.chat(messages)

    global_users = {}
    global_places = {}

    for user in response.split(" "):
        if user == "Barbara":
            continue
        global_users[user.strip().upper()] = []

    for i in range(10):
        for user, cites in global_users.copy().items():
            cites_txt = send_people(user)["message"]
            if cites_txt == "[**RESTRICTED DATA**]":
                continue
            if "https" in cites_txt:
                continue
            if "data" in cites_txt or "query" in cites_txt:
                pass
            for city in cites_txt.split(" "):
                global_users[user].append(city)
                if city not in global_places:
                    global_places[city] = [user]
                elif user not in global_places[city]:
                    global_places[city].append(user)
            global_users[user] = list(set(global_users[user]))
        for city, users in global_places.copy().items():
            people_txt = send_places(city)["message"]
            if people_txt == "[**RESTRICTED DATA**]":
                continue
            if "https" in people_txt:
                continue
            if "data" in people_txt or "query" in people_txt:
                pass
            for user in people_txt.split(" "):
                global_places[city].append(user)
                if user not in global_users:
                    global_users[user] = [city]
                elif city not in global_users[user]:
                    global_users[user].append(city)
            global_places[city] = list(set(global_places[city]))
        if len(global_users["BARBARA"]) > 2:
            break
    answer = global_users["BARBARA"][1]
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="photos")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
