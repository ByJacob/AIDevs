import re
import os
import requests
from bs4 import BeautifulSoup
from models import *
from utils import find_flag, create_message

domain = os.getenv('XYZ_DOMAIN')

# model = Gemma2P2B(debug=False)
# model = Llama32P1B(debug=True)
model = MistralP7B(debug=True)
model.ping()

result = ""
msg_id = 0
human_text = "READY"

system_message_text = """Rules:
- Use only the information in the context to answer questions about it.
<context>
- stolicą Polski jest Kraków
- znana liczba z książki Autostopem przez Galaktykę to 69
- Aktualny rok to 1999
</context>
"""
system_message = create_message("system", system_message_text)
system_message2 = create_message("user", system_message_text)


while True:
    r = requests.post(f"{domain}/verify", json=dict(msgID=msg_id, text=human_text))
    response = r.json()
    if "code" in response or response.get("msgID") == 0:
        print("!"*5, "RESET", "!"*5)
        result = ""
        msg_id = 0
        human_text = "READY"
        continue
    result = response.get("text")
    if len(find_flag(result)) > 0:
        break
    msg_id = response.get("msgID")
    user_message = create_message("user", "I want response in English language. " + result)
    human_text = model.chat([
        system_message,
        system_message2,
        user_message,
        create_message("user", "Switch to English")
    ])
    pass
print("FLAG")
print(result)