import re
import os
import requests
from bs4 import BeautifulSoup
from models import *
from utils import find_flag

domain = os.getenv('XYZ_DOMAIN')
main_page = requests.get(domain)

soup = BeautifulSoup(main_page.text, features="html.parser")

question = soup.find("p", id="human-question").text.split(":")[-1]

# model = Gemma2P2B()
model = Llama32P8B(debug=True)
# model = Qwen25P3B(debug=True)
# model = MistralP7B(debug=True)
model.ping()

prompt = f"""You can return only year in number.
<question>
{question}
</question>
"""

messages = [
  {
    'role': 'user',
    'content': prompt,
  },
]

result = model.chat(messages)
result = re.sub(r"[^0-9]", "", result)

data = {
    'username': 'tester',
    'password': os.getenv('XYZ_PASSWORD'),
    'answer': result.strip(),
}

response = requests.post(domain, data=data)

flag = find_flag(response.text)

print(flag)