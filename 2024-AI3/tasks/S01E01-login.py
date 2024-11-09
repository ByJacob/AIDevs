import re
import os
import requests
from bs4 import BeautifulSoup
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_observation(
        name=script_name
    )

    domain = os.getenv('XYZ_DOMAIN')

    # model = Gemma2P2B()
    # model = Llama32P1B()
    # model = Llama32P8B(debug=False)
    # model = Qwen25P3B(debug=True)
    model = MistralP7B(debug=True)
    model.ping()
    count = 0
    while True:
        count += 1
        print(f"count: {count}")
        response = try_login(domain, model)

        if len(find_flag(response.text)) > 0:
            break
        if count > 10:
            raise Exception("cannot get good answer")

    flag = find_flag(response.text)

    langfuse_context.update_current_observation(
        output=flag
    )

    print(flag)


@observe(capture_input=False, capture_output=False)
def try_login(domain, model):
    main_page = requests.get(domain)
    soup = BeautifulSoup(main_page.text, features="html.parser")
    question = soup.find("p", id="human-question").text.split(":")[-1]
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
    result = model.chat(messages + [
        create_message("assistant", result),
        create_message("user", "Get year from previous message")
    ])
    data = {
        'username': 'tester',
        'password': os.getenv('XYZ_DOMAIN_PASSWORD'),
        'answer': result.strip(),
    }
    response = requests.post(domain, data=data)
    langfuse_context.update_current_observation(
        input=question,
        output=result
    )
    return response


if __name__ == '__main__':
    main()
