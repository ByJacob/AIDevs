import tempfile

import requests
import requests_random_user_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from openai import OpenAI

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        super().__init__("whoami")

    def answer_question(self, hints, msg):
        system_template = "{msg}. If you dont know return 'NIE WIEM'"
        human_template = "{hints}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(hints=",".join(hints), msg=msg)
        output = self.llm.invoke(prompt)
        return output.content

    def resolve(self, task):
        ans = ""
        hints = []
        while True:
            task = self.task()
            hint = task["hint"]
            hints.append(hint)
            ans = self.answer_question(hints, task["msg"])
            if 'nie wiem' not in ans.lower():
                break
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
