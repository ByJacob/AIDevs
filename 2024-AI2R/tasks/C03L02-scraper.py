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
        super().__init__("scraper")

    def get_url(self, url):
        response_code = 0
        res = None
        while response_code != 200:
            res = requests.get(url)
            response_code = res.status_code
        return res.text

    def answer_question(self, question, content):
        system_template = "Answer to question using givent. Prepare simple response in Polish language.\n content\n ```content\n{content}```"
        human_template = "{question}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(question=question, content=content)
        output = self.llm.invoke(prompt)
        return output.content

    def resolve(self, task):
        url = task["input"]
        question = task["question"]
        text = self.get_url(url)
        ans = self.answer_question(question, text)
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
