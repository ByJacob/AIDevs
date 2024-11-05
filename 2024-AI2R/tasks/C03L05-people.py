import json
import os
import tempfile

import requests
import requests_random_user_agent
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from openai import OpenAI
from tqdm import tqdm

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        super().__init__("people")

    def get_url(self, url):
        response_code = 0
        res = None
        while response_code != 200:
            res = requests.get(url)
            response_code = res.status_code
        return res.json()

    def get_name_surname(self, question):
        system_template = "Return only name and surname from question."
        human_template = "{question}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(question=question)
        output = self.llm.invoke(prompt)
        return output.content

    def answer(self, data, name_surname, question):
        parts = name_surname.lower().split(" ")
        context = list(filter(lambda x: x["imie"].lower() == parts[0] and x["nazwisko"].lower() == parts[1], data))
        system_template = "Answer for question with context.\n context```{context}```"
        human_template = "{question}"
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", human_template),
        ])
        prompt = chat_prompt.format_messages(question=question, context=json.dumps(context))
        output = self.llm.invoke(prompt)
        return output.content

    def resolve(self, task):
        ans = ""
        url = task["data"]
        data = self.get_url(url)
        name_surname = self.get_name_surname(question=task["question"])
        return self.answer(data, name_surname, task["question"])


if __name__ == "__main__":
    t = Task()
    t.process()
