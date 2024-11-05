import json
import os
import tempfile

import requests
import requests_random_user_agent
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from openai import OpenAI
from tqdm import tqdm

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
        super().__init__("tools")

    def get_type(self, question):
        messages = [
            SystemMessage(content="""
            Dzisiaj jest 2024-05-13.
            Zkategoryzuj mi dane wejsciowe i ustaw odpowiedni output. Dane zwracasz w JSONie. 
            Masz 2 kategorie:

            # Calendar
            - ma zdefiniowany termin wykonania
            - ma określenia czasu takie jak 'jutro', 'pojutrze', 'we wtorek' itp.

            # ToDo
            - nie ma terminu wykonania

            Przykłady:
            - Przypomnij mi, że mam kupić mleko
            - {"tool":"ToDo","desc":"Kup mleko" }
            - Jutro mam spotkanie z Marianem
            - {"tool":"Calendar","desc":"Spotkanie z Marianem","date":"2023-11-15"}
            """),
            HumanMessage(content=question),
        ]
        output = self.llm.invoke(messages)
        return json.loads(output.content)

    def resolve(self, task):
        ans = ""
        question = task["question"]
        question_category = self.get_type(question)
        return question_category


if __name__ == "__main__":
    t = Task()
    t.process()
