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
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from tqdm import tqdm

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        self.llm = OpenAI()
        super().__init__("gnome")

    def answer(self, image_url):
        prompt = """
        Czy na zdjęciu jest gnom lub skrzat? Jeżeli tak zwróć kolor jego czapki. Odpowiedź w formie JSONa. Nie formatuj odpowiedzi.

        Przykładowe odpowiedzi:
        - {"skrzat":true,"kolor_czapki":"niebieski"} - jeżeli na zdjeciu jest skrzat i ma niebieską czapkę
        - {"skrzat":true,"kolor_czapki":"zielony"} - jeżeli na zdjeciu jest gnom i ma zieloną czapkę
        - {"skrzat":true,"kolor_czapki":""} - jeżeli na zdjeciu jest gnom lub skrzat, ale nie ma czapki
        - {"skrzat":false,"kolor_czapki":""} - jeżeli na zdjeciu nie ma gnoma i skrzata
        """

        response = self.llm.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        question_category = json.loads(response.choices[0].message.content)

        if not question_category["skrzat"] or len(question_category["kolor_czapki"]) <= 0:
            ans = "error"
        else:
            ans = question_category["kolor_czapki"]

        return ans

    def resolve(self, task):
        ans = ""
        return self.answer(task["url"])


if __name__ == "__main__":
    t = Task()
    t.process()
