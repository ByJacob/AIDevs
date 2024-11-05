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
        super().__init__("knowledge")

    def get_type(self, question):
        messages = [
            SystemMessage(content="""
           Zkategoryzuj mi dane wejsciowe i ustaw odpowiedni output. Dane zwracasz w JSONie
           Masz 3 kategorie:
           - "currency" - dla pytania o kurs walut, zwracasz z dodatkowym polem zawierajacym nazwe waluty
           - "population" - dla pytań o populacje kraju, zwracasz dodtkowo pole zawierające nazwe kraju
           - "other" - zapytania nie pasujace do pozostałych kategorii, W dodatkowym polu zwracasz odpowiedź na dane wejściowe
        
           Przykłady:
           - podaj mi aktualny kurs PLN
           - {"category":"currency","desc":"PLN"}
           - jaka jest populacja Polski
           - {"category":"population","desc":"Poland"}
           - Jaki kolor ma niebo
           - {"category":"other", "desc":"niebieski"}
           """),
            HumanMessage(content=question)
        ]
        output = self.llm.invoke(messages)
        return json.loads(output.content)

    def resolve(self, task):
        ans = ""
        question = task["question"]
        question_category = self.get_type(question)
        if question_category["category"] == "currency":
            res = requests.get("http://api.nbp.pl/api/exchangerates/tables/A/").json()[0]["rates"]
            ans = list(filter(lambda x: x["code"] == question_category['desc'], res))[0]["mid"]
            test = 123
        elif question_category["category"] == "population":
            res = requests.get(f"https://restcountries.com/v3.1/name/{question_category['desc']}").json()[0][
                "population"]
            test = 123
            ans = res
        else:
            ans = question_category["desc"]
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
