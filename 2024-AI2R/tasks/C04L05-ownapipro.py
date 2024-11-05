import json
import os
import tempfile
import threading

import requests
import requests_random_user_agent
from flask import app, request, Flask
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from serpapi import GoogleSearch
from tqdm import tqdm
from threading import Thread

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS


def ask_serpapi(query):
    api_key = os.environ.get('SERPAPI_KEY')
    serpapi_params = {
        "q": "Coffee",
        "location": "Poland",
        "hl": "pl",
        "gl": "pl",
        "google_domain": "google.com",
        "api_key": api_key
    }
    local_param = serpapi_params.copy()
    local_param["q"] = query
    search = GoogleSearch(local_param)
    results = search.get_dict()
    return results["organic_results"][0]["link"]


system_message = """
    Dzisiaj jest 2024-05-13.
    Odpowiedz krótko na podane pytanie.
    W przypadku porównań odpowiedz tylko jednym słowem.
    Do odpowiedzi użyj poniższych informacji:
    
    context```
    {context}
    ```
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", '{question}'),
])

def ask_llm(query, local_information, cp=chat_prompt):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    context = "- " + "\n - ".join(local_information)
    answer = llm.invoke(cp.format_messages(question=query, context=context)).content
    return answer


app = Flask(__name__)

information = []

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def catch_all(path):
    params = request.json if request.is_json else request.args
    app.logger.info(f'Przychodzi request do endpointu /{path} z parametrami: {dict(params)}')

    if request.method == 'POST' and request.is_json:
        question = request.json["question"]

        app.logger.info(f'Question: {question}')
        answer = ask_llm(question, information)
        app.logger.info(f'Answer: {answer}')
        information.append(question)
        return {"reply": answer}

    return 'Dane zostały zalogowane!'


class Task(BaseTasks):
    def __init__(self):
        self.llm = OpenAI()
        super().__init__("ownapipro")

    def start_server(self):
        # cloudflared tunnel --url http://localhost:3000
        def web():
            app.run(debug=True, use_reloader=False, host='0.0.0.0', port=3000)
        threading.Thread(target=web, daemon=True).start()

    def resolve(self, task):
        ans = ""
        self.start_server()
        return "https://walk-ram-orlando-johnson.trycloudflare.com"


if __name__ == "__main__":
    t = Task()
    t.process()
