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
        self.embedding_function = OpenAIEmbeddings()
        super().__init__("search")
        self.db = None

    def get_url(self, url):
        response_code = 0
        res = None
        while response_code != 200:
            res = requests.get(url)
            response_code = res.status_code
        return res.json()

    def load_vector_db(self, data):
        index_file = "tmp/c03l04-faiss_index"
        if not os.path.exists(index_file):
            for d in tqdm(data, desc="vector data"):
                db1 = FAISS.from_texts([d["title"]], self.embedding_function, metadatas=[{"msg": d["info"], "url": d["url"]}])
                if self.db is None:
                    self.db = db1
                else:
                    self.db.merge_from(db1)
            self.db.save_local(index_file)
        self.db = FAISS.load_local(index_file, self.embedding_function, allow_dangerous_deserialization=True)
        print(self.db.index.ntotal)
        print("Finish load DB")

    def answer_question(self, question):
        docs = self.db.similarity_search_with_score(question)
        return docs[0][0].metadata["url"]

    def resolve(self, task):
        ans = ""
        url = task["msg"].split(" - ")[-1]
        question = task["question"]
        data = self.get_url(url)
        self.load_vector_db(data)
        ans = self.answer_question(question)
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
