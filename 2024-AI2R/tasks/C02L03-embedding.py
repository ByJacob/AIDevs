from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import difflib

from base import BaseTasks


class Task(BaseTasks):
    def __init__(self):
        super().__init__("embedding")

    def resolve(self, task):
        params = task['msg'].replace("params: ", "\n").split("\n")[-1]
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        embedded_query = embeddings_model.embed_query(params)
        return embedded_query


if __name__ == "__main__":
    t = Task()
    t.process()
