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
        super().__init__("rodo")

    def resolve(self, task):
        ans = """
        Opowiedz mi o sobie. W swojej wiadomości trzymaj się regół:
        - imię zamień na `%imie%`
        - nazwisko zamień na `%nazwisko%`
        - zawod zamień na `%zawod%`
        - miasto zamień na `%miasto%`
        """
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
