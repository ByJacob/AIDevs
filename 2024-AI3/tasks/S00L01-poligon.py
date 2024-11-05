from base import BaseTasks

import requests
import requests_random_user_agent

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        super().__init__("POLIGON")

    def answer(self):
        r = requests.get(f"{self.url}/dane.txt").text
        return r.strip().split("\n")


if __name__ == "__main__":
    t = Task()
    t.process()
