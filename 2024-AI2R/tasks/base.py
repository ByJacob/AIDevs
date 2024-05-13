import os
from abc import ABC, abstractmethod

import requests


class BaseTasks(ABC):

    def __init__(self, task_name):
        self.url = "https://tasks.aidevs.pl"
        self.api_key = os.getenv('AIDEV2_API_KEY')
        self.task_name = task_name
        self.token = None
        if self.api_key is None:
            raise ValueError("Require environment variable AIDEV2_API_KEY")

    def get_token(self):
        params = {"apikey": self.api_key}
        res = requests.post(f"{self.url}/token/{self.task_name}", json=params).json()
        self.token = res["token"]

    def task(self):
        res2 = requests.get(f"{self.url}/task/{self.token}")
        return res2.json()

    def task_post(self, data):
        res2 = requests.post(f"{self.url}/task/{self.token}", data=data)
        return res2.json()

    def send_answer(self, answer):
        data = dict(answer=answer)
        res3 = requests.post(f"{self.url}/answer/{self.token}", json=data)
        return res3.json()

    @abstractmethod
    def resolve(self, task):
        pass

    def process(self):
        self.get_token()
        task = self.task()
        print("*"*20)
        print("TASK")
        print(task)
        answer = self.resolve(task)
        print("*"*20)
        print(f"MY ANSWER: {answer}")
        answer_response = self.send_answer(answer)
        print("*" * 20)
        print("ANSWER RESPONSE")
        print(answer_response)
