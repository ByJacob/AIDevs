import os
from abc import ABC, abstractmethod

import requests


class BaseTasks(ABC):

    def __init__(self, task_name):
        self.url = "https://poligon.aidevs.pl"
        self.api_key = os.getenv('AIDEV3_API_KEY')
        self.task_name = task_name
        if self.api_key is None:
            raise ValueError("Require environment variable AIDEV2_API_KEY")

    def send_answer(self, answer):
        data = dict(answer=answer, apikey=self.api_key, task=self.task_name)
        res3 = requests.post(f"{self.url}/verify", json=data)
        return res3.json()

    @abstractmethod
    def answer(self):
        pass

    def process(self):
        answer = self.answer()
        print("*" * 20)
        print(f"TASK: {self.task_name}")
        print("*"*20)
        print(f"MY ANSWER: {answer}")
        answer_response = self.send_answer(answer)
        print("*" * 20)
        print("ANSWER RESPONSE")
        print(answer_response)
