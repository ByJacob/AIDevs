from base import BaseTasks
from openai import OpenAI

class Task(BaseTasks):
    def __init__(self):
        super().__init__("moderation")

    def resolve(self, task):
        client = OpenAI()
        results = []
        for text in task["input"]:
            response = client.moderations.create(input=text)
            results.append(int(response.results[0].flagged))
        return results


if __name__ == "__main__":
    t = Task()
    t.process()
