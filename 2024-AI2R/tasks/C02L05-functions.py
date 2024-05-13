import tempfile

import requests
import requests_random_user_agent

from openai import OpenAI

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        super().__init__("functions")

    def resolve(self, task):
        ans = {
            "name": "addUser",
            "description": "Add user",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name"
                    },
                    "surname": {
                        "type": "string",
                        "description": "Surname"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year"
                    }
                }
            }
        }
        return ans


if __name__ == "__main__":
    t = Task()
    t.process()
