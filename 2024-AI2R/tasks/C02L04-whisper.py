import tempfile

import requests
import requests_random_user_agent

from openai import OpenAI

from base import BaseTasks

_ = requests_random_user_agent.USER_AGENTS

class Task(BaseTasks):
    def __init__(self):
        super().__init__("whisper")

    def save_audio_file(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            return temp_file_path
        else:
            # Handle the case when the request fails
            print("Failed to download the audio file.")
            return None

    def resolve(self, task):
        client = OpenAI()

        url = task['msg'].replace("file: ", "\n").split("\n")[-1]
        file_name = self.save_audio_file(url)
        audio_file = open(file_name, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcription.text


if __name__ == "__main__":
    t = Task()
    t.process()
