import json
import os
import time

import requests
from langfuse.decorators import langfuse_context

from utils import download_and_extract_zip, create_message
from langfuse.openai import OpenAI

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')

langfuse_context.configure(
    secret_key=os.getenv('AIDEVS3_LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('AIDEVS3_LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('AIDEVS3_LANGFUSE_HOST'),
)

directory_path = os.path.join("tmp", "s04e02")

extracted_path = os.path.join(directory_path, "zip")
correct_path = os.path.join(extracted_path, "correct.txt")
incorrect_path = os.path.join(extracted_path, "incorrect.txt")
verify_path = os.path.join(extracted_path, "verify.txt")

def part1():
    download_url = f"{domain}/dane/lab_data.zip"
    download_and_extract_zip(directory_path, download_url)
    train_data = []
    for result, path in [("1", correct_path), ("0", incorrect_path)]:
        with open(path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                train_line = dict(messages=[create_message("user", line.strip()), create_message("assistant", result)])
                train_data.append(json.dumps(train_line) + "\n")
    pass
    train_data_path = os.path.join(extracted_path, "train.jsonl")
    with open(train_data_path, "w", encoding="utf-8") as f:
        f.writelines(train_data)

    client = OpenAI()
    files_list = list(map(lambda x: x.filename, client.files.list().data))

    if os.path.basename(train_data_path) not in files_list:
        client.files.create(
            file=open(train_data_path, "rb"),
            purpose="fine-tune"
        )

    file_id = list(filter(lambda x: x.filename == os.path.basename(train_data_path), client.files.list().data))[0].id

    def get_exist_job():
        return list(filter(lambda x: x.training_file == file_id, client.fine_tuning.jobs.list().data))

    exist_job = get_exist_job()

    if len(exist_job) <=0 :
        result = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-4o-mini-2024-07-18"
        )
    print("Wait for e-mail with finish job")
    return

    exist_job_id = get_exist_job()[0].id

def part2():
    client = OpenAI()
    answer = []
    with open(verify_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            id, msg = line.split("=")
            response = client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:agusiowo::AYGvLnOM",
                messages=[create_message("user", msg)]
            )
            odp = response.choices[0].message.content
            if "1" in odp:
                answer.append(id)
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="research")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)



if __name__ == '__main__':
    part2()
