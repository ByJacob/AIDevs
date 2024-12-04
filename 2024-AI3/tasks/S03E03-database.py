import glob
import json
import re

import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


def send_query(query):
    apidb = f"{domain}/apidb"
    body = {
        "task": "database",
        "apikey": mykey,
        "query": query
    }
    result = requests.post(apidb, json=body)
    return result.json()

@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )

    system_msg_text = """Your task is to help me create an SQL query based on my question. 
You can only perform operations like: select, show tables, desc table, show create table
We'll proceed step by step:
1. Start with `SHOW TABLES` to provide me with a list of available tables.
2. Based on my description, we'll identify which tables are relevant.
3. List all columns in the selected tables using `DESCRIBE [table_name]` to understand their structure.
4. Base on user question select important queries.
5. Assist in defining filters, sorting, grouping, or joins (if needed).
6. Build the final SQL query step by step.
7. After finish job in query return `EXIT`.

Provide your responses in JSON format with the following structure:

- `_thinking`: A brief explanation of why you chose the particular SQL query or command.
- `query`: The SQL query or command you are suggesting for the next step.

### Examples:

USER: Którzy pracownicy są w pracy (is_active=1)
AI: {"thinking_": "First i need check which databases exist in server", "query": "SHOW TABLES"}
USER: [{"Tables_in_banan":"connections"},{"Tables_in_banan":"correct_order"},{"Tables_in_banan":"datacenters"},{"Tables_in_banan":"users"}]
AI: {"thinking_": "Question is about user, so first check structure for user table", "query": "SHOW TABLES"}
USER: [{"Table":"users","Create Table":"CREATE TABLE `users` (\n  `id` int(11) NOT NULL AUTO_INCREMENT,\n  `username` varchar(20) DEFAULT NULL,\n  `access_level` varchar(20) DEFAULT 'user',\n  `is_active` int(11) DEFAULT 1,\n  `lastlog` date DEFAULT NULL,\n  PRIMARY KEY (`id`)\n) ENGINE=InnoDB AUTO_INCREMENT=98 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci"}]
...
    
"""
    answer = []
    system_msg = create_message('system', system_msg_text)
    model = OpenAi4oMini(debug=True)
    user_msg_text = "które aktywne datacenter (DC_ID) są zarządzane przez pracowników, którzy są na urlopie (is_active=0)"
    messages = [system_msg, create_message('user', user_msg_text)]
    error_count = 0
    response_query = None
    while True:
        response = model.chat(messages)
        try:
            response_obj = json.loads(response)

            messages.append(create_message("assistant", response))
            if response_query is not None and response_obj["query"] == "EXIT":
                for item in response_query["reply"]:
                    answer.append(item["dc_id"])
                break
            response_query = send_query(response_obj["query"])
            if response_query["error"] != "OK":
                raise Exception(response)
            messages.append(create_message("user", json.dumps(response_query["reply"])))
        except Exception as e:
            error_count += 1
            print(e)
            if error_count > 5:
                raise Exception("Cannot resolve this error, check logs")

    print(answer)
    data = dict(answer=answer, apikey=mykey, task="database")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
