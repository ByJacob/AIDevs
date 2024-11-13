import requests
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    model = Gemma2P2B(debug=False)
    model.ping()

    data_url = f"{domain}/data/{mykey}/json.txt"

    response = requests.get(data_url)
    data = response.json()

    data["apikey"] = mykey

    system_message_text = """Rules:
1. Don’t mention or refer to the context in the answer.
2. Keep answers short and direct. 

- **Question:** Jaka jest stolica Niemiec?  
- **Expected Answer:** Berlin  
"""
    system_message = create_message("system", system_message_text)
    for item in data["test-data"]:
        item["answer"] = eval(item["question"])
        if "test" in item:
            messages = [system_message, create_message("user", item["test"]["q"])]
            llm_response = model.chat(messages)
            item["test"]["a"] = llm_response

    data = dict(answer=data, apikey=mykey, task="JSON")
    answer_result = requests.post(f"{domain}/report", json=data)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )



if __name__ == '__main__':
    main()
