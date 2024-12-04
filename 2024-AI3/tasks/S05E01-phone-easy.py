import glob
import re

import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image, extract_json
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')

keyword_system_msg = """
**Prompt:**  
You are an advanced text analysis tool. Your task is to generate a set of the most important keywords based on the provided conversation text. Carefully analyze the content, focusing on:

1. **Main topics and themes** – Identify the core subjects discussed in the text.  
2. **Repeated words and phrases** – Highlight significant, frequently used terms and expressions.  
3. **Concepts and ideas** – Capture the primary concepts and ideas present in the conversation.  
4. **Important proper nouns** – Include key names, companies, places, or product names mentioned.  
5. **Tone and context** – Determine whether the conversation relates to technology, health, business, or any other field.  

Based on this analysis, return the output in JSON format with the following fields:
- **_thinking**: A brief explanation of the rationale behind selecting these keywords.  
- **keywords**: A list of 5 to 15 keywords that best represent the text's content and tone.


**Expected JSON output:**  
{
  "_thinking": "Explanation of how the keywords were derived, mentioning key themes and repeated concepts.",
  "keywords": ["keyword1", "keyword2", "keyword3", ..., "keywordN"]
}
"""

@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    facts_path = os.path.join("tmp", "s03e01", "facts")

    facts = []

    for file_name in os.listdir(facts_path):
        with open(os.path.join(facts_path, file_name), "r", encoding="UTF-8") as f:
            content = f.read()
            if "entry deleted" not in content:
                facts.append(content)
    pass

    phone_url = f"{domain}/data/{mykey}/phone_sorted.json"
    phone_question_url = "{domain}/data/{mykey}/phone_questions.json"

    phone_data = requests.get(phone_url).json()

    talks = {}
    model = OpenAi4oMini()
    for talk_id, talk in phone_data.items():
        sentences = ""
        for idx, line in enumerate(talk):
            person_id = 1 + idx%2
            sentences += f"<person{person_id}>{line}</person{person_id}>\n        "
        user_msg = f"""
<data>
    <talk_id>{talk_id}</talk_id>
    <sentences>
        {sentences}
    </sentences>
</data>        
        """
        result = model.chat([create_message("system", keyword_system_msg), create_message("user", user_msg)])
        keywords = json.loads(extract_json(result))["keywords"]
        talks[talk_id] = keywords
    pass


    # print(answer)
    # data = dict(answer=answer, apikey=mykey, task="dokumenty")
    # answer_result = requests.post(f"{domain}/report", json=data)
    # print(answer_result.text)
    # flag = find_flag(answer_result.text)
    # print(flag)
    # langfuse_context.update_current_observation(
    #     output=flag
    # )


if __name__ == '__main__':
    main()
