import glob
import json
import re

import pyhtml2md
import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image, extract_json
from tqdm import tqdm

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')


class PageSearch():

    def __init__(self):
        self.urls = ["https://softo." + domain.split(".", 1)[-1]]
        self.model = OpenAi4oMini()

    def input(self, question):
        answer = self._search_answer(question)
        return answer

    def _system_answer_question(self, markdown):
        return create_message("system",f"""
You are an expert AI system tasked with extracting information from a webpage. Given a user's question and the content of the page, follow these steps:

1. Analyze the content for relevant information.
2. If you find the answer, explain your reasoning.
3. If you cannot find the answer, explain why.

Return your response as a JSON object with three fields:
- `_thinking`: Your detailed explanation of how you processed the information.
- `answer`: The direct answer to the user's question, or `null` if not found.

<page_content>
{markdown}
</page_content>
""")

    def _describe_all_links(self, link):
        return create_message("system", f"""
Given the markdown code provided, parse and identify all links. For each link, do the following:

Actual link: {link}

1. If the link is relative (e.g., `./path/to/file`), convert it to an absolute URL by appending it to the base URL (for example, 'https://example.com').
2. For each link (whether originally absolute or converted from relative), return a description with the following details:
   - **Link URL**: The absolute URL of the link.
   - **Link Title**: The text that appears as the clickable link (the anchor text). If there is no anchor text, return 'No title' or 'No text'.
   - **Link Analysis**: Analyze the URL structure and explain its components. This may include:
     - **Target page content prediction**: Provide an educated guess about the content or purpose of the target page based on the URL's structure. For example:
       - A URL with `/about` might lead to an "About Us" page.
       - A URL with `/products` might point to an e-commerce product page.
       - A URL with `/search` and query parameters might point to a search results page.
       - A URL with `/aktualnosci` can have information about awards and acquired standards and certificates
   
Ignore any non-HTTP-based links (e.g., mailto, tel, etc.).

Example output:

{{
  "links": [{{
    "Link URL": "https://example.com",
    "Link Title": "Example Website",
    "Link Analysis": {{
      "Target Page Content Prediction": "The target page is likely the homepage or landing page of the website."
    }}
  }},
  ...
  ]
}}

""")
    def _find_interesting_link(self, actual_link, markdown, link_description):
        return create_message("system", f"""
**System Role Description:**  
You are an assistant that analyzes a given web page, identifies relevant links based on a user's question, 
and provides structured JSON responses. Your goal is to ensure the chosen link directly or indirectly addresses the user's query.
Don't return actual link in response.

Actual URL: {actual_link}

### **Instructions:**  

1. **Input:** 
   - **User Question:** The specific information the user is seeking.  

2. **Tasks:**  
   - Analyze the links_descriptions. 
   - Identify the most relevant links based on the user's question.  
   - If the link is related to the user's query but not an exact match still consider it as a valid link.  
   - Convert any relative links to absolute URLs using the provided base URL.  
   - Provide a clear explanation of why the links was chosen.  
   - based on information from links_descriptions also return links that may contain information about the user's question
   
3. **Output Format:**  
   Respond with a JSON object containing two fields:  
   ```json
   {{
     "_thinking": "<Your explanation for selecting the link or the base URL>",
     "links": ["link1", "link2", ...]
   }}

<links_descriptions>
{json.dumps(link_description["links"])}
</links_descriptions>
   """)

    @staticmethod
    def _download_page(link):
        text = requests.get(link).text
        return pyhtml2md.convert(text)

    def _search_answer(self, question):
        answer = None
        try_count = 0
        links = self.urls
        while answer is None:
            try_count += 1
            if try_count > 10:
                raise Exception("Cannot search answer for qiven question")
            new_links = []
            for link in links.copy():
                markdown = self._download_page(link)
                messages = [self._system_answer_question(markdown), create_message("user", question)]
                result = self._return_json(messages)
                if result["answer"] is None:
                    messages2 = [self._describe_all_links(link), create_message("user", markdown)]
                    result2 = self._return_json(messages2)
                    messages3 = [self._find_interesting_link(link, markdown, result2), create_message("user", question)]
                    result3 = self._return_json(messages3)
                    new_links += result3["links"]
                    if link in new_links:
                        new_links.remove(link)
                else:
                    answer = str(result["answer"])
                    break
            links = new_links
        return answer

    def _return_json(self, messages):
        try_count = 0
        while True:
            try_count += 1
            if try_count > 10:
                raise Exception("Cannot return JSON object")
            try:
                result = self.model.chat(messages)
                obj = json.loads(extract_json(result))
                return obj
            except Exception as e:
                pass




@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )

    questions = requests.get(f"{domain}/data/{mykey}/softo.json").json()
    answer = {}
    for idx, q in questions.items():
        ps = PageSearch()
        result = ps.input(q)
        print(f"{q}: {result}")
        answer[idx] = result


    print(answer)
    data = dict(answer=answer, apikey=mykey, task="softo")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
