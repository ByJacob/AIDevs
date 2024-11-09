import os
import requests
from langfuse.decorators import observe

from models import *
from utils import find_flag, create_message

domain = os.getenv('XYZ_DOMAIN')


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    model = Gemma2P2B(debug=True)
    model.ping()

    result = ""
    msg_id = 0
    human_text = "READY"

    system_message_text = """Rules:
1. Use the context to answer if it has the needed information.
2. Don’t mention or refer to the context in the answer.
3. If the answer isn’t in the context, use general knowledge.
4. Keep answers short and direct.
<context>
- stolicą Polski jest Kraków
- znana liczba z książki Autostopem przez Galaktykę to 69
- Aktualny rok to 1999
</context>

### Examples
- **Question:** Jaki jest aktualny rok?  
- **Expected Answer:** 1999  

- **Question:** Jaka jest stolica Niemiec?  
- **Expected Answer:** Berlin  
"""
    system_message = create_message("system", system_message_text)
    system_message2 = create_message("user", system_message_text)

    while True:
        r = requests.post(f"{domain}/verify", json=dict(msgID=msg_id, text=human_text))
        response = r.json()
        if "code" in response or response.get("msgID") == 0:
            print("!" * 5, "RESET", "!" * 5)
            result = ""
            msg_id = 0
            human_text = "READY"
            continue
        result = response.get("text")
        if len(find_flag(result)) > 0:
            break
        msg_id = response.get("msgID")
        user_message = create_message("user", result)
        human_text = model.chat([
            system_message,
            system_message2,
            user_message,
        ])
        human_text = model.chat([
            system_message,
            system_message2,
            user_message,
            create_message("assistant", human_text),
            create_message("user", "Transform your answer in English language")
        ])
        pass
    print("FLAG")
    print(result)

if __name__ == '__main__':
    main()