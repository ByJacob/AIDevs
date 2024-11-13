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
    model = BielikP11B(debug=True)

    system_message_text = """You are the censor and are tasked with returning the user's opinion, but in a censored form according to the rules.
    
##Rules:
1. **Replace sensitive information**:
   - Full names (first + last), street name + number, city, and person's age → **CENZURA**
2. **Maintain all punctuation and spacing** exactly as in the original text.
3. **Do not rephrase or alter the text** beyond replacing sensitive data.
    
## Examples
User: Tożsamość podejrzanego: Michał Wiśniewski. Mieszka we Wrocławiu na ul. Słonecznej 20. Wiek: 30 lat.
AI: Tożsamość podejrzanego: CENZURA. Mieszka we CENZURA na ul. CENZURA. Wiek: CENZURA lat.

USER: Osoba podejrzana to Andrzej Mazur. Adres: Gdańsk, ul. Długa 8. Wiek: 29 lat.
AI: Osoba podejrzana to CENZURA. Adres: CENZURA, ul. CENZURA. Wiek: CENZURA lat.

USER: Świadek: Anna Kowalska, zamieszkała w Krakowie, ul. Kwiatowa 15. Wiek świadka: 45 lat.
AI: Świadek: CENZURA, zamieszkała w CENZURA, ul. CENZURA. Wiek świadka: CENZURA lat.

USER: Zgłaszający to Piotr Nowak z Warszawy, ul. Lipowa 32. Ma 52 lata.
AI: Zgłaszający to CENZURA z CENZURA, ul. CENZURA. Ma CENZURA lata.

USER: Świadectwo wystawił Janusz Malinowski, zamieszkały w Poznaniu, ul. Różana 5. Ma 60 lat.
AI: Świadectwo wystawił CENZURA, zamieszkały w CENZURA, ul. CENZURA. Ma CENZURA lat.

USER: Pani Barbara Zielińska mieszka na ul. Jasna 10, w Łodzi. Ma 41 lat.
AI: Pani CENZURA mieszka na ul. CENZURA, w CENZURA. Ma CENZURA lat.

USER: Klient: Tomasz Kaczmarek, ul. Szeroka 7, Katowice. Wiek: 37 lat.
AI: Klient: CENZURA, ul. CENZURA, CENZURA. Wiek: CENZURA lat.
"""
    system_message = create_message("system", system_message_text)

    data_url = f"{domain}/data/{mykey}/cenzura.txt"

    response = requests.get(data_url)
    data = response.text

    messages = [system_message, create_message("user", data)]
    llm_response = model.chat(messages)

    data = dict(answer=llm_response, apikey=mykey, task="CENZURA")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )



if __name__ == '__main__':
    main()
