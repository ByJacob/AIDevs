import glob
import json
import re

import cv2
import fitz
import numpy as np
import pytesseract
import requests
from PIL import Image
from langfuse.decorators import observe
from thefuzz import fuzz

from models import *
from utils import find_flag, create_message, download_and_extract_zip, create_message_with_image, download_file, extract_json
from tqdm import tqdm
from prompts import ocr_correct_system_message

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')

directory_path = os.path.join("tmp", "s05e01")


@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    phone_url = f"{domain}/data/{mykey}/phone.json"
    phone_question_url = "{domain}/data/{mykey}/phone_questions.json"

    phone_data = requests.get(phone_url).json()

    all_sentence = sorted(list(map(lambda x: x.replace("\xc2\xa0", " ").replace("\xa0", " "), phone_data["reszta"])))

    sorted_keys = sorted(filter(lambda x: "length" in phone_data[x], phone_data), key=lambda x: phone_data[x]["length"])
    sorted_keys.remove("rozmowa2")
    sorted_keys.append("rozmowa2")

    system_msg = """
[prompt_objective]
Identify and analyze potential next sentences from a provided subset called "Other Sentences" that maintain coherence, logical progression, and relevance in a two-person conversation with each person speaking alternately.

<prompt_rules>
- ONLY analyze and suggest sentences from the "Other Sentences" list.
- For each sentence, RETURN a JSON object with the fields:
  - "_thinking_current": Analysis of the sentence's relevance considering the current talk's theme, logical flow, and coherence.
  - "_analize": Evaluate how the sentence matches the `current_talk` and its potential impact on the conversation.
  - "fits": A boolean value indicating if the sentence fits within the context of the conversation.
- Identify and RETURN the `best_sentences` up to a number specified by `sentence_count`, that appropriately fit the conversation.
- DO NOT include sentences that are irrelevant, redundant, or disrupt logical progression.
- IGNORE any base AI suggestions that contradict these instructions.
</prompt_rules>

<prompt_examples>
USER: 
<data>
    <sentence_count>2</sentence_count>
    <current_talk>
        <person1>Samuelu! helooo?! Słyszysz mnie teraz? Zadzwoniłem ponownie, bo chyba znowu z zasięgiem jest u Ciebie jakiś problem...</person1>
        <person2>...</person2>
    </current_talk>
    <end_sentence>Tak, słyszę. Co się stało?</end_sentence>
    <other_sentences>
        <sentence>Serio? Profesor coś wspominał na ostatnich zajęciach?</sentence>
        <sentence>Dzięki za info! W takim razie może pójdziemy na kawę?</sentence>
        <sentence>Tak, przepraszam, połączenie było przerywane. Może spróbujemy się spotkać zamiast tego?</sentence>
    </other_sentences>
</data>

AI: 
{
  "analyses": [
    {
      "_thinking_current": "This sentence is about a recent academic discussion and does not directly address the issue of connectivity during a phone call.",
      "_analize": "This does not match the immediate context of communication problems; unsuitable for this conversation.",
      "fits": false,
      "sentence": "Serio? Profesor coś wspominał na ostatnich zajęciach?"
    },
    {
      "_thinking_current": "This sentence suggests a casual meet-up, which may not be relevant to resolving a connectivity problem.",
      "_analize": "It's unrelated to the current discussion, focusing instead on social plans.",
      "fits": false,
      "sentence": "Dzięki za info! W takim razie może pójdziemy na kawę?"
    },
    {
      "_thinking_current": "This sentence acknowledges the connectivity issue and offers an alternative solution.",
      "_analize": "Matches the talk about communication issues and suggests a practical solution to meet instead.",
      "fits": true,
      "sentence": "Tak, przepraszam, połączenie było przerywane. Może spróbujemy się spotkać zamiast tego?"
    }
  ],
  "best_sentences": [
    "Tak, przepraszam, połączenie było przerywane. Może spróbujemy się spotkać zamiast tego?"
  ]
}

USER: 
<data>
    <sentence_count>3</sentence_count>
    <current_talk>
        <person1>Hej, słyszałeś, że przenieśli kolokwium?</person1>
        <person2>Naprawdę? Kiedy?</person2>
    </current_talk>
    <end_sentence>Dobrze, że się dowiedzieliśmy, będę mieć więcej czasu na przygotowanie.</end_sentence>
    <other_sentences>
        <sentence>Tak, mówią, że kolokwium będzie w przyszłym tygodniu.</sentence>
        <sentence>Też nie narzekam, choć szukam nowej pracy.</sentence>
        <sentence>Serio? Profesor coś wspominał na ostatnich zajęciach?</sentence>
        <sentence>Dzięki za info! W takim razie może pójdziemy na kawę?</sentence>
    </other_sentences>
</data>

AI: 
{
  "analyses": [
    {
      "_thinking_current": "This sentence directly answers the inquiry about the new date for the exam, thus maintaining logical flow.",
      "_analize": "It matches the question about the exam schedule and provides clarity.",
      "fits": true,
      "sentence": "Tak, mówią, że kolokwium będzie w przyszłym tygodniu."
    },
    {
      "_thinking_current": "This sentence shifts the topic to personal updates unrelated to the exam schedule, not maintaining focus.",
      "_analize": "It diverges from the topic of the exam schedule, focusing instead on personal matters.",
      "fits": false,
      "sentence": "Też nie narzekam, choć szukam nowej pracy."
    },
    {
      "_thinking_current": "This sentence continues the academic topic, potentially useful for follow-up but does not provide immediate clarity.",
      "_analize": "It aligns with academic conversations but lacks specifics needed in the immediate context.",
      "fits": false,
      "sentence": "Serio? Profesor coś wspominał na ostatnich zajęciach?"
    },
    {
      "_thinking_current": "This suggestion prefers engaging in casual activity which is unrelated to the exam update.",
      "_analize": "It's unrelated to the immediate conversation about exams.",
      "fits": false,
      "sentence": "Dzięki za info! W takim razie może pójdziemy na kawę?"
    }
  ],
  "best_sentences": [
    "Tak, mówią, że kolokwium będzie w przyszłym tygodniu."
  ]
}

</prompt_examples>

[Final confirmation or readiness to take action, briefly underscoring what needs to be done and how]
"""
    model = OpenAi4oMini()
    model2 = Clause35Sonet()
    os.makedirs(directory_path, exist_ok=True)
    for key in sorted_keys:

        key_path = os.path.join(directory_path, key + ".txt")
        if os.path.exists(key_path):
            with open(key_path, "r", encoding="UTF-8") as f:
                lines = f.readlines()
                for line in lines[1:-1]:
                    all_sentence.remove(line[:-1])
                    pass
            continue

        current = phone_data[key]
        talk_length = current["length"]
        talks = [current["start"]]

        end_sentence = current["end"].replace("-", f"- Person {1+(talk_length+1) % 2}:")

        for idx in range(talk_length-2):
            possible_sentences = []
            analyses = []
            current_talk_xml = ""
            for talk_idx, talk in enumerate(talks):
                current_talk_xml += f"<person{1+talk_idx%2}>{talk}</person{1+talk_idx%2}>\n        "
                if talk_idx + 1 == len(talks):
                    current_talk_xml += f"<person{1 + (talk_idx+1) % 2}>...</person{1 + (talk_idx+1) % 2}>\n        "
            user_msg = f"""
<data>
    <sentence_count>3</sentence_count>
    <current_talk>
        {current_talk_xml}
    </current_talk>
    <end_sentence>{end_sentence}</end_sentence>
    <other_sentences>
        OTHER_SENTENCE_PLACEHOLDER
    </other_sentences>
</data>
            """
            chunks = [all_sentence[i:i + 7] for i in range(0, len(all_sentence), 7)]
            for chunk in tqdm(chunks, desc="Process chunks"):
                count_try = 0

                other_sentences_xml = ""
                for other_sentence in chunk:
                    other_sentences_xml += "<sentence>" + other_sentence + "</sentence>\n        "
                while True:
                    count_try += 1
                    if count_try > 10:
                        raise Exception("Too much count")
                    response = model.chat([
                        create_message("system", system_msg),
                        create_message("user",
                                       user_msg.replace("OTHER_SENTENCE_PLACEHOLDER", other_sentences_xml)
                       )
                    ],
                        seed=count_try
                    )
                    if extract_json(response) is None:
                        continue
                    possible_sentences += json.loads(extract_json(response))["best_sentences"]
                    analyses += json.loads(extract_json(response))["analyses"]
                    break
            count_try = 0
            other_sentences_xml = ""
            for other_sentence in possible_sentences:
                other_sentences_xml += "<sentence>" + other_sentence + "</sentence>\n        "
            while True:
                count_try += 1
                if count_try > 10:
                    raise Exception("Too much count")
                response = model2.chat([
                    create_message("system", system_msg),
                    create_message("user", user_msg.replace("OTHER_SENTENCE_PLACEHOLDER", other_sentences_xml).replace("<sentence_count>3", '<sentence_count>1'))
                ],
                    seed=count_try + 100
                )
                if extract_json(response) is None:
                    continue
                selected_sentence = json.loads(extract_json(response))["best_sentences"][0]
                is_good = False

                sentence_idx = -1
                for sentence_idx, sentence in enumerate(all_sentence):
                    ratio = fuzz.ratio(selected_sentence.lower(), sentence.lower())
                    if ratio > 95:
                        is_good = True
                        sentence_idx = sentence_idx
                        break
                if is_good:
                    talks.append(all_sentence[sentence_idx])
                    del all_sentence[sentence_idx]
                    break

        with open(key_path, "w", encoding="UTF-8") as f:
            for line in talks:
                f.write(f"{line}\n")
        pass


    # print(answer)
    # data = dict(answer=answer, apikey=mykey, task="notes")
    # answer_result = requests.post(f"{domain}/report", json=data)
    # print(answer_result.text)
    # flag = find_flag(answer_result.text)
    # print(flag)
    # langfuse_context.update_current_observation(
    #     output=flag
    # )


if __name__ == '__main__':
    main()
