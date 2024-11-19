import glob
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
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


def file_ocr(full_file_path):
    return pytesseract.image_to_string(Image.open(full_file_path), lang='pol')


def strip_text(text):
    # Find the position of the first '{' and the last '}'
    left = text.find('{')
    right = text.rfind('}')

    # Return the content between the braces if both are found
    if left != -1 and right != -1:
        return text[left:right+1]
    return None

@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    download_url = f"{domain}/dane/pliki_z_fabryki.zip"
    directory_path = os.path.join("tmp", "s02e04")
    download_and_extract_zip(directory_path, download_url)

    # List to store file names
    file_list = []

    texts = []

    ocr_correct_system_message = """Correct OCR-induced errors in the text, ensuring it flows coherently with the previous context. Follow these guidelines:

1. Fix OCR-induced typos and errors:
   - Correct words split across line breaks
   - Fix common OCR errors (e.g., 'rn' misread as 'm')
   - Use context and common sense to correct errors
   - Only fix clear errors, don't alter the content unnecessarily
   - Do not add extra periods or any unnecessary punctuation

2. Maintain original structure:
   - Keep all headings and subheadings intact

3. Preserve original content:
   - Keep all important information from the original text
   - Do not add any new information not present in the original text
   - Remove unnecessary line breaks within sentences or paragraphs
   - Maintain paragraph breaks
   
4. Maintain coherence:
   - Ensure the content connects smoothly with the previous context
   - Handle text that starts or ends mid-sentence appropriately

IMPORTANT: Respond ONLY with the corrected text. Preserve all original formatting, including line breaks. Do not include any introduction, explanation, or metadata.
"""

    user_find_system_prompt = """
Please carefully analyze the text provided and extract information about the captured persons or any traces 
indicating their presence. 

Signs or traces of presence: Document any objects, materials, markings or environmental signs indicating 
their presence. This can include personal belongings, footprints, handwriting, physical evidence (such as fingerprints) 
or other observable indicators.

Do not detect the presence of other animals, like mouse and cats.

In your answer, return 1 if you found such information, or 0 if there is no such information. Output should be in JSON.
Provide your explanation in the JSON response in the poly "_thinking"

Examples:

USER: "Grupa potwierdziła, że wzięła trzech zakładników i czeka na odpowiedź na swoje żądania."  
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: "Nasz przyjaciel podróżuje po Europie i powinien dotrzeć do celu jutro."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Pojawiły się doniesienia, że kilka kotów zostało zatrzymanych na granicy z powodu problemów wizowych."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Widziałem odciski palców człowieka przy głównej bramie."  
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: "Kilku żołnierzy zostało pojmanych podczas bitwy i obecnie przebywa w nieznanym miejscu."  
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: "Proszę, przypomnij wszystkim, aby zamknęli drzwi przed wyjściem z biura dziś wieczorem."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Policja znalazła ślady podejżanego"  
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: "Piec bierze urlop i planuje spędzić kilka miesięcy w odległej wiosce."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Porywacze przetrzymują rodzinę jako zakładników i żądają dużego okupu za ich uwolnienie."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Organizujemy grupę poszukiwawczą, aby odnaleźć zaginionych w górach turystów."  
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: "Przeszukaliśmy domy w pobliskiej wsi w poszukiwaniu rebeliantów."
AI: {"_thinking": "justify your answer here", "answer": 0}
    
"""

    hardware_find_system_msg = """Analyze the text to determine if it contains information about repairing 
or fixing hardware defects only. Look for terms related to diagnosing hardware issues, replacing or repairing 
physical components (e.g., hard drives, motherboards, screens), or restoring functionality to broken hardware. 
Ignore any references to software issues, applications, or operating systems. 

To Detect:
- Hardware Terms: hard drive, motherboard, screen, battery, power supply, processor, memory, fan, circuit, physical component
- Hardware Repair Actions: repair, replace, swap, diagnose, troubleshoot, resolved hardware issue
- Defective Hardware: broken, faulty, malfunctioning, damaged, physical defect, component failure
To Ignore:
- Software-Related Terms: software, application, program, operating system, update, bug, patch, install software, code, crash, driver issues

If the text includes details specific  to hardware repairs, answer it as '1' else return 0. Output should be in JSON.
Provide your explanation in the JSON response in the poly "_thinking"

Examples:

USER: Example with Hardware Repair Information
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: Example with Software Defect Information (Should be Ignored)
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: Example with Mixed Hardware and Software Information (Ignore Software Part)
AI: {"_thinking": "justify your answer here", "answer": 1}

USER: Example with General Product Description (No Repair Information)
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: Example with Hardware Malfunction without Repair
AI: {"_thinking": "justify your answer here", "answer": 0}

USER: Example with Detailed Hardware Repair Process
AI: {"_thinking": "justify your answer here", "answer": 1}
    
"""

    # Iterate through files in the directory
    for file_name in os.listdir(directory_path):
        # Check if the file has the correct extension
        full_path = os.path.join(directory_path, file_name)
        if full_path.endswith(".txt"):
            with open(full_path, "r", encoding="UTF-8") as f:
                texts.append([file_name, f.read()])
        if full_path.endswith(".mp3"):
            transcription_path = full_path + ".transcription"
            if not os.path.exists(transcription_path):
                whisper_client = OpenAiWhisper()
                transcript = whisper_client.transcript(full_path, language="en")
                with open(transcription_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
            with open(transcription_path, "r", encoding="UTF-8") as f:
                texts.append([file_name, f.read()])
        if full_path.endswith(".png"):
            transcription_path = full_path + ".transcription"
            if not os.path.exists(transcription_path):
                ocr_text = file_ocr(full_path)
                model = OpenAi4oMini(debug=True)
                system_msg = create_message("system", ocr_correct_system_message)
                user_msg = create_message_with_image("user", ocr_text, [full_path], model.formats)
                response = model.chat([system_msg, user_msg])
                with open(transcription_path, "w", encoding="utf-8") as f:
                    f.write(response)
            with open(transcription_path, "r", encoding="UTF-8") as f:
                content = ""
                for line in f.readlines():
                    if "REPAIR NOTE" not in line and "FROM" not in line and "APPROVED BY" not in line:
                        content += line.replace("\n", " ").replace("  ", " ").strip()
                        content += " "
                texts.append([file_name, content])
            pass
    model = Llama31P8B(debug=False)
    find_user_system = create_message("system", user_find_system_prompt)
    hardware_system = create_message("system", hardware_find_system_msg)
    find_user_result = []
    find_hardware_result = []
    for file_name, text in tqdm(texts):
        for system, result in [(find_user_system, find_user_result), (hardware_system, find_hardware_result)]:
            while True:
                try:
                    response = model.chat([system, create_message("user", text)])
                    response_json_str = strip_text(response)
                    response_json = json.loads(response_json_str)
                    if response_json["answer"] == 1:
                        result.append(file_name)
                    break
                except:
                    pass
    pass
    answer = {
      "people": find_user_result,
      "hardware": find_hardware_result,
    }
    print(answer)
    data = dict(answer=answer, apikey=mykey, task="kategorie")
    answer_result = requests.post(f"{domain}/report", json=data)
    print(answer_result.text)
    flag = find_flag(answer_result.text)
    print(flag)
    langfuse_context.update_current_observation(
        output=flag
    )


if __name__ == '__main__':
    main()
