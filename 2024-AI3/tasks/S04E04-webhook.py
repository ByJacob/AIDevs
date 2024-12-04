# cloudflared tunnel --url http://localhost:3000

import signal
import threading
import time

import requests
from PIL import Image
import os

from flask import Flask, request, jsonify
from tqdm import tqdm

from models import *
from utils import create_message, create_message_with_image, find_flag

domain = os.getenv('CENTRALA_DOMAIN')
mykey = os.getenv('AIDEV3_API_KEY')

directory_path = os.path.join("tmp", "s04e04")

base_image_path = os.path.join(directory_path, "mapa_s04e04.png")

def split_image():
    # Open the image
    img = Image.open(base_image_path)
    width, height = img.size

    # Calculate the dimensions of each grid cell
    cell_width = width // 4
    cell_height = height // 4

    # Create output folder if it doesn't exist
    os.makedirs(directory_path, exist_ok=True)

    # Loop to split the image
    for i in range(4):
        for j in range(4):
            left = j * cell_width
            top = i * cell_height
            right = left + cell_width
            bottom = top + cell_height

            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))

            # Save the cropped image
            cropped_img.save(os.path.join(directory_path, f'cell_{i}_{j}.png'))
    print(f"Image successfully split into 4x4 grid and saved in '{directory_path}'.")
    model = OpenAi4oMini()
    system = create_message("system", "Your task is to analyze an image and respond with a single word that best captures its overall essence or primary subject. Response in Polish language")
    for file in tqdm(os.listdir(directory_path), desc="Process image"):
        file_abs = os.path.join(directory_path, file)
        if "cell" not in file or "txt" in file:
            continue
        user = create_message_with_image("user", "", [file_abs], model.formats)
        response = model.chat([system, user])
        with open(file_abs + ".txt", "w", encoding="utf-8") as f:
            f.write(response)
    pass

def part1():
    split_image()
    pass

def part2():

    prompt_msg = """
Given a 4x4 grid map, starting from the position (0,0), you will receive a series of instructions in Polish that 
describe movements on the map. Interpret these instructions to determine the final position on the map. The directions are:

- Up: Move one square up.
- Down: Move one square down.
- Left: Move one square left.
- Right: Move one square right.

If the instruction contains phrases like "maksymalnie" (maximum), move as far as possible in the specified direction 
until you reach the edge of the map. If instructions indicate hesitation or cancellation 
(e.g., "nie idziemy" or "Zaczynamy od nowa"), reset to (0,0) and continue interpreting from that point.

Examples:

User: Idziemy na sam dół mapy. Albo nie! nie! nie idziemy. Zaczynamy od nowa. W prawo maksymalnie idziemy. Co my tam mamy?
AI: {
  "_thinking": "Started at (0,0). Ignored 'Idziemy na sam dół' due to 'nie idziemy' and reset. Then moved right maximally to (0,3).",
  "points": [
    [0, 0],
    [0, 3]
  ],
  "last_point": [0, 3]
}

User: W lewo maksymalnie. Potem do góry. Ale nie! Wracamy na start. Na dół maksymalnie idziemy.
AI: {
  "_thinking": "Started at (0,0). Attempted to move left and up but already at the edge. Reset and then moved down maximally to (3,0).",
  "points": [
    [0, 0],
    [3, 0]
  ],
  "last_point": [3, 0]
}


"""

    app = Flask(__name__)

    model = OpenAi4oMini()
    system = create_message("system", prompt_msg)

    # Function to execute when the app starts
    def startup_task():
        print("Startup task is running...")
        time.sleep(10)  # Simulate some processing (replace with your logic)
        answer = "https://recreation-posters-valve-snowboard.trycloudflare.com"
        print(answer)
        data = dict(answer=answer, apikey=mykey, task="webhook")
        answer_result = requests.post(f"{domain}/report", json=data)
        print(answer_result.text)
        flag = find_flag(answer_result.text)
        print(flag)
        langfuse_context.update_current_observation(
            output=flag
        )
        print("Startup task completed. Shutting down the server...")
        shutdown_server()

    # Function to shut down the Flask app gracefully
    def shutdown_server():
        os.kill(os.getpid(), signal.SIGINT)  # Sends a signal to stop the server

    @app.route('/', methods=['POST'])
    def handle_instruction():

        data = request.get_json()
        print(data)

        instruction = data['instruction']
        print(f"Received instruction: {instruction}")

        response = model.chat([system, create_message("user", instruction)])

        print(response)
        point = json.loads(response)['last_point']

        with open(os.path.join(directory_path, f"cell_{point[0]}_{point[1]}.png.txt")) as f:
            desc = f.read()

        return jsonify({
            "description": desc
        })

    if __name__ == '__main__':
        # Run the startup task in a separate thread
        thread = threading.Thread(target=startup_task)
        thread.start()

        # Start the Flask app
        app.run(debug=True, host='0.0.0.0', port=3000)

@observe(capture_input=False, capture_output=False)
def main():
    script_name = os.path.basename(__file__)
    langfuse_context.update_current_trace(
        name=script_name
    )
    part2()



    # print(answer)
    # data = dict(answer=answer, apikey=mykey, task="softo")
    # answer_result = requests.post(f"{domain}/report", json=data)
    # print(answer_result.text)
    # flag = find_flag(answer_result.text)
    # print(flag)
    # langfuse_context.update_current_observation(
    #     output=flag
    # )


if __name__ == '__main__':
    main()
