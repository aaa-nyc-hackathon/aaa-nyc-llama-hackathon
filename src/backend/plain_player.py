"""
code to test generating player jersey numbers from the images from the video.
"""

import json
import os
from llama_api_client import LlamaAPIClient

import base64
import subprocess
from dotenv import load_dotenv
import shutil
load_dotenv()

from pydantic import BaseModel, validator
from typing import List, Optional, Union

class Coordinates(BaseModel):
    x_coordinate: float
    y_coordinate: float

class Player(BaseModel):
    coordinates: Coordinates
    team: str
    jerseyNumber: int

class Team(BaseModel):
    teamName: str
    players: List[Player]
    score: int

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')

def get_player_positions(inputvideo, plot=False):
    client = LlamaAPIClient()
    #check if images directory exists, if not create it
    if os.path.exists("images"):
        shutil.rmtree("images")
        os.makedirs("images")

    ffmpeg_cmd_retry = ['ffmpeg', '-i', inputvideo, '-vf', 'fps=fps=1', str(os.path.join(os.getcwd(), "images", "frame%d.jpg"))]
    subprocess.run(ffmpeg_cmd_retry, check=True)
    # Execute a command and wait for it to finish
    return_code = subprocess.call(ffmpeg_cmd_retry)
    if return_code == 0:
        print("Command executed successfully.")
    else:
        print("Command failed with exit code:", return_code)

    contents = [{"type": "text",
                 "text": "For each player in each image, identify the team they are on, their jersey number, and their coordinates. Their should be at least 3 or 4 players from each team.",
                }]

    files = [entry.name for entry in os.scandir("images") if entry.is_file()]
    # TODO modify this to order by %d
    files.sort()

    # sample each third file we have a list of
    SAMPLE_INTERVAL = 3
    results_output = []

    for frame_id, file in enumerate(files):
        # frames index starting at 1
        frame_end = int(file.strip("frame").strip(".jpg"))
        base_64_image = image_to_base64("images/" + file)
        image_data = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base_64_image}"
            },
        }
        if len(contents) < 2:
            contents.append(image_data)
        else:
            contents[1] = image_data

        # print(f"frame_id: {frame_id!r} and contents: {contents!r}\n\n")
        response = client.chat.completions.create(
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": contents,
                },
            ],
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Team",
                "schema": Team.model_json_schema(),
                }
            }
        )
        try:
            image_llama_data = json.loads(response.completion_message.content.text)
        except Exception as e:
            print(frame_id, e)
            continue
        # Create a result dictionary that includes timestamps and the player data
        result = {
            'start': frame_end-1,
            'end': frame_end,
            'image_data': image_llama_data
        }

        results_output.append(result)

    print(results_output[0]['image_data'].keys())
    return results_output
