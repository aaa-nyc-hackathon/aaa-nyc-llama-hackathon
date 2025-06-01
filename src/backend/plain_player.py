"""
code to test generating player jersey numbers from the images from the video.
"""

import json
import os
from llama_api_client import LlamaAPIClient

import cv2
import base64
import jsonlines
import subprocess
from dotenv import load_dotenv
import shutil
load_dotenv()


def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')

def get_player_positions(inputvideo, plot=False):
    try:
      client = LlamaAPIClient(
          api_key=os.environ.get("LLAMA_API_KEY"), # This is the default and can be omitted
      )
    except:
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
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": contents,
                },
            ],
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "schema": {
                    "title": "Llama4 Game Data",
                    "type": "object",
                        "properties": {
              "players": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "jerseyNumber": {
                      "type": "integer",
                      "description": "An integer representing the Jersey number on the player's jersey"
                    },
                    "coordinates": {
                      "type": "object",
                      "description": "The current X/Y coordinates of the player.",
                      "properties": {
                        "x_coordinate": {
                          "type": "number",
                          "description": "The percentage inside the image in the X coordinate to the player"
                        },
                        "y_coordinate": {
                          "type": "number",
                          "description": "The percentage inside the image in the Y coordinate to the player"
                        }
                      },
                      "required": ["x_coordinate", "y_coordinate"]
                    },
                    "team": {
                      "type": "string",
                      "description": "One of the teams names as a string",
                      "enum": ["Clemson", "Alabama"]
                    }
                  },
                  "required": ["jerseyNumber", "coordinates", "team"]
                }
              },
              "scores": {
                "type": "object",
                "properties": {
                  "team1": {
                    "type": "string",
                    "description": "Name of the first team",
                    "enum": ["Clemson"], 
                  },
                  "team1_score": {
                    "type": "integer",
                    "description": "Score of the first team"
                  },
                  "team2": {
                    "type": "string",
                    "description": "Name of the second team",
                    "enum": ["Alabama"],
                  },
                  "team2_score": {
                    "type": "integer",
                    "description": "Score of the second team"
                  }
                },
                "required": ["team1", "team1_score", "team2", "team2_score"]
              }
            },
         "required": ["players", "scores"]
        }
    }
})
        image_llama_data = json.loads(response.completion_message.content.text)
        
        # Create a result dictionary that includes timestamps and the player data
        result = {
            'start': frame_end-1,
            'end': frame_end,
            'image_data': image_llama_data
        }

        results_output.append(result)

    return results_output
