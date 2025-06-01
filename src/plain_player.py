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

load_dotenv()


# check for access
if not os.environ.get("LLAMA_API_KEY"):
    raise ValueError(f"you need to obtain a LLAMA api key first")

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

HERD = {"mav": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "scout": "Llama-4-Scout-17B-16E-Instruct-FP8"
        }

argparser = ArgumentParser(
           prog='Video processing',
           description='This program processes a video from local fs and pipes stuff to a .jsonl',
           epilog='')

argparser.add_argument("--inputvideo", type=str, default="alabama_clemson_30s_clip.mp4", help='filename locally')
argparser.add_argument("--plot", action='store_true', help='make a plot')

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')

if __name__ == "__main__":
    args = argparser.parse_args()
    try:
      client = LlamaAPIClient(
          api_key=os.environ.get("LLAMA_API_KEY"), # This is the default and can be omitted
      )
    except:
      client = LlamaAPIClient()

    #check if images directory exists, if not create it
    if not os.path.exists("images"):
        os.makedirs("images")
    ffmpeg_cmd_retry = ['ffmpeg', '-i', args.inputvideo, '-vf', 'fps=fps=1', str(os.path.join(os.getcwd(), "images", "frame%d.jpg"))]


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
                        "description": "One of the teams names as a string"
                      }
                    },
                    "required": ["jerseyNumber", "coordinates", "team"],
                  }
                }
            }
        })
        print(response.completion_message.content.text)
        image_llama_data = json.loads(response.completion_message.content.text)
        with jsonlines.open("output.jsonl", mode='a') as writer:
            writer.write({'start': frame_end-1, 'end': frame_end, 'image_data': image_llama_data})

#image = cv2.imread(image_filepath)
if args.plot:
    height, width, channels = image.shape
    # Write this data to a .jsonl file...
    for llama_data in image_llama_data:
        x_perc, y_perc = llama_data['coordinates']['x_coordinate'], llama_data['coordinates']['y_coordinate']
        if llama_data['team'].lower() == "clemson":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        # Calculate the pixel coordinates
        x_coordinate = int(width * x_perc)
        y_coordinate = int(height * y_perc)
        center_coordinates = (x_coordinate, y_coordinate)
        radius = 10  # Example radius
        thickness = -1  # Filled circle (negative value)
        # Draw the circular point
        image = cv2.circle(image, center_coordinates, radius, color, thickness)
        cv2.imwrite("image_with_points.jpg", image)