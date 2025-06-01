import base64
import os
import json
import jsonlines
from dotenv import load_dotenv
load_dotenv()

from llama_api_client import LlamaAPIClient

def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode('utf-8')



def get_player_feedback(input_img):
  HERD = {"mav": "Llama-4-Maverick-17B-128E-Instruct-FP8",
          "scout": "Llama-4-Scout-17B-16E-Instruct-FP8"}

  # file = "frame6-boxed.jpg"
  file = input_img
  ## text i/o example
  try:
    client = LlamaAPIClient(
        api_key=os.environ.get("LLAMA_API_KEY"), # This is the default and can be omitted
    )
  except:
    client = LlamaAPIClient()

  contents = [{"type": "text",
               "text": "For the player in the box in the image, identify any good or bad things they are doing in the game.",
              }]

  base_64_image = image_to_base64(file)
  image_data = {
      "type": "image_url",
      "image_url": {
          "url": f"data:image/jpeg;base64,{base_64_image}"
      },
  }
  contents.append(image_data)

  completion = client.chat.completions.create(
      model=HERD["mav"],
      messages=[
          {
              "role": "user",
              "content": contents
          }
      ],
      response_format = {
       "type": "json_schema",
       "json_schema": {
       "schema": {
        "title": "Llama Analysis Output",
        "type": "object",
        "properties": {
          "key_observations": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "A list of key observations from the analysis"
            },
            "description": "Key observations from the analysis",
            "maxItems": 2
          },
          "potential_issues": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "A list of potential issues identified during the analysis"
            },
            "description": "Potential issues identified during the analysis",
            "maxItems": 2
          },
          "positives": {
            "type": "array",
            "items": {
              "type": "string",
              "description": "A list of positive aspects identified during the analysis"
            },
            "description": "Positive aspects identified during the analysis",
            "maxItems": 2
          },
          "final_conclusion": {
            "type": "string",
            "description": "The final conclusion based on the analysis"
          }
        },
        "required": ["key_observations", "potential_issues", "positives", "final_conclusion"]
      }
    }
  }
  )

  feedback = json.loads(completion.completion_message.content.text)
  #with jsonlines.open("feedback_output.jsonl", mode='a') as writer:
  #    writer.write(feedback)
  ## also return the same json
  return feedback