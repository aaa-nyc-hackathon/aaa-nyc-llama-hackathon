import base64
import json
from dotenv import load_dotenv

load_dotenv()

from llama_api_client import LlamaAPIClient
from pydantic import BaseModel, Field
from typing import List


class FeedbackFormat(BaseModel):
    key_observations: List[str] = Field(
        ...,  # ... indicates that this field is required
        max_items=2,
        description="Key observations from the analysis",
    )
    potential_issues: List[str] = Field(
        ..., max_items=2, description="Potential issues identified during the analysis"
    )
    positives: List[str] = Field(
        ..., max_items=2, description="Positive aspects identified during the analysis"
    )
    final_conclusion: str = Field(
        ..., description="The final conclusion based on the analysis"
    )

    class Config:
        schema_extra = {"title": "Llama Analysis Output"}


def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")


def get_player_feedback(input_img):
    HERD = {
        "mav": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "scout": "Llama-4-Scout-17B-16E-Instruct-FP8",
    }

    # file = "frame6-boxed.jpg"
    file = input_img
    ## text i/o example
    client = LlamaAPIClient()

    contents = [
        {
            "type": "text",
            "text": "For the player in the box in the image, identify any good or bad things they are doing in the game.",
        }
    ]

    base_64_image = image_to_base64(file)
    image_data = {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base_64_image}"},
    }
    contents.append(image_data)

    completion = client.chat.completions.create(
        model=HERD["mav"],
        messages=[{"role": "user", "content": contents}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FeedbackFormat",
                "schema": FeedbackFormat.model_json_schema(),
            },
        },
    )

    feedback = json.loads(completion.completion_message.content.text)
    # with jsonlines.open("feedback_output.jsonl", mode='a') as writer:
    #    writer.write(feedback)
    ## also return the same json
    return feedback
