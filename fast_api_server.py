
"""
Human Behaviour Analysis API

This FastAPI application provides a REST API for analyzing
human behaviour from video frames using Qwen2.5-VL-3B-Instruct.

Workflow
--------
1. Client extracts frames from a video.
2. Frames are encoded as Base64 strings.
3. Frames are sent to the /analyze endpoint.
4. The server decodes the images.
5. Qwen2.5-VL analyzes the sequence of frames.
6. A natural language description is returned.

Model
-----
Qwen/Qwen2.5-VL-3B-Instruct

Endpoints
---------
GET  /
    Health check endpoint.

POST /analyze
    Analyze a sequence of video frames and generate
    a behaviour description.
"""

import io
import base64
from typing import List

import torch
from PIL import Image

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

# ============================================================
# Model Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

print(f"Loading model: {MODEL_NAME}")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully.")

# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Human Behaviour Analysis API",
    description=(
        "Analyze human behaviour from video frames "
        "using Qwen2.5-VL."
    ),
    version="1.0"
)

# ============================================================
# Request Schema
# ============================================================

class FrameRequest(BaseModel):
    """
    Request body for behaviour analysis.

    Attributes
    ----------
    frames : List[str]
        List of Base64 encoded image frames extracted
        from a video.
    """

    frames: List[str]

# ============================================================
# Utility Functions
# ============================================================

def decode_frames(frame_list: List[str]) -> List[Image.Image]:
    """
    Decode Base64 image strings into PIL images.

    Parameters
    ----------
    frame_list : List[str]
        List of Base64 encoded images.

    Returns
    -------
    List[PIL.Image.Image]
        Decoded image objects.
    """

    images = []

    for frame in frame_list:

        image_bytes = base64.b64decode(frame)

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        images.append(image)

    return images


def analyze_frames(images: List[Image.Image]) -> str:
    """
    Analyze a sequence of images using Qwen2.5-VL.

    Parameters
    ----------
    images : List[PIL.Image.Image]
        Sequential frames extracted from a video.

    Returns
    -------
    str
        Natural language description of the observed
        actions and behaviour.
    """

    prompt = """
You are an expert human behavior analyst.

The provided images are sequential frames extracted from a video.

Analyze:

1. Actions being performed.
2. Human behaviour patterns.
3. Object interactions.
4. Overall activity.

Respond with only a concise description.
"""

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {
                        "type": "image",
                        "image": image
                    }
                    for image in images
                ],
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # Generate chat template
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Prepare model inputs
    inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True
    )

    inputs = inputs.to(model.device)

    # Generate response
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    # Decode generated tokens
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return output

# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
def root():
    """
    Health check endpoint.

    Returns
    -------
    dict
        Service status and model information.
    """

    return {
        "status": "running",
        "model": MODEL_NAME
    }


@app.post("/analyze")
async def analyze(request: FrameRequest):
    """
    Analyze human behaviour from a sequence of frames.

    Parameters
    ----------
    request : FrameRequest
        Incoming request containing Base64 encoded
        video frames.

    Returns
    -------
    dict
        Behaviour description generated by the model.
    """

    try:

        if len(request.frames) == 0:

            return {
                "error": "No frames provided."
            }

        images = decode_frames(
            request.frames
        )

        description = analyze_frames(
            images
        )

        return {
            "description": description
        }

    except Exception as e:

        import traceback

        traceback.print_exc()

        return {
            "error": str(e)
        }

# ============================================================
# Application Entry Point
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

