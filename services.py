import os
import replicate
import base64
from openai import OpenAI
from fastapi import HTTPException, UploadFile
import requests
from dotenv import load_dotenv

load_dotenv() # Load env variables

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

try:
    openai_client = OpenAI()
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set.")
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN environment variable not set.")

def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


async def get_musical_description_from_openai(image: UploadFile):
    """Sends image to OpenAI and returns a musical description."""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")
    if not image.content_type or not image.content_type.startswith("image/"):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        image_bytes = await image.read()
        base64_image = encode_image_to_base64(image_bytes)
        response = openai_client.chat.completions.create(
            model="o4-mini",
            messages=[
                {
                "role": "developer",
                "content": [
                    {
                    "type": "text",
                    "text": "I want you to be an intermediary to a music generative model. So, you will take an image and create a description based on the image to create a musical query to send to a gen music model. I want you to account for the vibe, genre, colors, and feeling of the picture. Describe the image from a musical perspective. What kind of music or sound does it evoke? Think about mood, rhythm, instrumentation, genre, etc. Max 2 sentences. Return only the description, no other text."
                    },
                ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image.content_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            response_format={
                "type": "text"
            },
            reasoning_effort="low",
            store=False
            )
        description = response.choices[0].message.content
        if not description:
             raise HTTPException(status_code=500, detail="OpenAI returned an empty description.")
        return description

    except Exception as e:
        print(f"Error processing image with OpenAI: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get musical description from OpenAI: {e}")


async def generate_audio_from_replicate(prompt: str) -> str:
    """Generates audio from a prompt using Replicate and returns the audio URL."""
    if not REPLICATE_API_TOKEN:
         raise HTTPException(status_code=500, detail="Replicate API token not configured.")

    model_identifier = "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"

    try:
        print(f"Running Replicate model {model_identifier} with input: {prompt}")
        output = replicate.run(
            model_identifier,
            input={
                "top_k": 150,
                "top_p": 0,
                "prompt": prompt,
                "duration": 8,
                "temperature": 1,
                "continuation": False,
                "model_version": "stereo-large",
                "output_format": "wav",
                "continuation_start": 0,
                "multi_band_diffusion": False,
                "normalization_strategy": "loudness",
                "classifier_free_guidance": 3
            }
        )
        print(f"Replicate output: {output, type(output)}")


        # Make sure output is a str url
        audio_url = None
        if hasattr(output, 'url') and isinstance(getattr(output, 'url', None), str):
            audio_url = output.url
        elif isinstance(output, str) and output.startswith("http"):
            audio_url = output
        if not audio_url:
            print(f"Unexpected or missing audio URL in Replicate output: {output}")
            # Be more specific about the error
            raise HTTPException(status_code=500, detail="Audio generation succeeded but failed to get a valid audio URL from the output.")

        return audio_url # Return only the extracted URL string

    except replicate.exceptions.ReplicateError as e:
         print(f"Replicate API error: {e}")
         raise HTTPException(status_code=502, detail=f"Audio generation service error: {e}")
    except Exception as e:
        print(f"Error during Replicate audio generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio via Replicate: {e}")

