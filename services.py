import os
import replicate
import base64
from openai import OpenAI
from fastapi import HTTPException, UploadFile
import requests
from dotenv import load_dotenv

load_dotenv()
# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- Initialize Clients ---
try:
    openai_client = OpenAI()
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        # You might want more robust handling here depending on requirements
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    openai_client = None

if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN environment variable not set.")
    # You might want more robust handling here

# --- Helper Functions ---
def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')

# --- Service Functions ---

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
                    "text": "I want you to be an intermediary to a music generative model. So, you will take an image and create a description based on the image to create a musical query to send to a gen music model. I want you to account for the vibe, genre, colors, and feeling of the picture. Describe the image from a musical perspective. What kind of music or sound does it evoke? Think about mood, rhythm, instrumentation, genre, etc. Max 2 sentences. Return the description in a json with the key \"text\". "
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
                "type": "json_object"
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
        # Consider more specific exception handling
        raise HTTPException(status_code=500, detail=f"Failed to get musical description from OpenAI: {e}")


async def generate_audio_from_replicate(prompt: str) -> str:
    """Generates audio from a prompt using Replicate and returns the audio URL."""
    if not REPLICATE_API_TOKEN:
         raise HTTPException(status_code=500, detail="Replicate API token not configured.")

    # Replace with the specific model you want to use
    model_identifier = "riffusion/riffusion:8cf61ea6c56afd61d8f5b9ffd14d7c216c0a93844ce2d82ac1c9ecc9c7f24e05"
    # This specific model might need more structured input, adjust accordingly
    input_data = {"prompt_a": prompt} # Adjust based on the chosen model's API

    try:
        print(f"Running Replicate model {model_identifier} with input: {input_data}")
        output = replicate.run(
            model_identifier,
            input=input_data
        )
        print(f"Replicate output: {output}")

        # --- Handling Replicate Output ---
        audio_url = None
        if isinstance(output, dict) and "audio" in output and output["audio"]:
            audio_url = output["audio"]
        elif isinstance(output, str) and output.startswith("http"):
             audio_url = output

        if not audio_url:
            print(f"Unexpected output format from Replicate: {output}")
            raise HTTPException(status_code=500, detail="Unexpected output format from audio generation model.")

        return audio_url

    except replicate.exceptions.ReplicateError as e:
         print(f"Replicate API error: {e}")
         raise HTTPException(status_code=502, detail=f"Audio generation service error: {e}")
    except Exception as e:
        print(f"Error during Replicate audio generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio via Replicate: {e}")

async def stream_audio_from_url(audio_url: str):
    """Downloads and streams audio content from a URL."""
    try:
        # Use a context manager for the request
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
            # Stream content in chunks
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # Filter out keep-alive new chunks
                    yield chunk
    except requests.exceptions.RequestException as e:
        print(f"Error streaming audio from {audio_url}: {e}")
        # Propagate the error or handle it gracefully
        # Option 1: Raise an exception that FastAPI can handle
        # raise HTTPException(status_code=502, detail=f"Failed to stream audio from source: {e}")
        # Option 2: Yield an empty byte string or a specific error message if the client expects it
        yield b"" # Yield empty bytes on error to keep the streaming response structure
    except Exception as e:
        # Catch any other unexpected errors during streaming
        print(f"Unexpected error streaming audio: {e}")
        yield b""
