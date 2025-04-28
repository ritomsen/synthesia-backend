# backend.py
import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from services import (
    get_musical_description_from_openai,
    generate_audio_from_replicate,
)

# Get API keys 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN environment variable not set.")

# Initialize FastAPI App
app = FastAPI()

# --- CORS Middleware ---
origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)


# Endpoints 

@app.post("/describe-image-musically")
async def describe_image_musically_endpoint(image: UploadFile = File(...)):
    """
    Endpoint to get a musical description for an uploaded image.
    """
    try:
        description = await get_musical_description_from_openai(image)
        return {"description": description}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /describe-image-musically endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/generate-audio")
async def generate_audio_endpoint(prompt: str = Form(...)):
    """
    Endpoint to generate audio from a text prompt.
    """
    try:
        audio_url = await generate_audio_from_replicate(prompt)
        return {"audio_url": audio_url}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Unexpected error in /generate-audio endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating audio: {e}")


