# backend.py
import os
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import service functions
from services import (
    get_musical_description_from_openai,
    generate_audio_from_replicate,
)

# --- Configuration --- (Keep API key loading for potential direct use or config checks)
# Make sure to set these in your environment before running the app
# export OPENAI_API_KEY='your_openai_key'
# export REPLICATE_API_TOKEN='your_replicate_token'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    # Provide a default or raise an error if required for startup
if not REPLICATE_API_TOKEN:
    print("Warning: REPLICATE_API_TOKEN environment variable not set.")
    # Provide a default or raise an error if required for startup

# Initialize FastAPI App
app = FastAPI()

# --- CORS Middleware ---
# Allow requests from your frontend development server
# In production, restrict origins more specifically
origins = [
    "http://localhost:3000", # Default Next.js dev port
    "http://127.0.0.1:3000",
    # Add any other origins if needed (e.g., your deployed frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)


# --- Helper Functions (Keep only those specific to backend.py) ---
# Remove encode_image_to_base64 (moved to services.py)
# Keep stream_audio_from_url IF it wasn't moved to services.py, otherwise remove.
# Based on the previous step, stream_audio_from_url was moved, so we remove it here.


# --- API Endpoints ---

@app.post("/describe-image-musically")
async def describe_image_musically_endpoint(image: UploadFile = File(...)):
    """
    Endpoint to get a musical description for an uploaded image.
    Delegates the core logic to the services module.
    """
    try:
        # Call the service function
        description = await get_musical_description_from_openai(image)
        # Return description with key "description"
        return {"description": description}
    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service
        raise e
    except Exception as e:
        # Catch unexpected errors during endpoint handling
        print(f"Unexpected error in /describe-image-musically endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/generate-audio")
async def generate_audio_endpoint(prompt: str = Form(...)):
    """
    Endpoint to generate audio from a text prompt.
    Delegates generation to the services module and returns the audio URL.
    """
    try:
        # Call the service function to get the audio URL
        audio_url = await generate_audio_from_replicate(prompt)
        # Return the URL with key "audio_url"
        return {"audio_url": audio_url}

    except HTTPException as e:
        # Re-raise HTTPExceptions raised by the service
        raise e
    except Exception as e:
        # Catch unexpected errors during endpoint handling
        print(f"Unexpected error in /generate-audio endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating audio: {e}")


# --- Run the App (for local development) ---
# Use uvicorn to run: uvicorn backend:app --reload
if __name__ == "__main__":
    import uvicorn
    print("Starting Uvicorn server...")
    print("Remember to set OPENAI_API_KEY and REPLICATE_API_TOKEN environment variables.")
    print("Run with: uvicorn backend.backend:app --reload --host 0.0.0.0 --port 8000")
    # Note: Running uvicorn programmatically like this is mainly for convenience.
    # In production, you'd typically use the uvicorn command directly.
    # uvicorn.run(app, host="0.0.0.0", port=8000) # This line won't work well with --reload