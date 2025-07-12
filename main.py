import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
# Replace with the actual URL of your hosted Hugging Face model
# You can find this on your model's page on the Hugging Face Hub.
HUGGING_FACE_API_URL = "https://nobv4nvnk5nb9v9x.us-east-1.aws.endpoints.huggingface.cloud"
# It's best practice to store your API key as an environment variable
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models for Data Validation ---
# This model defines the structure of the incoming request body.
# FastAPI will automatically validate that the request has a 'text' field which is a string.
class TextInput(BaseModel):
    text: str

# This model defines the structure of the JSON response.
class HSLResponse(BaseModel):
    h: int
    s: int
    l: int

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This is crucial because your frontend (GitHub Pages) and backend (Heroku) are on different domains.
# It allows your frontend to make requests to your backend.
origins = [
    "http://localhost",
    "http://localhost:8080",
    # Add the URL of your GitHub Pages site here
    "https://benjaminrongey.github.io/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoint ---
@app.post("/analyze", response_model=HSLResponse)
async def analyze_text_and_get_color(input_data: TextInput):
    """
    Receives text, gets VAD scores from a Hugging Face model,
    and converts them into HSL color values.
    """
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="Hugging Face API key not configured on the server.")

    # 1. Send Text to Hugging Face Model
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": input_data.text}

    try:
        response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        model_output = response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face API: {e}")

    # The expected output from the model is a list containing a dictionary with the VAD scores.
    # e.g., [[{'label': 'VAD', 'score_v': 0.6, 'score_a': 0.3, 'score_d': 0.8}]]
    # You MUST adapt the parsing below to match the ACTUAL JSON structure your model returns.
    try:
        # This is a hypothetical structure. Please adjust to your model's output.
        vad_scores = model_output[0][0]
        valence = vad_scores.get('score_v', 0.5) # Default to neutral
        arousal = vad_scores.get('score_a', 0.5)
        dominance = vad_scores.get('score_d', 0.5)

        if any(v is None for v in [valence, arousal, dominance]):
             raise KeyError("VAD scores not found in model response.")

    except (IndexError, TypeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Could not parse VAD scores from model response: {e}")


    # 2. Convert VAD Floats to HSL Integers
    # VAD scores are typically floats between 0.0 and 1.0.
    # HSL values have specific ranges: Hue (0-360), Saturation (0-100), Lightness (0-100).

    # Valence (Pleasure) -> Hue (Color)
    # 0.0 (unpleasant) -> 0 (red)
    # 0.5 (neutral) -> 120 (green)
    # 1.0 (pleasant) -> 240 (blue)
    # We will map the 0.0-1.0 range to a 0-240 degree range on the color wheel.
    hue = int(valence * 240)

    # Arousal (Excitement) -> Saturation (Intensity)
    # 0.0 (calm) -> low saturation
    # 1.0 (excited) -> high saturation
    # We map this to a practical range of 40-100% to avoid washed-out colors.
    saturation = int(40 + (arousal * 60))

    # Dominance (Control) -> Lightness (Brightness)
    # 0.0 (submissive) -> darker color
    # 1.0 (dominant) -> lighter color
    # We map this to a practical range of 30-70% to avoid pure black or white.
    lightness = int(30 + (dominance * 40))

    return HSLResponse(h=hue, s=saturation, l=lightness)