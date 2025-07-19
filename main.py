import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
HUGGING_FACE_API_URL = "https://nobv4nvnk5nb9v9x.us-east-1.aws.endpoints.huggingface.cloud"
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str

class HSLResponse(BaseModel):
    h: int
    s: int
    l: int

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://benjaminrongey.github.io",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    payload = {"inputs": input_data.text, "parameters": {"top_k": 3}} # Request all 3 scores

    try:
        response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        model_output = response.json()
        print(f"Hugging Face API Response: {model_output}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face API: {e}")

    # --- START OF FINAL PARSING LOGIC ---
    try:
        # The corrected model output is a list of three dictionaries:
        # e.g., [{'label': 'V', 'score': 3.49}, {'label': 'A', 'score': 3.62}, {'label': 'D', 'score': 3.02}]
        
        # Create a dictionary for easy lookup: {'V': 3.49, 'A': 3.62, 'D': 3.02}
        scores = {item['label']: item['score'] for item in model_output}

        # Get the VAD scores from the new dictionary
        # The model was trained on scores from 1 to 5.
        valence_score = scores.get('V', 3.0) # Default to neutral 3.0
        arousal_score = scores.get('A', 3.0)
        dominance_score = scores.get('D', 3.0)

        # Normalize scores from the model's [1, 5] range to a [0.0, 1.0] range
        # that our HSL conversion logic can use.
        MAX_SCORE = 5.0
        MIN_SCORE = 1.0
        valence = (valence_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        arousal = (arousal_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        dominance = (dominance_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        
        # Clamp values to ensure they are safely within the 0.0-1.0 range
        valence = min(max(valence, 0.0), 1.0)
        arousal = min(max(arousal, 0.0), 1.0)
        dominance = min(max(dominance, 0.0), 1.0)

    except (IndexError, TypeError, KeyError) as e:
        # This error will trigger if the model output is not the expected list of dicts.
        raise HTTPException(status_code=500, detail=f"Could not parse VAD scores from model response. Error: {e}, Response: {model_output}")
    # --- END OF FINAL PARSING LOGIC ---


    # 2. Convert VAD Floats (0.0-1.0) to HSL Integers
    # Valence (Pleasure) -> Hue (Color)
    hue = int(valence * 240)

    # Arousal (Excitement) -> Saturation (Intensity)
    saturation = int(40 + (arousal * 60))

    # Dominance (Control) -> Lightness (Brightness)
    lightness = int(30 + (dominance * 40))

    return HSLResponse(h=hue, s=saturation, l=lightness)