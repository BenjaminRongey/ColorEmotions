import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import time
import pickle
import numpy as np

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
HUGGING_FACE_API_URL = "https://nobv4nvnk5nb9v9x.us-east-1.aws.endpoints.huggingface.cloud"
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- Load Pre-computed CDF values ---
try:
    with open("cdf_transformers.pkl", "rb") as f:
        cdfs = pickle.load(f)
    # Assuming cdfs is a dictionary like {'V': (x, y), 'A': (x, y), 'D': (x, y)}
    cdf_v = cdfs['V']
    cdf_a = cdfs['A']
    cdf_d = cdfs['D']
except FileNotFoundError:
    # Handle case where file is not found, maybe during local testing
    # You can raise an error or set cdfs to None and handle it in the endpoint
    cdfs = None
    print("Warning: cdf_values.pkl not found. CDF transformation will be skipped.")


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
    applies CDF transformation, and converts them into HSL color values.
    """
    if not HUGGING_FACE_API_KEY:
        raise HTTPException(status_code=500, detail="Hugging Face API key not configured on the server.")

    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": input_data.text, "parameters": {"top_k": 3}}

    # === START OF RETRY LOGIC ===
    max_retries = 5
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload, timeout=20)
            if response.status_code == 503:
                raise requests.exceptions.RequestException(f"Service Unavailable (503), attempt {attempt + 1}")
            response.raise_for_status()
            model_output = response.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Error connecting to Hugging Face: {e}. Retrying in {base_delay} seconds...")
                time.sleep(base_delay)
            else:
                print(f"All {max_retries} attempts failed. Error: {e}")
                raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face API after multiple attempts: {e}")
    # === END OF RETRY LOGIC ===

    # --- Parsing Logic ---
    try:
        label_to_vad_map = {"LABEL_0": "V", "LABEL_1": "A", "LABEL_2": "D"}
        scores = {}
        output_list = model_output[0] if isinstance(model_output[0], list) else model_output

        for item in output_list:
            generic_label = item.get("label")
            score_value = item.get("score")
            if generic_label in label_to_vad_map:
                vad_dimension = label_to_vad_map[generic_label]
                scores[vad_dimension] = score_value

        valence_score = scores.get('V', 3.0)
        arousal_score = scores.get('A', 3.0)
        dominance_score = scores.get('D', 3.0)

    except (IndexError, TypeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Could not parse VAD scores. Error: {e}, Response: {model_output}")

    # --- VAD Score Transformation using CDF ---
    if cdfs is None:
        raise HTTPException(status_code=500, detail="CDF data not loaded on the server.")

    # The cdf_v/a/d objects are interp1d functions, so we can call them directly.
    valence = cdf_v(valence_score)
    arousal = cdf_a(arousal_score)
    dominance = cdf_d(dominance_score)
    
    # Ensure the final values are clipped between 0.0 and 1.0
    valence = min(max(valence, 0.0), 1.0)
    arousal = min(max(arousal, 0.0), 1.0)
    dominance = min(max(dominance, 0.0), 1.0)

    # --- HSL Conversion ---
    hue = int(valence * 240)
    saturation = int(40 + (arousal * 60))
    lightness = int(30 + (dominance * 40))

    return HSLResponse(h=hue, s=saturation, l=lightness)