import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import time # Import the time module for delays

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

    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    payload = {"inputs": input_data.text, "parameters": {"top_k": 3}}

    # === START OF NEW RETRY LOGIC ===
    max_retries = 5
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Make the request to the Hugging Face API
            response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload, timeout=20) # 20-second timeout

            # If the server is still starting up, it might return a 503
            if response.status_code == 503:
                # Raise an exception to trigger the retry logic
                raise requests.exceptions.RequestException(f"Service Unavailable (503), attempt {attempt + 1}")

            # If the request was successful (e.g., status 200), break the loop
            response.raise_for_status()
            model_output = response.json()
            break # Exit the loop on success

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                # Wait before the next attempt
                print(f"Error connecting to Hugging Face: {e}. Retrying in {base_delay} seconds...")
                time.sleep(base_delay)
            else:
                # If all retries fail, raise the final error
                print(f"All {max_retries} attempts failed. Error: {e}")
                raise HTTPException(status_code=503, detail=f"Error communicating with Hugging Face API after multiple attempts: {e}")
    # === END OF NEW RETRY LOGIC ===


    # --- Parsing Logic (No changes needed here) ---
    try:
        label_to_vad_map = {
            "LABEL_0": "V",
            "LABEL_1": "A",
            "LABEL_2": "D"
        }
        scores = {}
        # The model output is sometimes nested, so we check for that
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

        MAX_SCORE = 5.0
        MIN_SCORE = 1.0
        valence = (valence_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        arousal = (arousal_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        dominance = (dominance_score - MIN_SCORE) / (MAX_SCORE - MIN_SCORE)
        
        valence = min(max(valence, 0.0), 1.0)
        arousal = min(max(arousal, 0.0), 1.0)
        dominance = min(max(dominance, 0.0), 1.0)

    except (IndexError, TypeError, KeyError) as e:
        raise HTTPException(status_code=500, detail=f"Could not parse VAD scores. Error: {e}, Response: {model_output}")
    
    # --- HSL Conversion (No changes needed here) ---
    hue = int(valence * 240)
    saturation = int(40 + (arousal * 60))
    lightness = int(30 + (dominance * 40))

    return HSLResponse(h=hue, s=saturation, l=lightness)