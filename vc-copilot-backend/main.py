import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# --- Local Imports ---
# This correctly imports the api_router where all your endpoints (/analyze, /find-matches, etc.) live.
from app.api.endpoints import api_router
# This correctly imports the dictionary that will hold the ML models.
from app.core.ml import ml_models

# --- Lifespan Management (Your Correct Logic) ---
# This is the best practice for loading models. It ensures they are loaded
# only once when the application starts, which is efficient and correct.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles application startup and shutdown events."""
    # --- Code to run on startup ---
    logging.info("Loading ML models into application state...")
    try:
        ml_models["bert"] = SentenceTransformer('all-MiniLM-L6-v2')
        ml_models["sentiment"] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", return_all_scores=True)
        ml_models["ner"] = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        logging.info("ML models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading ML models: {e}")
    
    yield # The application is now running
    
    # --- Code to run on shutdown ---
    ml_models.clear()
    logging.info("ML models cleared.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="VC Co-Pilot & Founders Fuel API",
    version="1.0.1",
    description="Unified API for pitch deck analysis and investor matching.",
    lifespan=lifespan  # This correctly registers the startup/shutdown logic
)

# --- CORS Middleware Configuration (The Fix for Browser Errors) ---
# This is the security configuration that allows your React frontend
# (running on localhost:5173) to communicate with this backend.
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    # You can add your production frontend URL here later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # Specifies which websites are allowed to connect.
    allow_credentials=True,      # Allows cookies to be included in requests.
    allow_methods=["*"],         # Allows all HTTP methods (GET, POST, etc.).
    allow_headers=["*"],         # Allows all request headers.
)

# --- API Router Inclusion ---
# This line correctly attaches all the API endpoints defined in your endpoints.py file.
# All your routes will be available under the /api prefix (e.g., /api/find-matches).
app.include_router(api_router, prefix="/api")

# --- Root Endpoint for Health Check ---
@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Welcome to the Founders Fuel API"}