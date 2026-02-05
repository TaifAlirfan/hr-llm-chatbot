import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DB_PATH = BASE_DIR / "db" / "hr.db"
TABLE_NAME = "employees"

# Mode: "groq" or "local"
MODEL_MODE = os.getenv("MODEL_MODE", "groq").strip().lower()

# Groq (OpenAI-compatible)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Local model (HF)
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "google/gemma-2b-it")
HF_TOKEN = os.getenv("HF_TOKEN", "")

