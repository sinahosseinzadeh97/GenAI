"""
Package initializer: loads .env so that all internal modules
have access to OPENAI_API_KEY, SENDGRID_API_KEY, etc.
"""
from pathlib import Path
from dotenv import load_dotenv

# ریشهٔ پروژه = یک پوشه بالاتر از همین فایل
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
