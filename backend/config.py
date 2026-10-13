import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, '.env'))

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-120b')
# How much the model thinks before answering: "low", "medium" or "high"
GROQ_REASONING_EFFORT = os.environ.get('GROQ_REASONING_EFFORT', 'medium')