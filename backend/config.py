import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, '.env'))

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-haiku-5-5')
# How much the model thinks before answering: "low", "medium" or "high"
ANTHROPIC_EFFORT = os.environ.get('ANTHROPIC_EFFORT', 'medium')
