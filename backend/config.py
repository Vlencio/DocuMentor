import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, '.env'))

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

data_dir = os.path.join(BASE_DIR, 'data')
os.makedirs(data_dir, exist_ok=True)

documents_dir = os.path.join(BASE_DIR, 'documents')
os.makedirs(documents_dir, exist_ok=True)