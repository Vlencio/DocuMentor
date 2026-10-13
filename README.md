# DocuMentor

An AI-powered programming tutor that teaches using the **PRIMM methodology** — guiding learners step by step through Predict, Run, Investigate, Modify, and Make.

## What it does

- Applies the PRIMM teaching framework one stage at a time — no spoilers until the learner earns them
- Gets to know the learner first (name, favorite language, topic) and builds every lesson around them
- Adapts tone and depth to three user levels: **Beginner**, **Intermediate**, and **Advanced**
- Replies in the language the learner writes in (English, Portuguese or Spanish)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Groq API (`openai/gpt-oss-120b`) with tool calling |
| Language detection | langdetect |

## How it works

The Streamlit frontend sends each message to the FastAPI backend, together with the conversation history and the learner's state (current PRIMM stage and profile). The backend calls the model, which uses two tools to update that state:

- `update_user_information` — saves the learner's name, favorite language and topic
- `update_primm_step` — moves the lesson to the next PRIMM stage (one stage per reply)

The updated state goes back to the frontend, which sends it again with the next message. The API itself keeps no data between requests.

## Project Structure

```
DocuMentor/
├── backend/
│   ├── api.py               # FastAPI app (POST /chat)
│   ├── config.py            # Settings loaded from backend/.env
│   ├── tools.json           # Tool definitions for the model
│   ├── llm/
│   │   └── client.py        # Groq calls, tool loop, learner state
│   └── tutor/
│       └── prompts.py       # System prompt with the PRIMM instructions
├── frontend/
│   └── app.py               # Streamlit UI
└── .streamlit/
    └── config.toml          # Theme
```

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Create a `backend/.env` file:

```
GROQ_API_KEY=your_key_here
```

Optional settings for the same file:

```
GROQ_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=medium
```

### 3. Run the backend and the frontend

From the project root, in two terminals:

```bash
uvicorn backend.api:app --reload
```

```bash
streamlit run frontend/app.py
```

The frontend expects the API at `http://127.0.0.1:8000`. To use another address, set the `DOCUMENTOR_API_URL` environment variable.

## PRIMM Methodology

PRIMM is a structured approach to teaching programming:

| Step | What happens |
|---|---|
| **Predict** | Student guesses what a code snippet does |
| **Run** | The actual output is revealed and discussed |
| **Investigate** | Student breaks down and explains the code |
| **Modify** | Student adapts the code for a new goal |
| **Make** | Student writes their own implementation |

DocuMentor advances through these steps in order, only moving forward when the learner is ready.

## User Levels

Select your level in the sidebar:

- **Beginner** — analogies, line-by-line comments, encouragement
- **Intermediate** — balanced theory and practice, assumes basic knowledge
- **Advanced** — deep technical discussion, edge cases, performance focus
