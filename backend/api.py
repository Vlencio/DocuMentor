from fastapi import FastAPI
from pydantic import BaseModel

from backend.llm.client import generate_message

app = FastAPI(debug=True)

class ChatRequest(BaseModel):
    user_query: str
    chat_history: list[dict] = []
    user_level: str = "intermediate"
    primm_step: int = 0
    user_profile: dict = {}

@app.post('/chat')
def chat(req: ChatRequest):
    response, state = generate_message(
        user_query=req.user_query,
        chat_history=req.chat_history,
        user_level=req.user_level,
        primm_step=req.primm_step,
        user_profile=req.user_profile
    )

    return {"response": response, **state}
