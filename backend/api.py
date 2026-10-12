from fastapi import FastAPI
from pydantic import BaseModel

from backend.llm.client import generate_message

app = FastAPI(debug=True)

class ChatRequest(BaseModel):
    user_query: str
    chat_history: list[dict] = []
    user_lvl: str = "intermediate"
    primm_step: int = 0

@app.post('/chat')
def chat(req: ChatRequest):
    response = generate_message(
        user_query=req.user_query,
        chat_history=req.chat_history,
        user_level=req.user_lvl,
        primm_step=req.primm_step
    )

    return {"response": response}
