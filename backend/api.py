import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.llm.client import generate_message

app = FastAPI()

class ChatRequest(BaseModel):
    user_query: str
    chat_history: list[dict] = []
    user_level: str = "intermediate"
    primm_step: int = 0
    user_profile: dict = {}

@app.post('/chat')
def chat(req: ChatRequest):
    try:
        response, state = generate_message(
            user_query=req.user_query,
            chat_history=req.chat_history,
            user_level=req.user_level,
            primm_step=req.primm_step,
            user_profile=req.user_profile
        )
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after")
        raise HTTPException(
            status_code=429,
            detail="rate_limited",
            headers={"Retry-After": retry_after} if retry_after else None,
        )
    except anthropic.BadRequestError as e:
        # Happens when the prepaid credits run out
        if "credit balance" in str(e.message).lower():
            raise HTTPException(status_code=503, detail="out_of_credits")
        raise HTTPException(status_code=502, detail="model_error")
    except (anthropic.APIStatusError, anthropic.APIConnectionError):
        raise HTTPException(status_code=502, detail="model_error")

    return {"response": response, **state}
