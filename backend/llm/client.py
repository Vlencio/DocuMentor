import json

from groq import Groq
from backend.config import GROQ_API_KEY, BASE_DIR

from backend.tutor.prompts import prompt_1

import os

primm_step = 0

def build_message(
    system_prompt, context, user_query, chat_history, primm_step
) -> list[dict[str, None]]:

    # Detecta a língua e reforça no system prompt
    system_with_lang = (
        system_prompt
        + f"\n\nIMPORTANT: The user's message is in the language you must reply in. Do NOT switch languages under any circumstance."
    )

    message = [{"role": "system", "content": system_with_lang}]

    if chat_history:
        message.extend(chat_history)

    # Wrapper neutro, sem forçar idioma
    final_prompt = (
        f"PRIMM Step: {primm_step}\n"
        f"Context:\n{context}\n\n"
        f"{user_query}"  # ← Mensagem do usuário pura, sem wrapper em PT
    )

    message.append({"role": "user", "content": final_prompt})

    return message

def format_context_llm(chunks=None, user_lvl: str | None = None) -> str:
    string = f"User level: {user_lvl}\n"

    if chunks:
        for chunk in chunks:
            string += f"Chunk metadata: {chunk.metadata}\nChunk page_content: {chunk.page_content}\n"

    return string

def call_llm(messages) -> str | None:
    with open(os.path.join(BASE_DIR, "tools.json"), "r") as file:
        tools = json.load(file)

    print(primm_step)

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        tools=tools,
        tool_choice="auto",
    )

    return response.choices[0].message.content

def generate_message(
    chat_history, user_query, user_level, primm_step
) -> str | None:
    context = format_context_llm(user_lvl=user_level)
    messages = build_message(
        system_prompt=prompt_1,
        context=context,
        chat_history=chat_history,
        user_query=user_query,
        primm_step=primm_step,
    )
    response = call_llm(messages=messages)

    return response

client = Groq(
    api_key=GROQ_API_KEY
)