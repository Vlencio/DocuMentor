import json
import os

from groq import Groq
from langdetect import DetectorFactory, LangDetectException, detect_langs

from backend.config import BASE_DIR, GROQ_API_KEY, GROQ_MODEL, GROQ_REASONING_EFFORT
from backend.tutor.prompts import prompt_1

# Make language detection give the same answer every time for the same text
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en": "English", "pt": "Portuguese", "es": "Spanish"}

with open(os.path.join(BASE_DIR, "tools.json"), "r") as file:
    tools = json.load(file)

MAX_TOOLS_ROUND = 5

PRIMM_STEP_NAMES = ["lesson not started", "Predict", "Run", "Investigate", "Modify", "Make"]
ONBOARDING_FIELDS = ["name", "favorite_language", "topic"]

FALLBACK_REPLY = "Desculpe, não consegui concluir a resposta. Tente novamente."


def detect_language(text, previous=None):
    """Return the language of the text, or the previous one when it's unclear.

    Short messages like "Hi" are often misdetected, so only the supported
    languages count, and anything uncertain keeps the previous language.
    """
    try:
        candidates = detect_langs(text)
    except LangDetectException:
        return previous

    for candidate in candidates:
        if candidate.lang in SUPPORTED_LANGUAGES and candidate.prob >= 0.25:
            return SUPPORTED_LANGUAGES[candidate.lang]

    return previous


def run_tool(name, args, state, start_step):
    if name == "update_primm_step":
        new_step = args["primm_step"]
        # One stage per reply: the student has to answer before the lesson moves on
        if new_step > start_step + 1:
            return {
                "error": f"Can't jump from stage {start_step} to {new_step} in one reply. "
                f"Move to stage {start_step + 1} only and wait for the student's answer."
            }
        state["primm_step"] = new_step
        return {"ok": True, "primm_step": state["primm_step"]}

    if name == "update_user_information":
        value = args["value"].strip()
        if not value:
            return {"error": "Empty value not saved. Only save information the student actually gave."}
        state["user_profile"][args["key"]] = value
        return {"ok": True}

    return {"error": f"Unknown tool: {name}"}


def build_message(system_prompt, context, user_query, chat_history, reply_language=None) -> list[dict]:
    # gpt-oss reads app instructions from the "developer" role
    message = [{"role": "developer", "content": system_prompt}]

    if chat_history:
        message.extend(chat_history)

    # Context first, then the student's message, then the language reminder:
    # the end of the input weighs the most
    if reply_language:
        language_rule = f"Reply in {reply_language}."
    else:
        language_rule = "Reply in the same language as the text inside <student_message>."

    final_prompt = (
        f"{context}\n\n"
        f"<student_message>\n{user_query}\n</student_message>\n\n"
        f"{language_rule}"
    )

    message.append({"role": "user", "content": final_prompt})

    return message


def format_context_llm(user_profile, primm_step, user_lvl) -> str:
    missing = [field for field in ONBOARDING_FIELDS if field not in user_profile]

    string = "<learner_state>\n"
    string += f"Level: {user_lvl}\n"
    string += f"Profile: {json.dumps(user_profile, ensure_ascii=False)}\n"
    if missing:
        string += f"Onboarding: not complete. Missing: {', '.join(missing)}\n"
    else:
        string += "Onboarding: complete. Don't ask for this information again.\n"
        step_name = PRIMM_STEP_NAMES[primm_step] if 0 <= primm_step < len(PRIMM_STEP_NAMES) else "unknown"
        string += f"Current PRIMM stage: {primm_step} ({step_name})\n"
    string += "</learner_state>"

    return string


def call_llm(messages, state) -> str:
    start_step = state["primm_step"]

    for _ in range(MAX_TOOLS_ROUND):
        response = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            tools=tools,
            tool_choice="auto",
            reasoning_effort=GROQ_REASONING_EFFORT,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            if message.content:
                return message.content
            # gpt-oss sometimes answers only in its hidden reasoning: ask again
            continue

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            try:
                args = json.loads(call.function.arguments)
                result = run_tool(call.function.name, args, state, start_step)
            except (json.JSONDecodeError, KeyError) as e:
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return FALLBACK_REPLY


def generate_message(chat_history, user_query, user_level, primm_step, user_profile):
    state = {"primm_step": primm_step, "user_profile": dict(user_profile)}

    # Detected in code: the model switched languages at random when left to decide
    reply_language = detect_language(user_query, previous=state["user_profile"].get("reply_language"))
    if reply_language:
        state["user_profile"]["reply_language"] = reply_language

    context = format_context_llm(
        user_profile=state["user_profile"],
        primm_step=state["primm_step"],
        user_lvl=user_level,
    )
    messages = build_message(
        system_prompt=prompt_1,
        context=context,
        chat_history=chat_history,
        user_query=user_query,
        reply_language=reply_language,
    )
    response = call_llm(messages=messages, state=state)

    return response, state


client = Groq(api_key=GROQ_API_KEY)
