import json
import os
import re

import anthropic
from langdetect import DetectorFactory, LangDetectException, detect_langs

from backend.config import ANTHROPIC_API_KEY, ANTHROPIC_EFFORT, ANTHROPIC_MODEL, BASE_DIR
from backend.tutor.prompts import prompt_1

# Make language detection give the same answer every time for the same text
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"en": "English", "pt": "Portuguese", "es": "Spanish"}

with open(os.path.join(BASE_DIR, "tools.json"), "r") as file:
    tools = json.load(file)

MAX_TOOLS_ROUND = 5
# Only the most recent messages are sent to the model, to keep each request's cost bounded
MAX_HISTORY_MESSAGES = 15
# Room for the model's thinking plus a full reply
MAX_TOKENS = 8000

PRIMM_STEP_NAMES = ["lesson not started", "Predict", "Run", "Investigate", "Modify", "Make"]
ONBOARDING_FIELDS = ["name", "favorite_language", "topic"]

FALLBACK_REPLY = "Desculpe, não consegui concluir a resposta. Tente novamente."
REFUSAL_REPLY = "Desculpe, não posso ajudar com esse pedido. Vamos voltar à programação?"


# Lines with these are treated as code. Quotes, colons and apostrophes are left out
# because normal sentences use them too ("I'm", "Example: ...")
CODE_CHARACTERS = set("=()[]{}<>\\`")
# A message shorter than this is never trusted to change the conversation's language
MIN_WORDS_TO_SWITCH = 4
# How sure the detector must be before the language changes
MIN_CONFIDENCE = 0.9
# Setting the first language is less risky than changing it, so it needs less evidence
MIN_WORDS_FIRST = 3
MIN_CONFIDENCE_FIRST = 0.6


def prose_only(text):
    """Remove code from a message, so only the student's own words are analysed."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    lines = [line for line in text.splitlines() if not CODE_CHARACTERS & set(line)]
    words = [word for word in " ".join(lines).split() if any(ch.isalpha() for ch in word)]
    return " ".join(words), len(words)


def detect_language(text, previous=None):
    """Return the conversation's language, switching only on clear evidence.

    The detector often gets short messages and code wrong ("no idea" comes back as
    Portuguese), and whatever it returns is sent to the model as an order. So the
    language only changes when a message has enough real words and the detector
    is very sure. Otherwise the previous language stays.
    """
    min_words = MIN_WORDS_FIRST if previous is None else MIN_WORDS_TO_SWITCH
    min_confidence = MIN_CONFIDENCE_FIRST if previous is None else MIN_CONFIDENCE

    prose, word_count = prose_only(text)
    if word_count < min_words:
        return previous

    try:
        candidates = detect_langs(prose)
    except LangDetectException:
        return previous

    for candidate in candidates:
        if candidate.lang in SUPPORTED_LANGUAGES and candidate.prob >= min_confidence:
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


def recent_history(chat_history) -> list[dict]:
    # Ignore anything that isn't a plain user/assistant text message
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chat_history or []
        if isinstance(msg, dict)
        and msg.get("role") in ("user", "assistant")
        and isinstance(msg.get("content"), str)
        and msg["content"].strip()
    ]

    history = history[-MAX_HISTORY_MESSAGES:]

    # The conversation sent to the model has to start with a user message
    while history and history[0]["role"] != "user":
        history.pop(0)

    return history


def build_message(context, user_query, chat_history, reply_language=None) -> list[dict]:
    # The system prompt goes in its own parameter, not in the messages
    message = recent_history(chat_history)

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


def response_text(response) -> str:
    # A response can start with thinking blocks, so read the text blocks by type
    return "".join(block.text for block in response.content if block.type == "text").strip()


def call_llm(messages, state) -> str:
    start_step = state["primm_step"]

    for _ in range(MAX_TOOLS_ROUND):
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            # The system prompt and tools are the same on every request: cache them
            system=[{"type": "text", "text": prompt_1, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            messages=messages,
            output_config={"effort": ANTHROPIC_EFFORT},
        )

        if response.stop_reason == "refusal":
            return REFUSAL_REPLY

        if response.stop_reason != "tool_use":
            # end_turn, or max_tokens with a partial reply
            return response_text(response) or FALLBACK_REPLY

        # Send back the whole assistant turn unchanged (thinking blocks included)
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                result = run_tool(block.name, block.input, state, start_step)
            except (KeyError, TypeError, AttributeError) as e:
                result = {"error": f"Invalid arguments: {e}"}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
                "is_error": "error" in result,
            })

        # All results go back together in one user message
        messages.append({"role": "user", "content": tool_results})

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
        context=context,
        chat_history=chat_history,
        user_query=user_query,
        reply_language=reply_language,
    )
    response = call_llm(messages=messages, state=state)

    return response, state


client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
