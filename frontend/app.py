import os

import requests
import streamlit as st

API_URL = os.environ.get("DOCUMENTOR_API_URL", "http://127.0.0.1:8000")

USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🎓"

LEVELS = {
    "beginner": {
        "label": "🌱 Beginner",
        "caption": "I'm just starting",
        "details": [
            "Detailed explanations with analogies",
            "Core concepts always explained",
            "Code commented line by line",
            "More encouragement and context",
        ],
    },
    "intermediate": {
        "label": "🌿 Intermediate",
        "caption": "I have some experience",
        "details": [
            "Balance between concepts and practice",
            "Assumes basic knowledge",
            "Comments on the important lines",
            "Focus on good practices",
        ],
    },
    "advanced": {
        "label": "🌳 Advanced",
        "caption": "I'm experienced",
        "details": [
            "In-depth technical discussions",
            "Less basic context",
            "Edge cases and optimizations",
            "Performance and architecture",
        ],
    },
}

SUGGESTIONS = [
    "What is a function in Python?",
    "How does a for loop work?",
    "What's the difference between a list and a tuple?",
    "Explain recursion with an example",
]

PRIMM_STEPS = [
    ("Predict", "You predict what the code does"),
    ("Run", "Run it and compare with your prediction"),
    ("Investigate", "Explore how and why it works"),
    ("Modify", "Change the code to change its behavior"),
    ("Make", "Write your own code from scratch"),
]

# Page Configuration
st.set_page_config(page_title="DocuMentor", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 780px; }
    .dm-hero { text-align: center; padding: 2.5rem 0 1.5rem; }
    .dm-hero .dm-logo { font-size: 3rem; line-height: 1; }
    .dm-hero h1 { font-size: 2.4rem; margin: 0.5rem 0 0.25rem; padding: 0; }
    .dm-hero p { opacity: 0.7; font-size: 1.05rem; margin: 0; }
    .dm-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
    .dm-header .dm-logo { font-size: 1.6rem; }
    .dm-header h3 { margin: 0; padding: 0; }
    .dm-section-label {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em;
        text-transform: uppercase; opacity: 0.6; margin: 1.5rem 0 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def send_message(user_query, chat_history, user_level, primm_step, user_profile):
    response = requests.post(
        f"{API_URL}/chat",
        json={
            "user_query": user_query,
            "chat_history": chat_history,
            "user_level": user_level,
            "primm_step": primm_step,
            "user_profile": user_profile,
        },
        timeout=120,
    )
    response.raise_for_status()
    # {"response": ..., "primm_step": ..., "user_profile": ...}
    return response.json()


def reset_conversation():
    st.session_state.messages = []
    st.session_state.primm_step = 0
    st.session_state.user_profile = {}


# Creating chat history
if "messages" not in st.session_state:
    reset_conversation()

# Sessions started before user_profile existed
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "user_level" not in st.session_state:
    st.session_state.user_level = "intermediate"

with st.sidebar:
    st.markdown(
        '<div class="dm-header"><span class="dm-logo">🎓</span><h3>DocuMentor</h3></div>',
        unsafe_allow_html=True,
    )
    st.caption("Your programming tutor, using the PRIMM method.")

    st.markdown('<div class="dm-section-label">Your level</div>', unsafe_allow_html=True)
    level = st.radio(
        "Select your level:",
        options=list(LEVELS),
        format_func=lambda x: LEVELS[x]["label"],
        captions=[LEVELS[x]["caption"] for x in LEVELS],
        index=list(LEVELS).index(st.session_state.user_level),
        key="level_selector",
        label_visibility="collapsed",
    )

    if level != st.session_state.user_level:
        st.session_state.user_level = level
        st.toast(f"Level changed to {LEVELS[level]['label']}")

    with st.expander("What changes at this level?"):
        st.markdown("\n".join(f"- {item}" for item in LEVELS[level]["details"]))

    st.markdown('<div class="dm-section-label">How I teach</div>', unsafe_allow_html=True)
    with st.expander("The PRIMM method"):
        for i, (name, description) in enumerate(PRIMM_STEPS, start=1):
            st.markdown(f"**{i}. {name}** — {description}")

    st.divider()
    if st.button(
        "New conversation",
        icon=":material/refresh:",
        use_container_width=True,
        disabled=not st.session_state.messages,
    ):
        reset_conversation()
        st.rerun()

prompt = st.chat_input("Ask anything about programming...")
prompt = prompt or st.session_state.pop("pending_prompt", None)

# Empty state: welcome screen with suggestions
if not st.session_state.messages and not prompt:
    st.markdown(
        """
        <div class="dm-hero">
            <div class="dm-logo">🎓</div>
            <h1>DocuMentor</h1>
            <p>Send me your questions, and I'll help you find the answers yourself.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        if cols[i % 2].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_prompt = suggestion
            st.rerun()

# Showing chat history
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        error = None
        with st.spinner("Thinking..."):
            try:
                data = send_message(
                    user_query=prompt,
                    chat_history=st.session_state.messages[:-1],
                    user_level=st.session_state.user_level,
                    primm_step=st.session_state.primm_step,
                    user_profile=st.session_state.user_profile,
                )
            except (requests.RequestException, ValueError) as e:
                # ValueError: the server answered with something that isn't JSON
                error = e

        if error:
            # Drop the unanswered message so the user can try again
            st.session_state.messages.pop()
            status = getattr(getattr(error, "response", None), "status_code", None)
            if isinstance(error, requests.ConnectionError):
                st.error(
                    "Couldn't connect to the server. Check that the API is running.",
                    icon=":material/cloud_off:",
                )
            elif status == 429:
                wait = error.response.headers.get("Retry-After")
                st.warning(
                    f"Too many messages right now. Wait {wait + ' seconds' if wait else 'a few seconds'} and try again.",
                    icon=":material/hourglass_top:",
                )
            elif status == 503:
                st.warning(
                    "This demo has reached its usage limit. Thanks for trying it!",
                    icon=":material/savings:",
                )
            else:
                st.error(
                    "The server ran into an error. Please try again.",
                    icon=":material/error:",
                )
            with st.expander("Technical details"):
                st.code(f"{API_URL}\n{error}", language=None)
            st.stop()

        # Keep what the tools changed so the next message starts from it
        st.session_state.primm_step = data.get("primm_step", st.session_state.primm_step)
        st.session_state.user_profile = data.get("user_profile", st.session_state.user_profile)

        # Never store None in the history: it would be sent back on every request
        response = data.get("response") or "Sorry, I couldn't generate a reply. Please try again."
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
