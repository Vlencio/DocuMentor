import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from backend.rag_engine import RagEngine


# Page Configuration
st.set_page_config(page_title="DocuMentor", page_icon="🎓")
st.title("DocuMentor")
st.caption("Me mande suas dúvidas que eu te devolvo conhecimento.")

# RAG Inicialization
@st.cache_resource
def init_rag():
    rag = RagEngine()
    return rag
rag = init_rag()

# Creating chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if 'key' not in st.session_state:
    st.session_state.key = 'value'

# Showing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = rag.generate_message(user_query=prompt, chat_history=st.session_state.messages)
        
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
