import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from backend.rag_engine import RagEngine


# Page Configuration
st.set_page_config(page_title="DocuMentor", page_icon="🎓")
st.title("DocuMentor")
st.caption("Me mande suas dúvidas, eu te devolvo conhecimento.")

# RAG Inicialization
@st.cache_resource
def init_rag():
    rag = RagEngine()
    return rag
rag = init_rag()

# Creating chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if  "user_level" not in st.session_state:
    st.session_state.user_level = "intermediate"

if "level_changed" not in st.session_state:
    st.session_state.level_changed = False

with st.sidebar:
    st.header("⚙️ Config")

    st.subheader("User Level")
    level = st.radio(
        "Selecione seu nível:",
        options=["beginner", "intermediate", "advanced"],
        format_func=lambda x: {
            "beginner": "🌱 Iniciante - Estou começando",
            "intermediate": "🌿 Intermediário - Tenho experiência",
            "advanced": "🌳 Avançado - Sou experiente"
        }[x],
        index=["beginner", "intermediate", "advanced"].index(
            st.session_state.user_level
        ),
        key="level_selector"
    )
    
    # Detectar mudança de nível
    if level != st.session_state.user_level:
        st.session_state.user_level = level
        st.session_state.level_changed = True
        st.success(f"Nível alterado para: {level}")
    
    # Descrição do nível selecionado
    level_descriptions = {
        "beginner": """
        **🌱 Modo Iniciante:**
        - Explicações detalhadas com analogias
        - Conceitos fundamentais sempre explicados
        - Código com comentários linha a linha
        - Mais encorajamento e contexto
        """,
        "intermediate": """
        **🌿 Modo Intermediário:**
        - Equilíbrio entre conceito e implementação
        - Assume conhecimento básico
        - Código com comentários principais
        - Foco em boas práticas
        """,
        "advanced": """
        **🌳 Modo Avançado:**
        - Discussões técnicas profundas
        - Menos contexto básico
        - Edge cases e otimizações
        - Performance e arquitetura
        """
    }
    
    with st.expander("ℹ️ Sobre este nível"):
        st.markdown(level_descriptions[level])

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
            response = rag.generate_message(user_query=prompt, chat_history=st.session_state.messages, user_level=st.session_state.user_level)
        
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
