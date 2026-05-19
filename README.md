# DocuMentor

An AI-powered programming tutor that teaches using the **PRIMM methodology** — guiding learners step by step through Predict, Run, Investigate, Modify, and Make. Built with RAG (Retrieval-Augmented Generation) so it can answer questions grounded in your own documents.

## What it does

- Loads PDF documents and indexes them into a vector store (Chroma + HuggingFace embeddings)
- Answers programming questions using context retrieved from those documents
- Applies the PRIMM teaching framework one step at a time — no spoilers until the learner earns them
- Adapts tone and depth to three user levels: **Beginner**, **Intermediate**, and **Advanced**
- Responds in whatever language the user writes in (Portuguese or English, etc.)

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent, local) |
| Document Loading | LangChain + PyPDFLoader |

## Project Structure

```
DocuMentor/
├── backend/
│   ├── rag_engine.py        # Core RAG pipeline and LLM calls
│   ├── prompts.py           # System prompt with PRIMM instructions
│   ├── document_processor.py
│   └── main.py
├── frontend/
│   └── app.py               # Streamlit UI
└── data/
    ├── documents/           # Place your PDFs here
    └── vectorstore/         # Auto-generated Chroma indexes
```

## Getting Started

### 1. Install dependencies

```bash
pip install streamlit langchain langchain-chroma langchain-community \
            langchain-huggingface langchain-text-splitters \
            groq python-dotenv pypdf
```

### 2. Set your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 3. Add a PDF document

Place any PDF you want the tutor to reference inside `data/documents/`. The vector store is created automatically on first run.

### 4. Run the app

```bash
streamlit run frontend/app.py
```

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
