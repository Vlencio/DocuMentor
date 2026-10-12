import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from backend.ingestion.chunker import get_chunks

from backend.config import data_dir

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vector_store(doc_name, store_name):
    doc_path = os.path.join(data_dir, "documents", doc_name)
    chunks = get_chunks(doc_path)

    persistent_dir = os.path.join(data_dir, "vectorstore", store_name)

    if not os.path.exists(persistent_dir):
        print(f"\n--- Creating vector store {store_name} ---")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persistent_dir,
        )
        print(f"\n--- Finished creating vector store {store_name} ---")

    else:
        vector_store = Chroma(
            persist_directory=persistent_dir, embedding_function=embeddings
        )
        print(
            f"\n--- Vector store {store_name} already exists. No need to initialize ---"
        )

    return vector_store

def retrieve_relevant_chunks(query, vector_store, k=5):
    chunks = vector_store.similarity_search(query, k=k)
    return chunks
