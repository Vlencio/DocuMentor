import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import List

import anthropic
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.prompts import prompt_1

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")


class RagEngine:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.primm_step = None

    # --- Document Processing ---
    def clean_text(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )

        return emoji_pattern.sub(r"", text)

    def get_chunks(
        self,
        document_path: str,
        chunk_size: float = 1000,
        chunk_overlap: float = 200,
        separators: List[str] | None = None,
    ):
        # Load the document
        loader = PyPDFLoader(document_path)  # path for the document
        documents = loader.load()

        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)

        if not separators:
            sample_text = " ".join([doc.page_content for doc in documents[:3]])
            doc_type = self.detect_doc_type(sample_text)
            separators = self.smart_separators(doc_type=doc_type)

        if not chunk_size:
            sample_text = " ".join([doc.page_content for doc in documents])
            chunk_size = self.get_adaptative_chunk_size(sample_text)
            chunk_overlap = chunk_size * 0.13

        # Create the splitter using RecursiveCharacterTextSplitter module, arguments can be changed
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,  # Generaly 10% - 20% of the chunk size
            separators=separators,
            length_function=len,
        )

        # Generate and return the chunks
        chunks = splitter.split_documents(documents)
        print(f"Documents split into {len(chunks)} chunks.")
        print("Enriching metadata.")
        chunks = self.enrich_chunks_metadata(chunks)
        print("Metadata enriched.")

        return chunks

    def detect_doc_type(self, text_sample: str) -> str:
        if "```" in text_sample and ("def" in text_sample or "function" in text_sample):
            return "code_docs"

        elif any(method in text_sample for method in ["POST", "GET", "PUT", "DELETE"]):
            return "api_docs"

        return "general_docs"

    def smart_separators(self, doc_type: str = "general") -> list[str]:
        separators_map = {
            "code_docs": [
                "\n## ",  # Headers markdown
                "\n### ",
                "\n```\n",  # Blocos de código
                "\n\n",  # Parágrafos
                "\n",
                ". ",
            ],
            "api_docs": [
                "\n# ",  # Títulos principais
                "\n## ",  # Endpoints/seções
                "\nPOST ",  # Métodos HTTP
                "\nGET ",
                "\n\n",
                "\n",
            ],
            "general": [
                "\n\n",
                "\n",
                ". ",
                "! ",
                "? ",
            ],
        }
        return separators_map.get(doc_type, separators_map["general"])

    def get_adaptative_chunk_size(self, text: str) -> int:
        if text.count("```") > 5:
            return 800

        avg_sentence_len = len(text) / (text.count(".") + 1)
        if avg_sentence_len > 50:
            return 1200

        return 1000

    def enrich_chunks_metadata(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

            chunk.metadata["has_code"] = "```" in chunk.page_content

            pedagogical_words = [
                "conceito",
                "concept",
                "entender",
                "understand",
                "funciona",
                "works",
                "exemplo",
                "example",
                "porque",
                "why",
            ]
            chunk.metadata["is_conceptual"] = any(
                word in chunk.page_content.lower() for word in pedagogical_words
            )

            lines = chunk.page_content.split("\n")
            for line in lines[:3]:
                if line.startswith("#"):
                    chunk.metadata["section"] = line.strip("# ")
                    break

        return chunks

    # --- Document Retrieval -- -
    def retrieve_relevant_chunks(self, query, vector_store, k=5):
        chunks = vector_store.similarity_search(query, k=k)
        return chunks

    # --- Generation ---
    def build_message(
        self, system_prompt, context, user_query, chat_history, primm_step
    ) -> tuple[str, list]:
        system_with_lang = (
            system_prompt
            + f"\n\nIMPORTANT: The user's message is in the language you must reply in. Do NOT switch languages under any circumstance."
        )

        messages = []
        for m in (chat_history or []):
            content = m.get("content")
            role = m.get("role")
            if content is None:
                continue
            # Anthropic API requires array content for assistant messages in tool-use conversations
            if role == "assistant" and isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append({"role": role, "content": content})

        final_prompt = (
            f"PRIMM Step: {primm_step}\n"
            f"Context:\n{context}\n\n"
            f"{user_query}"
        )
        messages.append({"role": "user", "content": final_prompt})

        return system_with_lang, messages

    def format_context_llm(self, chunks=None, user_lvl: str | None = None) -> str:
        string = f"User level: {user_lvl}\n"

        if chunks:
            for chunk in chunks:
                string += f"Chunk metadata: {chunk.metadata}\nChunk page_content: {chunk.page_content}\n"

        return string

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "update_primm_step":
            self.update_primm_step(str(tool_input.get("primm_step", self.primm_step)))
            return f"PRIMM step updated to {tool_input.get('primm_step')}"
        if tool_name == "update_user_information":
            return f"Updated {tool_input.get('key')} = {tool_input.get('value')}"
        return "Tool executed."

    def call_llm(self, system: str, messages: list) -> str | None:
        with open("backend/tools.json", "r") as file:
            tools = json.load(file)

        current_messages = list(messages)

        while True:
            response = self.client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=4096,
                system=system,
                tools=tools,
                messages=current_messages,
            )

            if response.stop_reason != "tool_use":
                for block in response.content:
                    if block.type == "text":
                        return block.text
                return None

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            content_dicts = []
            for block in response.content:
                if block.type == "text":
                    content_dicts.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    content_dicts.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
            current_messages.append({"role": "assistant", "content": content_dicts})
            current_messages.append({"role": "user", "content": tool_results})

    def generate_message(
        self, chat_history, user_query, user_level, primm_step
    ) -> str | None:
        context = self.format_context_llm(user_lvl=user_level)
        system, messages = self.build_message(
            system_prompt=prompt_1,
            context=context,
            chat_history=chat_history,
            user_query=user_query,
            primm_step=self.primm_step,
        )
        return self.call_llm(system=system, messages=messages)

    # --- Vector Store ---
    def get_vector_store(self, doc_name, store_name):
        doc_path = os.path.join(data_dir, "documents", doc_name)
        chunks = self.get_chunks(doc_path)

        persistent_dir = os.path.join(data_dir, "vectorstore", store_name)

        if not os.path.exists(persistent_dir):
            print(f"\n--- Creating vector store {store_name} ---")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persistent_dir,
            )
            print(f"\n--- Finished creating vector store {store_name} ---")

        else:
            vector_store = Chroma(
                persist_directory=persistent_dir, embedding_function=self.embeddings
            )
            print(
                f"\n--- Vector store {store_name} already exists. No need to initialize ---"
            )

        self.vector_store = vector_store

    # --- Complete Pipeline ---
    def main_method(self, user_query) -> str | None:
        vector_store = self.get_vector_store(doc_name="eloz.pdf", store_name="eloz-api")

        chunks = self.retrieve_relevant_chunks(
            query=user_query, vector_store=vector_store
        )
        formated_context = self.format_context_llm(chunks, "begginer")
        system, messages = self.build_message(
            system_prompt=prompt_1,
            context=formated_context,
            user_query=user_query,
            chat_history=[],
            primm_step="0",
        )
        return self.call_llm(system=system, messages=messages)

    # --- Tools ---
    def update_primm_step(self, primm_step: str):
        self.primm_step = primm_step
