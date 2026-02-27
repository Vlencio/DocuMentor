import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import re
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from backend.prompts import prompt_1

from groq import Groq

load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')


class RagEngine:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.client = Groq()
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
    
    # --- Document Processing ---
    def clean_text(self, text:str) -> str:
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        
        return emoji_pattern.sub(r'', text)
    
    def get_chunks(self, document_path:str, chunk_size:float = 1000, chunk_overlap:float = 200, separators:List[str] | None = None):
        # Load the document
        loader = PyPDFLoader(document_path) #path for the document
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
            chunk_overlap=chunk_overlap, # Generaly 10% - 20% of the chunk size
            separators=separators,
            length_function=len
        )

        # Generate and return the chunks
        chunks = splitter.split_documents(documents)
        print(f"Documents split into {len(chunks)} chunks.")
        print("Enriching metadata.")
        chunks = self.enrich_chunks_metadata(chunks)
        print('Metadata enriched.')

        return chunks

    def detect_doc_type(self, text_sample: str) -> str:
        if "```" in text_sample and ('def' in text_sample or 'function' in text_sample):
            return 'code_docs'
        
        elif any(method in text_sample for method in ["POST", "GET", "PUT", "DELETE"]):
            return 'api_docs'
        
        return 'general_docs'

    def smart_separators(self, doc_type: str = "general") -> list[str]:
        separators_map = {
            "code_docs": [
                "\n## ",        # Headers markdown
                "\n### ",       
                "\n```\n",      # Blocos de código
                "\n\n",         # Parágrafos
                "\n",
                ". ",
            ],
            "api_docs": [
                "\n# ",         # Títulos principais
                "\n## ",        # Endpoints/seções
                "\nPOST ",      # Métodos HTTP
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
            ]
        }
        return separators_map.get(doc_type, separators_map['general'])

    def get_adaptative_chunk_size(self, text: str) -> int:
        if text.count('```') > 5:
            return 800
        
        avg_sentence_len = len(text) / (text.count('.') + 1)
        if avg_sentence_len > 50:
            return 1200
        
        return 1000

    def enrich_chunks_metadata(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.metadata['id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)

            chunk.metadata['has_code'] = '```' in chunk.page_content

            pedagogical_words = ['conceito', 'concept', 'entender', 'understand', 'funciona', 'works', 'exemplo', 'example', 'porque', 'why']
            chunk.metadata['is_conceptual'] = any(word in chunk.page_content.lower() for word in pedagogical_words)

            lines = chunk.page_content.split('\n')
            for line in lines[:3]:
                if line.startswith('#'):
                    chunk.metadata['section'] = line.strip('# ')
                    break
        
        return chunks

    # --- Document Retrieval -- -
    def retrieve_relevant_chunks(self, query, vector_store, k=5):
        chunks = vector_store.similarity_search(query, k=k)
        return chunks

    # --- Generation ---
    def build_message(self, system_prompt, context, user_query, chat_history) -> list[dict[str, None]]:
        
        # Detecta a língua e reforça no system prompt
        system_with_lang = system_prompt + f"\n\nIMPORTANT: The user's message is in the language you must reply in. Do NOT switch languages under any circumstance."
        
        message = [
            {
                "role": "system",
                "content": system_with_lang
            }
        ]

        if chat_history:
            message.extend(chat_history)

        # Wrapper neutro, sem forçar idioma
        final_prompt = (
            f"Context:\n{context}\n\n"
            f"{user_query}"  # ← Mensagem do usuário pura, sem wrapper em PT
        )

        message.append({"role": "user", "content": final_prompt})

        return message

    def format_context_llm(self, chunks = None , user_lvl: str | None = None) -> str:
        string = f'User level: {user_lvl}\n'

        if chunks:
            for chunk in chunks:
                string += f'Chunk metadata: {chunk.metadata}\nChunk page_content: {chunk.page_content}\n'

        return string

    def call_llm(self, messages) -> str | None:
        response = self.client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )

        return response.choices[0].message.content

    def generate_message(self, chat_history, user_query) -> str | None:
        context = self.format_context_llm(user_lvl='begginer')
        messages = self.build_message(system_prompt=prompt_1, context=context, chat_history=chat_history, user_query=user_query)
        response = self.call_llm(messages=messages)

        return response

    # --- Vector Store ---
    def get_vector_store(self, doc_name, store_name):
        doc_path = os.path.join(data_dir, 'documents', doc_name)
        chunks = self.get_chunks(doc_path)

        persistent_dir = os.path.join(data_dir, 'vectorstore', store_name)

        if not os.path.exists(persistent_dir):
            print(f"\n--- Creating vector store {store_name} ---")
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persistent_dir
            )
            print(f"\n--- Finished creating vector store {store_name} ---")
        
        else:
            vector_store = Chroma(
                persist_directory=persistent_dir,
                embedding_function=self.embeddings
            )
            print(f"\n--- Vector store {store_name} already exists. No need to initialize ---")
        
        self.vector_store = vector_store

    # --- Complete Pipeline ---
    def main_method(self, user_query) -> str | None:
        vector_store = self.get_vector_store(
            doc_name='eloz.pdf',
            store_name='eloz-api'
        )
        
        chunks = self.retrieve_relevant_chunks(query=user_query, vector_store=vector_store)
        formated_context = self.format_context_llm(chunks, 'begginer')
        message = self.build_message(system_prompt=prompt_1, context=formated_context, user_query=user_query, chat_history=[])
        response = self.call_llm(messages=message)
        
        return response
