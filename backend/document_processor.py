import os
from typing import List
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, 'data')

def clean_text(text:str) -> str:
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r'', text)

def get_chunks(document_path:str, chunk_size:int = 1000, chunk_overlap:int = 200, separators:List[str] = None):
    """
    Transforms a PDF document into smaller text chunks for processing.
    
    This function loads a PDF document from the specified path and splits it into 
    manageable chunks using LangChain's RecursiveCharacterTextSplitter. The splitting 
    strategy aims to maintain semantic coherence by preferring certain separators 
    (like paragraph breaks and markdown headers) over arbitrary character positions.
    
    Args:
        document_path (str): The file path to the PDF document to be processed.
        chunk_size (int, optional): The maximum size of each text chunk in characters. 
            Defaults to 1200 if not specified.
        chunk_overlap (int, optional): The number of characters that consecutive chunks 
            should overlap. This helps maintain context across chunk boundaries. 
            Defaults to 150 if not specified.
        separators (List[str], optional): A list of separator strings used to split the 
            text, in order of preference. Defaults to ["\\n\\n", "\\n\\n##", "\\n\\n```", "\\n"] 
            if not specified.
    
    Returns:
        List[Document]: A list of LangChain Document objects, each representing a chunk 
            of the original PDF with its content and metadata.
    
    Example:
        >>> chunks = get_chunks("path/to/document.pdf")
        >>> chunks = get_chunks("report.pdf", chunk_size=1000, chunk_overlap=100)
        >>> custom_seps = ["\\n\\n", "\\n", ". ", " "]
        >>> chunks = get_chunks("doc.pdf", separators=custom_seps)
    
    Note:
        - The function uses PyPDFLoader to extract text from PDF files
        - Default separators are optimized for markdown-formatted documents
        - Larger overlap values provide better context but may increase redundancy
    """

    # Load the document
    loader = PyPDFLoader(document_path) #path for the document
    documents = loader.load()

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    if not separators:
        sample_text = " ".join([doc.page_content for doc in documents[:3]])
        doc_type = detect_doc_type(sample_text)
        separators = smart_separators(doc_type=doc_type)
    
    if not chunk_size:
        sample_text = " ".join([doc.page_content for doc in documents])
        chunk_size = get_adaptative_chunk_size(sample_text)
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
    chunks = enrich_chunks_metadata(chunks)
    print('Metadata enriched.')

    return chunks
    
def get_vector_store(doc_name, store_name):
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

    doc_path = os.path.join(data_dir, 'documents', doc_name)
    chunks = get_chunks(doc_path)

    persistent_dir = os.path.join(data_dir, 'vectorstore', store_name)

    if not os.path.exists(persistent_dir):
        print(f"\n--- Creating vector store {store_name} ---")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persistent_dir
        )
        print(f"\n--- Finished creating vector store {store_name} ---")
    
    else:
        vector_store = Chroma(
            persist_directory=persistent_dir,
            embedding_function=embeddings
        )
        print(f"\n--- Vector store {store_name} already exists. No need to initialize ---")
    
    return vector_store

def detect_doc_type(text_sample: str) -> str:
    if "```" in text_sample and ('def' in text_sample or 'function' in text_sample):
        return 'code_docs'
    
    elif any(method in text_sample for method in ["POST", "GET", "PUT", "DELETE"]):
        return 'api_docs'
    
    return 'general_docs'

def smart_separators(doc_type: str = "general") -> dict:
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

def get_adaptative_chunk_size(text: str) -> int:
    if text.count('```') > 5:
        return 800
    
    avg_sentence_len = len(text) / (text.count('.') + 1)
    if avg_sentence_len > 50:
        return 1200
    
    return 1000

def enrich_chunks_metadata(chunks):
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


if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')


    vector_store = get_vector_store(
        doc_name='eloz.pdf',
        store_name='eloz-api'
    )

    
    print('\n=== Iniciando busca ===')
    results = vector_store.similarity_search("Como eu envio mensagens?", k=3)
    print(f'Encontrados {len(results)} resultados\n')
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:200])



