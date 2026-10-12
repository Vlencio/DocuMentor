import re
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text: str) -> str:
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

def enrich_chunks_metadata(chunks):
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

def get_chunks(
    document_path: str,
    chunk_size: float = 1000,
    chunk_overlap: float = 200,
    separators: List[str] | None = None,
):
    # Load the document
    loader = PyPDFLoader(document_path)  # path for the document
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
        chunk_overlap=chunk_overlap,  # Generaly 10% - 20% of the chunk size
        separators=separators,
        length_function=len,
    )

    # Generate and return the chunks
    chunks = splitter.split_documents(documents)
    print(f"Documents split into {len(chunks)} chunks.")
    print("Enriching metadata.")
    chunks = enrich_chunks_metadata(chunks)
    print("Metadata enriched.")

    return chunks

def detect_doc_type(text_sample: str) -> str:
    if "```" in text_sample and ("def" in text_sample or "function" in text_sample):
        return "code_docs"

    elif any(method in text_sample for method in ["POST", "GET", "PUT", "DELETE"]):
        return "api_docs"

    return "general_docs"

def smart_separators(doc_type: str = "general") -> list[str]:
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

def get_adaptative_chunk_size(text: str) -> int:
    if text.count("```") > 5:
        return 800

    avg_sentence_len = len(text) / (text.count(".") + 1)
    if avg_sentence_len > 50:
        return 1200

    return 1000

