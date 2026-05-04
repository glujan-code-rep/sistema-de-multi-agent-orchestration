# rag_utils/embedder.py

import os
import hashlib
import numpy as np
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv
from typing import List


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge_bases")


class SimpleLocalEmbeddings:
    """Fallback embeddings usando hash simple - para pruebas locales sin API key."""

    def __init__(self):
        self.dimension = 1536  # Compatible con OpenAI

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        hash_bytes = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer(hash_bytes, dtype=np.uint8)
        arr = arr.astype(np.float32) / 255.0
        if len(arr) < self.dimension:
            arr = np.tile(arr, (self.dimension // len(arr)) + 1)[: self.dimension]
        return arr.tolist()


class LMEStudioEmbeddings:
    """Embeddings usando LM Studio API."""

    def __init__(
        self,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        base_url: str = "http://127.0.0.1:1234/v1",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")

    def _embed(self, text: str) -> List[float]:
        import requests

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {"input": text, "model": self.model}
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class SentenceTransformerEmbeddings:
    """Embeddings usando sentence-transformers (all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()


def get_embeddings():
    """Obtiene el embeddings appropriate según disponibilidad de API key."""
    # Preferir sentence-transformers
    try:
        print("Usando sentence-transformers (all-MiniLM-L6-v2) para embeddings...")
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: sentence-transformers no disponibles: {e}")

    # Fallback a OpenAI si hay API key válida
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not api_key.startswith("sk-dummy"):
        try:
            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as e:
            print(f"Warning: OpenAI embeddings no disponibles: {e}")

    print("Usando embeddings locales (hash-based) para pruebas...")
    return SimpleLocalEmbeddings()


def build_vector_db(
    source_file: str, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50
):
    """
    Carga un archivo de datos, lo segmenta, genera embeddings y lo almacena en ChromaDB.

    Args:
        source_file (str): Ruta al archivo fuente (ej. data/rrh_policies.md)
        db_path (str): Directorio donde se guardará la base vectorial (ej. knowledge_bases/rrh_vector_db)
        chunk_size (int): Tamaño de cada segmento en caracteres
        chunk_overlap (int): Solapamiento entre segmentos consecutivos
    """
    # Asegurar directorio de salida
    os.makedirs(db_path, exist_ok=True)

    ext = os.path.splitext(source_file)[1].lower()

    if ext == ".csv":
        loader = CSVLoader(file_path=source_file)
    else:
        loader = TextLoader(source_file, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "---", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    print(f"{len(chunks)} segmentos generados desde '{source_file}'")

    embeddings = get_embeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name=os.path.basename(db_path),
    )
    print(f"Base vectorial guardada en: {db_path}")


def build_all():
    """Construye todas las bases vectoriales desde los archivos en data/."""

    # RRH policies (.md)
    rrh_file = os.path.join(DATA_DIR, "rrh_policies.md")
    rrh_db = os.path.join(KNOWLEDGE_DIR, "rrh_vector_db")
    if os.path.exists(rrh_file):
        build_vector_db(rrh_file, rrh_db)

    # IT FAQs (.txt)
    it_file = os.path.join(DATA_DIR, "it_faqs.txt")
    it_db = os.path.join(KNOWLEDGE_DIR, "it_vector_db")
    if os.path.exists(it_file):
        build_vector_db(it_file, it_db)

    # Finance rules (.csv)
    finance_file = os.path.join(DATA_DIR, "finance_rules.csv")
    finance_db = os.path.join(KNOWLEDGE_DIR, "finance_vector_db")
    if os.path.exists(finance_file):
        build_vector_db(finance_file, finance_db)


if __name__ == "__main__":
    print("Construyendo bases vectoriales desde data/...\n")
    build_all()
