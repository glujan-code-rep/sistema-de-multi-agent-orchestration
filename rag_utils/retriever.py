# /rag_utils/retriever.py

import os
from typing import Dict, Any, List

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma


class SimpleLocalEmbeddings:
    """Fallback embeddings usando hash simple - para pruebas locales sin API key."""

    def __init__(self):
        self.dimension = 1536

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embedding(text)

    def _hash_embedding(self, text: str) -> List[float]:
        import hashlib
        import numpy as np

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
        import os

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


class VectorRetriever:
    """
    Clase genérica para manejar la carga y recuperación de datos desde una base vectorial (Chroma).
    Esta clase es reutilizable por todos los agentes especializados.
    """

    def __init__(self, db_path: str):
        """
        Inicializa el retriever con la ruta de la DB.

        Args:
            db_path (str): La ruta al directorio donde se almacena la base vectorial.
        """
        self.db_path = db_path

        # 1. Cargar la base vectorial desde el disco
        try:
            # Usar sentence-transformers para mejor calidad de búsqueda
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            collection_name = os.path.basename(db_path)
            self.vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            print(
                f"Base vectorial '{collection_name}' cargada exitosamente desde: {db_path}"
            )
        except Exception as e:
            print(f"ERROR al cargar la base vectorial '{db_path}': {e}")
            # En un sistema real, aquí你应该 manejar el error de forma más robusta.
            self.vectorstore = None

    def retrieve(self, question: str, k: int = 4) -> str:
        """
        Realiza la búsqueda vectorial y formatea el contexto para el LLM.

        Args:
            question (str): La pregunta del usuario a buscar.
            k (int): El número de documentos relevantes a recuperar.

        Returns:
            str: El prompt completo listo para ser enviado al LLM.
        """
        if not self.vectorstore:
            return "Error: La base vectorial no está cargada. No se puede realizar la búsqueda."

        print(f"Buscando contexto para la pregunta: '{question}'")

        # 1. Realizar la búsqueda en la Vector Store
        docs = self.vectorstore.similarity_search(question, k=k)

        if not docs:
            return "No se encontraron documentos relevantes para esta consulta."

        # 2. Formatear el Contexto (El paso más importante del RAG)
        context = "\n---\n".join([doc.page_content for doc in docs])

        # 3. Retornar el contexto formateado (cada agente define su propio prompt)
        return context


# Nota: El __init__.py puede quedar vacío o con una simple declaración de paquete.
