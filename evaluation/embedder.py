from typing import Dict

import numpy as np
from sentence_transformers import SentenceTransformer


def embed_documents(
    model_name: str, documents: Dict[str, str]
) -> Dict[str, np.ndarray]:
    """
    Embeds all documents using the specified model.
    Returns dict: doc_id -> embedding vector.
    """
    model = SentenceTransformer(model_name)
    texts = list(documents.values())
    embeddings = model.encode(
        texts, normalize_embeddings=True
    )  # Normalize for cosine sim
    return {doc_id: emb for doc_id, emb in zip(documents.keys(), embeddings)}


def embed_query(model_name: str, query: str) -> np.ndarray:
    """
    Embeds a single query using the specified model.
    """
    model = SentenceTransformer(model_name)
    return model.encode([query], normalize_embeddings=True)[0]
