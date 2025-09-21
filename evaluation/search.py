from typing import Dict

import numpy as np


def cosine_similarity(
    query_emb: np.ndarray, doc_embs: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Computes cosine sim between query and all doc embeddings.
    Returns dict: doc_id -> similarity score.
    """
    similarities = {}
    for doc_id, doc_emb in doc_embs.items():
        similarities[doc_id] = np.dot(query_emb, doc_emb)  # Since normalized
    return similarities


def retrieve_top_doc(similarities: Dict[str, float]) -> str:
    """
    Returns the doc_id with the highest similarity score.
    """
    return max(similarities, key=similarities.get)
