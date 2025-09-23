import json
import os
from typing import Dict, List, Literal

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import AutoTokenizer

from .strategies import LongTextStrategy

# --- Data Loading ---


def load_documents(data_dir: str) -> Dict[str, str]:
    """Loads documents from a directory into a dictionary."""
    documents = {}
    for filename in tqdm(os.listdir(data_dir), desc="Loading Documents"):
        if filename.endswith(".txt"):
            doc_id = filename.split(".")[0]
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                documents[doc_id] = f.read()
    return documents


def load_queries(queries_path: str) -> List[Dict[str, str]]:
    """Loads queries from a .jsonl file."""
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_mock_data() -> tuple[Dict[str, str], List[Dict[str, str]]]:
    """Generates mock data for demonstration if real data is not found."""
    print("No 'data' directory found. Generating mock data.")
    # Mock Documents
    doc1_text = "The solar system consists of the Sun and the objects that orbit it. The largest objects are the eight planets. Jupiter is the largest planet, while Mercury is the smallest. Earth is the third planet from the Sun and the only known celestial body to harbor life."
    doc2_text = "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents'. AI applications include advanced web search engines, recommendation systems, and autonomous cars."
    documents = {
        "doc1": " ".join([doc1_text] * 20),  # Make it long
        "doc2": " ".join([doc2_text] * 20),
    }
    # Mock Queries
    queries = [
        {"query": "Which planet is the smallest?", "source_doc_id": "doc1"},
        {
            "query": "What are some applications of artificial intelligence?",
            "source_doc_id": "doc2",
        },
        {"query": "Tell me about life on other planets.", "source_doc_id": "doc1"},
    ]
    return documents, queries


# --- Core Benchmarking Functions ---


def embed_corpus(
    documents: Dict[str, str],
    strategy: LongTextStrategy,
    model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    embedding_type: Literal["query", "document"] = "document",
) -> Dict[str, np.ndarray]:
    """
    Embeds the entire corpus of documents using a given strategy.

    Returns:
        A dictionary mapping doc_id to its embedding vector.
    """
    doc_embeddings = {}
    for doc_id, text in tqdm(
        documents.items(), desc=f"Embedding Corpus with {strategy.__class__.__name__}"
    ):
        doc_embeddings[doc_id] = strategy.embed(
            text, model, tokenizer, embedding_type=embedding_type
        )
    return doc_embeddings

def embed_corpus_batch(
    documents: Dict[str, str],
    strategy: LongTextStrategy,
    model: SentenceTransformer,
    tokenizer: AutoTokenizer,
    embedding_type: Literal["query", "document"] = "document",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    """
    Embeds the entire corpus of documents using a given strategy in batch mode for performance.

    Args:
        documents (Dict[str, str]): A dictionary of document IDs to text content.
        strategy (LongTextStrategy): The strategy to use for handling long texts.
        model (SentenceTransformer): The Sentence Transformer model.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        embedding_type (Literal["query", "document"], optional): Type of embedding. Defaults to "document".
        batch_size (int, optional): The number of texts to process at once. Tune for your GPU. Defaults to 32.

    Returns:
        A dictionary mapping doc_id to its embedding vector.
    """
    print(f"Embedding corpus with {len(documents)} documents using {strategy.__class__.__name__}...")

    # Prepare lists of IDs and texts to maintain order
    doc_ids = list(documents.keys())
    texts_to_embed = list(documents.values())
    
    # Delegate the entire batch to the strategy's batch method
    embeddings = strategy.embed_batch(
        texts_to_embed,  
        model, 
        tokenizer, 
        embedding_type=embedding_type, 
        batch_size=batch_size
    )
    
    # Combine the IDs with their corresponding embeddings
    return {doc_id: embedding for doc_id, embedding in zip(doc_ids, embeddings)}



def run_search_and_evaluate(
    queries: List[Dict[str, str]],
    doc_embeddings: Dict[str, np.ndarray],
    model: SentenceTransformer,
    k: int = 10,
) -> float:
    """
    Runs queries against the corpus and calculates Mean Reciprocal Rank (MRR@k).
    """
    reciprocal_ranks = []

    # Prepare corpus
    corpus_ids = list(doc_embeddings.keys())
    corpus_embeddings = np.array([doc_embeddings[cid] for cid in corpus_ids])

    for query_info in tqdm(queries, desc="Running Queries"):
        query_text = query_info["query"]
        ground_truth_id = query_info["source_doc_id"]

        # Encode the query
        query_embedding = model.encode_query(query_text, normalize_embeddings=True)

        # Semantic search
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]
        retrieved_doc_ids = [corpus_ids[hit["corpus_id"]] for hit in hits]

        # Check where the ground truth appears in the ranking
        if ground_truth_id in retrieved_doc_ids:
            rank = retrieved_doc_ids.index(ground_truth_id) + 1  # ranks start at 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)  # not found in top-k

    # Return the mean across all queries
    return np.mean(reciprocal_ranks)
