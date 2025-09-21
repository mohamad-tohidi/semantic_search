import os

from evaluation.data_loader import load_documents, load_queries
from evaluation.embedder import embed_documents, embed_query
from evaluation.search import cosine_similarity, retrieve_top_doc


def run_benchmark(model_name: str, data_dir: str, queries_path: str) -> float:
    """
    Runs the benchmark for a given model.
    Returns accuracy (fraction of queries where top doc matches ground truth).
    """
    # Load data
    documents = load_documents(data_dir)
    queries = load_queries(queries_path)

    # Embed all documents once
    doc_embs = embed_documents(model_name, documents)

    correct = 0
    for q in queries:
        query_text = q["query"]
        ground_truth_id = q["source_doc_id"]

        # Embed query
        query_emb = embed_query(model_name, query_text)

        # Search
        sims = cosine_similarity(query_emb, doc_embs)
        top_doc_id = retrieve_top_doc(sims)

        if top_doc_id == ground_truth_id:
            correct += 1

    accuracy = correct / len(queries) if queries else 0
    return accuracy


if __name__ == "__main__":
    data_dir = "test_dataset"
    queries_path = os.path.join(data_dir, "queries.jsonl")

    # Run for e5
    e5_model = "intfloat/multilingual-e5-small"
    e5_acc = run_benchmark(e5_model, data_dir, queries_path)
    print(f"E5 Accuracy: {e5_acc:.2f}")
