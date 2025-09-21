import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from evaluation.benchmark import (
    embed_corpus,
    load_documents,
    load_mock_data,
    load_queries,
    run_search_and_evaluate,
)
from evaluation.plotter import plot_results
from evaluation.strategies import ChunkAndAverageStrategy, TruncationStrategy


def main():
    """Main function to run the benchmark."""
    # --- Configuration ---
    MODEL_NAMES = [
        "intfloat/multilingual-e5-small",
        "google/embeddinggemma-300m",
    ]
    DATA_DIR = "./test_dataset"
    QUERIES_PATH = "./test_dataset/queries.jsonl"
    NDCG_K = 10

    # --- 1. Load Data ---
    if os.path.exists(DATA_DIR) and os.path.exists(QUERIES_PATH):
        documents = load_documents(DATA_DIR)
        queries = load_queries(QUERIES_PATH)
    else:
        documents, queries = load_mock_data()

    if not documents or not queries:
        print("Error: No documents or queries found. Exiting.")
        return

    # --- 2. Define Strategies to Benchmark ---
    strategies_to_test = [
        TruncationStrategy(),
        ChunkAndAverageStrategy(chunk_size=512, overlap_size=64),
        ChunkAndAverageStrategy(chunk_size=384, overlap_size=64),
        ChunkAndAverageStrategy(chunk_size=256, overlap_size=32),
    ]

    # --- 3. Run Benchmark for each model ---
    all_results = {}
    for model_name in MODEL_NAMES:
        print(f"\n--- Loading and benchmarking model: {model_name} ---")
        model = SentenceTransformer(model_name)
        model.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        results_for_model = {}
        for strategy in strategies_to_test:
            # Generate a descriptive name for each strategy
            if isinstance(strategy, TruncationStrategy):
                strategy_name = "Truncation"
            elif isinstance(strategy, ChunkAndAverageStrategy):
                strategy_name = f"Chunk (Size: {strategy.chunk_size}, Overlap: {strategy.overlap_size})"
            else:
                strategy_name = strategy.__class__.__name__

            print(f"\n--- Benchmarking Strategy: {strategy_name} ---")

            # Embed the corpus with the current strategy
            doc_embeddings = embed_corpus(documents, strategy, model, tokenizer)

            # Run search and get the average nDCG score
            avg_ndcg = run_search_and_evaluate(queries, doc_embeddings, model, k=NDCG_K)

            results_for_model[strategy_name] = avg_ndcg
            print(f"Average nDCG@{NDCG_K} for {strategy_name}: {avg_ndcg:.4f}")

        all_results[model_name] = results_for_model

    # --- 4. Plot Results ---
    plot_results(all_results)


if __name__ == "__main__":
    main()
