from evaluation.data_loader import load_queries

queries_path = "./test_dataset/queries.jsonl"

queries = load_queries(queries_path=queries_path)

print(queries)
