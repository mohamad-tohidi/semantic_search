import json
import os
from typing import Dict, List


def load_documents(data_dir: str) -> Dict[str, str]:
    """
    Loads all .txt files from the data_dir.
    Key: doc_id (filename without .txt), Value: file content.
    """
    documents = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            doc_id = filename[:-4]  # Strip .txt
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                documents[doc_id] = f.read().strip()
    return documents


def load_queries(queries_path: str) -> List[Dict[str, str]]:
    """
    Loads queries from jsonl file.
    Each line: {"query": "...", "source_doc_id": "..."}
    """
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    return queries
