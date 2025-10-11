"""
this script
will create a collection in qdrant
from a given json file

it will embed the texts and do other necessary things

"""

import json
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Payload
from transformers import AutoTokenizer

from models import QARecord


model_name = "intfloat/e5-large"


def load_data(filepath: str) -> List[QARecord]:
    """Step 1: Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = [QARecord.model_validate(item) for item in data]
    print(f"Loaded {len(records)} records from {filepath}")
    return records


def extract_texts(records: List[QARecord]) -> List[Dict[str, Any]]:
    """Step 2: Extract texts from QA records."""
    texts = []
    
    for record in records:
        # Extract question text
        question_text = record.question.text.get('fa', '')
        if question_text:
            texts.append({
                'text': question_text,
                'elastic_id': record.elastic_id
            })
        
        # Extract answer texts
        for answer_group in record.answers:
            for answer in answer_group:
                answer_text = answer.text.get('fa', '')
                if answer_text:
                    texts.append({
                        'text': answer_text,
                        'elastic_id': record.elastic_id
                    })
    
    print(f"Extracted {len(texts)} texts")
    return texts


def chunk_texts(texts: List[Dict[str, Any]], max_tokens: int = 256, overlap: int = 50) -> List[Dict[str, Any]]:
    """Step 3: Chunk texts with overlap."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    chunked_texts = []
    
    for text_item in texts:
        text = text_item['text']
        elastic_id = text_item['elastic_id']
        
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Create chunks with overlap
        step_size = max_tokens - overlap
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
            
            if chunk_text:  
                chunked_texts.append({
                    'text': chunk_text,
                    'elastic_id': elastic_id
                })
    
    print(f"Created {len(chunked_texts)} chunks from {len(texts)} texts")
    return chunked_texts


def index_to_qdrant(chunked_texts: List[Dict[str, Any]]):
    """Step 4: Index chunked texts to Qdrant."""
    client = QdrantClient(path="./qdrant_db")
    collection_name = "parsa_003"
    


    print("Loading E5 model...")
    model = SentenceTransformer(model_name)
    

    # Collection metadata
    collection_metadata = Payload({
        "description": "Persian QA dataset from ParsaQA with chunked questions and answers",
        "embedding_model": model_name,
        "chunk_size": "256 tokens",
        "chunk_overlap": "50 tokens",
        "source": "Elasticsearch parsaqa_questions_003 index",
        "created_by": "Tohidi @ Hamta",
        "vector_size": model.get_sentence_embedding_dimension(),
        "content_type": "QA chunks (questions and answers)",
        "indexing_date": "2025"
    })
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=model.get_sentence_embedding_dimension(),  
                distance=Distance.COSINE
            ),
            metadata=collection_metadata
        )
        print(f"Created collection: {collection_name}")
        print("Collection metadata added successfully")
    except Exception as e:
        print(f"Collection might already exist: {e}")
    
    # Process in batches
    batch_size = 10
    total_indexed = 0
    
    for i in range(0, len(chunked_texts), batch_size):
        batch = chunked_texts[i:i + batch_size]
        
        # Extract texts for embedding
        batch_texts = [item['text'] for item in batch]
        
        # Create embeddings
        embeddings = model.encode(batch_texts, convert_to_tensor=False)
        
        # Create points
        points = []
        for j, (text_item, embedding) in enumerate(zip(batch, embeddings)):
            point = PointStruct(
                id=int(text_item['elastic_id']),
                vector=embedding.tolist(),
                payload={
                    'text': text_item['text'],
                    'elastic_id': text_item['elastic_id']
                }
            )
            points.append(point)
        
        # Store in Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        total_indexed += len(points)
        print(f"Indexed {total_indexed}/{len(chunked_texts)} chunks")
    
    print(f"Successfully indexed {total_indexed} chunks to Qdrant")
    
    # Test search
    print("\nTesting search...")
    test_query = "ازدواج موقت"
    query_embedding = model.encode([test_query], convert_to_tensor=False)[0]
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=5
    )
    
    print(f"Search results for '{test_query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.score:.3f}")
        print(f"   Elastic ID: {result.payload['elastic_id']}")
        print(f"   Text: {result.payload['text'][:100]}...")
        print()


def main():
    """Main function with simple 4-step flow."""
    # Step 1: Load data
    sample_file = "sample_es_data.json"
    if not Path(sample_file).exists():
        print(f"Sample file {sample_file} not found. Run es_fetch_random.py first.")
        return
    
    data = load_data(sample_file)

    print(f"Loaded {len(data)} records")

    # Step 2: Extract texts
    texts = extract_texts(data)

    print(f"Extracted {len(texts)} texts")
    
    # Step 3: Chunk texts with overlap
    chunked_texts = chunk_texts(texts)

    print(f"Chunked {len(chunked_texts)} texts")
    
    # Step 4: Index to Qdrant
    index_to_qdrant(chunked_texts)


if __name__ == "__main__":
    main()
