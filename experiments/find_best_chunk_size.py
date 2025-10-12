"""
Chunk Size and Overlap Experiment

This script tests different chunk sizes and overlaps to find the best configuration
for semantic search using F1 score evaluation.

Simple approach:
1. Create multiple Qdrant collections with different chunk configurations
2. Load test questions and their expected answers
3. Run searches and calculate F1 scores
4. Find the best configuration
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import Payload, PointStruct
from qdrant_client.http.models import QueryResponse
from transformers import AutoTokenizer

from models import QARecord


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    chunk_size: int
    overlap: int
    collection_name: str


@dataclass
class TestQuestion:
    """Test question with expected answer."""
    question: str
    elastic_id: str


class ChunkExperiment:
    """Simple experiment to find best chunk size and overlap."""
    
    def __init__(self, model_name: str = "intfloat/e5-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.client = QdrantClient(path="./qdrant_db")
        
        # Test configurations
        self.configs = [
            # ChunkConfig(128, 25, "chunk_128_25"),
            ChunkConfig(256, 50, "chunk_256_50"),
            ChunkConfig(512, 100, "chunk_512_100"),
        ]
    
    def load_sample_data(self, filepath: str) -> List[QARecord]:
        """Load sample data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = [QARecord.model_validate(item) for item in data]
        print(f"Loaded {len(records)} records")
        return records
    
    def extract_texts(self, records: List[QARecord]) -> List[Dict[str, Any]]:
        """Extract texts from QA records."""
        texts = []
        
        for record in records:
            # Extract question text
            question_text = record.question.text.get('fa', '')
            if question_text:
                texts.append({
                    'text': question_text,
                    'elastic_id': record.elastic_id,
                    'type': 'question'
                })
            
            # Extract answer texts
            for answer_group in record.answers:
                for answer in answer_group:
                    answer_text = answer.text.get('fa', '')
                    if answer_text:
                        texts.append({
                            'text': answer_text,
                            'elastic_id': record.elastic_id,
                            'type': 'answer'
                        })
        
        print(f"Extracted {len(texts)} texts")
        return texts
    
    def chunk_texts(self, texts: List[Dict[str, Any]], config: ChunkConfig) -> List[Dict[str, Any]]:
        """Chunk texts with given configuration."""
        chunked_texts = []
        
        for text_item in texts:
            text = text_item['text']
            elastic_id = text_item['elastic_id']
            text_type = text_item['type']
            
            # Tokenize the text
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Create chunks with overlap
            step_size = config.chunk_size - config.overlap
            for i in range(0, len(tokens), step_size):
                chunk_tokens = tokens[i:i + config.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
                
                if chunk_text:
                    chunked_texts.append({
                        'text': chunk_text,
                        'elastic_id': elastic_id,
                        'type': text_type,
                        'chunk_id': f"{elastic_id}_{i}"
                    })
        
        print(f"Created {len(chunked_texts)} chunks for config {config.chunk_size}_{config.overlap}")
        return chunked_texts
    
    def create_collection(self, config: ChunkConfig, chunked_texts: List[Dict[str, Any]]):
        """Create Qdrant collection with chunked texts."""
        print(f"Creating collection: {config.collection_name}")
        
        

        # Collection metadata
        collection_metadata :Payload = {
            "description": f"Chunk experiment: {config.chunk_size} tokens, {config.overlap} overlap",
            "embedding_model": self.model_name,
            "chunk_size": f"{config.chunk_size} tokens",
            "chunk_overlap": f"{config.overlap} tokens",
            "vector_size": self.model.get_sentence_embedding_dimension(),
        }
        
        try:
            self.client.create_collection(
                collection_name=config.collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                ),
                # NOTE: at the time that i am writing this code
                # this feature is not yet supported
                # but i know that it will be supported soon
                # so we comment it out for now
                # metadata=collection_metadata
            )
            print(f"Created collection: {config.collection_name}")
        except Exception as e:
            print(f"Collection might already exist: {e}")
        
        # Index texts in batches
        batch_size = 10
        total_indexed = 0
        
        for i in range(0, len(chunked_texts), batch_size):
            batch = chunked_texts[i:i + batch_size]
            
            # Extract texts for embedding
            batch_texts = [item['text'] for item in batch]
            
            # Create embeddings
            embeddings = self.model.encode(batch_texts, convert_to_tensor=False)
            
            # Create points
            points = []
            for j, (text_item, embedding) in enumerate(zip(batch, embeddings)):
                point = PointStruct(
                    id=total_indexed + j,
                    vector=embedding.tolist(),
                    payload={
                        'text': text_item['text'],
                        'elastic_id': text_item['elastic_id'],
                        'type': text_item['type'],
                        'chunk_id': text_item['chunk_id']
                    }
                )
                points.append(point)
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=config.collection_name,
                points=points
            )
            
            total_indexed += len(points)
            print(f"Indexed {total_indexed}/{len(chunked_texts)} chunks")
        
        print(f"Successfully indexed {total_indexed} chunks to {config.collection_name}")
    
    def load_test_questions(self, csv_filepath: str) -> List[TestQuestion]:
        """Load test questions from CSV file."""
        questions = []
        
        if not Path(csv_filepath).exists():
            print(f"Test questions file {csv_filepath} not found.")
            print("Please create a CSV file with columns: question, expected_answer, elastic_id")
            return questions
        
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(TestQuestion(
                    question=row['question'],
                    elastic_id=row['elastic_id']
                ))
        
        print(f"Loaded {len(questions)} test questions")
        return questions
    
    def search_collection(self, query: str, collection_name: str, limit: int = 5) -> List[Dict]:
        """Search a collection and return results."""
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        
        results: QueryResponse = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding.tolist(),
            limit=limit
        )
        
        return [
            {
                'score': result.score,
                'text': result.payload['text'],
                'elastic_id': result.payload['elastic_id'],
                'type': result.payload['type']
            }
            for result in results.points
        ]
    
    def calculate_f1_score(self, test_question: TestQuestion, search_results) -> float:
        """Calculate F1 score for a test question."""
        expected_id = test_question.elastic_id


        found_ids = [result["elastic_id"] for result in search_results]
        
        # Check if expected answer is in top results
        if expected_id in found_ids:
            # Find position of expected answer
            position = found_ids.index(expected_id) + 1
            # Higher score for better position (simple approach)
            return 1.0 / position
        else:
            return 0.0
    
    def run_experiment(self, sample_file: str, test_questions_file: str = None):
        """Run the complete experiment."""
        print("Starting Chunk Size Experiment")
        print("=" * 40)
        
        # Load sample data
        records = self.load_sample_data(sample_file)
        texts = self.extract_texts(records)
        
        # Create collections for each configuration
        for config in self.configs:
            print(f"\nProcessing config: {config.chunk_size} tokens, {config.overlap} overlap")
            chunked_texts = self.chunk_texts(texts, config)
            self.create_collection(config, chunked_texts)
        
        # Load test questions if available
        test_questions = []
        if test_questions_file:
            test_questions = self.load_test_questions(test_questions_file)
        
        if not test_questions:
            print("\nNo test questions available. Collections created successfully.")
            print("Please create a CSV file with test questions to run evaluation.")
            return
        
        # Run evaluation
        print(f"\nRunning evaluation with {len(test_questions)} test questions")
        results = {}
        
        for config in self.configs:
            print(f"\nEvaluating {config.collection_name}...")
            f1_scores = []
            
            for test_q in test_questions:
                search_results = self.search_collection(
                    test_q.question, 
                    config.collection_name, 
                    limit=5
                )
                f1_score = self.calculate_f1_score(test_q, search_results)
                f1_scores.append(f1_score)
            
            avg_f1 = sum(f1_scores) / len(f1_scores)
            results[config.collection_name] = {
                'config': config,
                'avg_f1': avg_f1,
                'scores': f1_scores
            }
            
            print(f"Average F1 Score: {avg_f1:.3f}")
        
        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1]['avg_f1'])
        print("\nBest Configuration:")
        print(f"Collection: {best_config[0]}")
        print(f"Chunk Size: {best_config[1]['config'].chunk_size}")
        print(f"Overlap: {best_config[1]['config'].overlap}")
        print(f"Average F1 Score: {best_config[1]['avg_f1']:.3f}")
        
        return results


def main():
    """Main function to run the experiment."""
    experiment = ChunkExperiment(model_name="intfloat/e5-large")
    
    # Run experiment
    sample_file = "./sample_es_data.json"
    test_questions_file = "./experiments/test_questions.csv"  # You'll create this later
    
    experiment.run_experiment(sample_file, test_questions_file)


if __name__ == "__main__":
    main()