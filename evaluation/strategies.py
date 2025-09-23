# strategies_batch.py

from abc import ABC, abstractmethod
from typing import List, Literal, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer
# from tqdm import trange # Use trange for nested loops if needed

# --- 1. Modified Abstract Strategy Interface ---

class LongTextStrategy(ABC):
    """
    An abstract base class defining the interface for long text embedding strategies.
    """

    @abstractmethod
    def embed(
        self,
        text: str,
        model: SentenceTransformer,
        tokenizer,
        embedding_type: Literal["query", "document"] = "document",
    ) -> np.ndarray:
        """Embeds a single text."""
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: List[str],
        model: SentenceTransformer,
        tokenizer,
        embedding_type: Literal["query", "document"] = "document",
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Embeds a batch of texts for high-throughput scenarios."""
        pass


# --- 2. Implement Concrete Strategies with Batching ---

class TruncationStrategy(LongTextStrategy):
    """
    Embeds text by simply truncating it to the model's maximum sequence length.
    This strategy is very fast and efficient in batch mode.
    """
    def embed(self, text: str, model: SentenceTransformer, tokenizer, embedding_type: Literal["query", "document"] = "document") -> np.ndarray:
        # This remains for single-text embedding
        if embedding_type == "query" and hasattr(model, "encode_query"):
            return model.encode_query(text)
        elif embedding_type == "document" and hasattr(model, "encode_document"):
            return model.encode_document([text])[0]
        else:
            return model.encode(text)

    def embed_batch(self, texts: List[str], model: SentenceTransformer, tokenizer, embedding_type: Literal["query", "document"] = "document", batch_size: int = 32) -> List[np.ndarray]:
        print(f"--> Using Truncation Strategy (Batch Mode) for {len(texts)} documents")
        
        # Directly use the model's batch encoding capabilities
        if embedding_type == "query" and hasattr(model, "encode_query"):
            embeddings = model.encode_query(texts, batch_size=batch_size, show_progress_bar=True)
        elif embedding_type == "document" and hasattr(model, "encode_document"):
            embeddings = model.encode_document(texts, batch_size=batch_size, show_progress_bar=True)
        else:
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
            
        return [emb for emb in embeddings]


class ChunkAndAverageStrategy(LongTextStrategy):
    """
    Embeds text by splitting it into overlapping chunks, embedding all chunks from all documents
    in a single batch, and then averaging the results for each original document.
    """
    def __init__(self, chunk_size: int = 512, overlap_size: int = 64):
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def embed(self, text: str, model: SentenceTransformer, tokenizer, embedding_type: Literal["query", "document"] = "document") -> np.ndarray:
        # This remains for single-text embedding, unchanged from your original
        tokens = tokenizer.tokenize(text)
        text_chunks = []
        step = self.chunk_size - self.overlap_size
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            text_chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))

        if not text_chunks:
            return np.zeros(model.get_sentence_embedding_dimension())

        if embedding_type == "query" and hasattr(model, "encode_query"):
            chunk_embeddings = np.array([model.encode_query(chunk) for chunk in text_chunks])
        elif embedding_type == "document" and hasattr(model, "encode_document"):
            chunk_embeddings = model.encode_document(text_chunks)
        else:
            chunk_embeddings = model.encode(text_chunks)
            
        return np.mean(chunk_embeddings, axis=0)

    def embed_batch(self, texts: List[str], model: SentenceTransformer, tokenizer, embedding_type: Literal["query", "document"] = "document", batch_size: int = 32) -> List[np.ndarray]:
        print(f"--> Using Chunk and Average Strategy (Batch Mode) for {len(texts)} documents")
        
        all_chunks = []
        doc_chunk_counts = []

        # Step 1: Tokenize and chunk all documents, keeping track of chunk counts
        for text in texts:
            tokens = tokenizer.tokenize(text)
            if not tokens:
                doc_chunk_counts.append(0)
                continue
                
            text_chunks = []
            step = self.chunk_size - self.overlap_size
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i : i + self.chunk_size]
                text_chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
            
            all_chunks.extend(text_chunks)
            doc_chunk_counts.append(len(text_chunks))
        
        if not all_chunks:
            return [np.zeros(model.get_sentence_embedding_dimension()) for _ in texts]

        # Step 2: Embed all chunks from all documents in a single, efficient batch call
        if embedding_type == "query" and hasattr(model, "encode_query"):
            all_chunk_embeddings = model.encode_query(all_chunks, batch_size=batch_size, show_progress_bar=True)
        elif embedding_type == "document" and hasattr(model, "encode_document"):
            all_chunk_embeddings = model.encode_document(all_chunks, batch_size=batch_size, show_progress_bar=True)
        else:
            all_chunk_embeddings = model.encode(all_chunks, batch_size=batch_size, show_progress_bar=True)
            
        # Step 3: Re-assemble and average the embeddings for each original document
        final_embeddings = []
        start_index = 0
        for count in doc_chunk_counts:
            if count == 0:
                # Handle empty or very short documents
                final_embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
                continue
                
            end_index = start_index + count
            # Select embeddings for the current document's chunks
            doc_embeddings = all_chunk_embeddings[start_index:end_index]
            # Average them
            avg_embedding = np.mean(doc_embeddings, axis=0)
            final_embeddings.append(avg_embedding)
            start_index = end_index
            
        return final_embeddings
