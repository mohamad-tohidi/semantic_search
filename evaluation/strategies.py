from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# --- 1. Define the Abstract Strategy Interface ---


class LongTextStrategy(ABC):
    """
    An abstract base class defining the interface for long text embedding strategies.
    """

    @abstractmethod
    def embed(self, text: str, model: SentenceTransformer, tokenizer) -> np.ndarray:
        """
        The contract for all concrete strategy classes.

        Args:
            text (str): The long text to embed.
            model (SentenceTransformer): The pre-loaded Sentence Transformer model.
            tokenizer: The tokenizer associated with the model.

        Returns:
            np.ndarray: The resulting embedding vector.
        """
        pass


# --- 2. Implement Concrete Strategies ---


class TruncationStrategy(LongTextStrategy):
    """
    Embeds text by simply truncating it to the model's maximum sequence length.
    This is the default behavior of many transformer models.
    """

    def embed(self, text: str, model: SentenceTransformer, tokenizer) -> np.ndarray:
        """
        Generates an embedding by truncating the input text.

        Args:
            text (str): The long text to embed.
            model (SentenceTransformer): The model to use for encoding.
            tokenizer: The tokenizer (not directly used, as model.encode handles it).

        Returns:
            np.ndarray: A single embedding vector for the truncated text.
        """
        print("--> Using Truncation Strategy")
        # SentenceTransformer's .encode() method handles truncation automatically.
        return model.encode(text)


class ChunkAndAverageStrategy(LongTextStrategy):
    """
    Embeds text by splitting it into overlapping chunks, embedding each chunk,
    and then averaging the resulting embeddings.
    """

    def __init__(self, chunk_size: int = 512, overlap_size: int = 64):
        """
        Initializes the chunking strategy with specific parameters.

        Args:
            chunk_size (int): The maximum number of tokens in each chunk. Should be
                              less than or equal to the model's max sequence length.
            overlap_size (int): The number of tokens to overlap between consecutive chunks
                                to preserve context.
        """
        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def embed(self, text: str, model: SentenceTransformer, tokenizer) -> np.ndarray:
        """
        Generates an embedding by chunking the text and averaging the embeddings.

        Args:
            text (str): The long text to embed.
            model (SentenceTransformer): The model to use for encoding.
            tokenizer: The tokenizer used to split text into tokens.

        Returns:
            np.ndarray: A single, averaged embedding vector for the entire text.
        """
        print(
            f"--> Using Chunk and Average Strategy (Chunk: {self.chunk_size}, Overlap: {self.overlap_size})"
        )

        # Tokenize the entire text without truncation
        tokens = tokenizer.tokenize(text)

        # Create text chunks from tokens
        text_chunks = []
        step = self.chunk_size - self.overlap_size
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            # Convert tokens back to a string for the model
            text_chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))

        if not text_chunks:
            # Handle case where text is shorter than a chunk
            return model.encode(text)

        # Get embeddings for all chunks. The model can process a list of sentences.
        chunk_embeddings = model.encode(text_chunks)

        # Average the embeddings to get a single vector representation
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        return avg_embedding


# --- 3. Define the Context Class that Uses a Strategy ---


class LongTextEmbedder:
    """
    This class uses a specified strategy to embed long texts.
    """

    def __init__(self, strategy: LongTextStrategy):
        """
        Initializes the embedder with a default strategy.

        Args:
            strategy (LongTextStrategy): An instance of a concrete strategy.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: LongTextStrategy):
        """Allows changing the strategy at runtime."""
        print(f"\nSwitching strategy to: {strategy.__class__.__name__}")
        self._strategy = strategy

    def embed(self, text: str, model: SentenceTransformer, tokenizer) -> np.ndarray:
        """
        Delegates the embedding task to the current strategy object.

        Args:
            text (str): The long text to embed.
            model (SentenceTransformer): The model to use.
            tokenizer: The tokenizer to use.

        Returns:
            np.ndarray: The final embedding vector.
        """
        return self._strategy.embed(text, model, tokenizer)


# --- 4. Example Usage ---

if __name__ == "__main__":
    # Load a model and its tokenizer
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model max sequence length: {model.max_seq_length} tokens")

    # Create a sample long text (much longer than the model's max length)
    paragraph = "The strategy pattern is a behavioral design pattern that enables an algorithm's behavior to be selected at runtime. The strategy pattern defines a family of algorithms, encapsulates each algorithm, and makes the algorithms interchangeable within that family."
    long_text = " ".join([paragraph] * 10)
    num_tokens = len(tokenizer.tokenize(long_text))
    print(f"Sample text length: {num_tokens} tokens\n")

    # --- Strategy 1: Truncation (the baseline) ---
    truncation_strategy = TruncationStrategy()
    embedder = LongTextEmbedder(truncation_strategy)
    embedding1 = embedder.embed(long_text, model, tokenizer)
    print(f"Embedding shape from Truncation: {embedding1.shape}")

    # --- Strategy 2: Chunking with large overlap ---
    # Good for preserving context, but more computationally expensive.
    chunking_strategy_1 = ChunkAndAverageStrategy(chunk_size=512, overlap_size=128)
    embedder.set_strategy(chunking_strategy_1)
    embedding2 = embedder.embed(long_text, model, tokenizer)
    print(f"Embedding shape from Chunking (512/128): {embedding2.shape}")

    # --- Strategy 3: Chunking with smaller overlap and smaller chunks ---
    # Faster than the above, but might lose some context at the seams.
    chunking_strategy_2 = ChunkAndAverageStrategy(chunk_size=256, overlap_size=32)
    embedder.set_strategy(chunking_strategy_2)
    embedding3 = embedder.embed(long_text, model, tokenizer)
    print(f"Embedding shape from Chunking (256/32): {embedding3.shape}")

    # You can now compare the embeddings
    print("\n--- Comparing Embeddings ---")
    cos_sim_1_2 = util.cos_sim(embedding1, embedding2).item()
    cos_sim_2_3 = util.cos_sim(embedding2, embedding3).item()
    print(f"Similarity between Truncation and Chunking(512/128): {cos_sim_1_2:.4f}")
    print(f"Similarity between two Chunking strategies: {cos_sim_2_3:.4f}")
