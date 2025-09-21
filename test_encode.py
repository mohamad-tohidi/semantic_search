from sentence_transformers import SentenceTransformer, util

# Load the multilingual-e5-small model
model = SentenceTransformer("intfloat/multilingual-e5-small")

# Example short and long texts
short_text = "The quick brown fox jumps over the lazy dog."

# A long text, repeated many times to exceed context window (512 tokens for this model)
long_text = "The quick brown fox jumps over the lazy dog. " * 200

# Encode texts
short_emb = model.encode(short_text, convert_to_tensor=True)
long_emb = model.encode(long_text, convert_to_tensor=True)

# Compare embeddings using cosine similarity
similarity = util.cos_sim(short_emb, long_emb)

print("Short embedding shape:", short_emb.shape)
print("Long embedding shape:", long_emb.shape)
print("Cosine similarity between short and long text:", similarity.item())
