import hashlib
import json
import os

# ==== CONFIG ====
INPUT_JSON = "query-1.json"  # path to your input JSON file
OUTPUT_DIR = "./test_dataset"  # folder to save txt files

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load JSON data
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep track of seen hashes
seen_hashes = set()

for item in data:
    answer_id = item.get("answer.id")
    content = item.get("answer.content", "")

    # Compute hash of content for deduplication
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    if content_hash not in seen_hashes:
        seen_hashes.add(content_hash)

        # Save content to a text file named after the answer_id
        output_path = os.path.join(OUTPUT_DIR, f"{answer_id}.txt")
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(content)
