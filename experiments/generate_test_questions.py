import json
from openai import OpenAI
from dotenv import load_dotenv 
from texttools import TheTool
from models import QARecord
from typing import List
from tqdm import tqdm
load_dotenv()



sample_file_path = "./sample_es_data.json"


def load_data(filepath: str) -> List[QARecord]:
    """Step 1: Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = [QARecord.model_validate(item) for item in data]
    print(f"Loaded {len(records)} records from {filepath}")
    return records

def chunk_text(text: str, max_words: int = 500) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


data_list = load_data(sample_file_path)


long_answers = []

for data in data_list:
    data: QARecord
    answer = data.answers[0][0].text["fa"]
    if len(answer.split()) > 500:
        long_answers.append((data, answer))

print(f"found {len(long_answers)} long answers")




# Setup tool
client = OpenAI()
tool = TheTool(client=client, model="gemma-3")
language = "Persian/Farsi"

# Generate questions
all_questions = []
for idx, (data, answer) in tqdm(enumerate(long_answers), desc="generating questions from chunks"):
    chunks = chunk_text(answer)
    for chunk in chunks:
        question = tool.generate_question_from_text(text=chunk, output_lang=language)
        all_questions.append({
            'id': data.elastic_id,
            'question': question
        })

print(f"Generated {len(all_questions)} questions.")
print("Sample questions:", all_questions[:3])

