"""
we need a way to see the questions
one by one

so we have to have this TUI
i will use it to feel the `test_questions.csv` file

"""
from typing import List
import json
from models import QARecord


class Style:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'

# Load the JSON data
with open('sample_es_data.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# Parse into Pydantic models
records: List[QARecord] = [QARecord(**item) for item in raw_data]

if not records:
    print(f"{Style.RED}No records found in the JSON file.{Style.RESET}")
    exit()

# Function to display a record professionally
def display_record(rec: QARecord, index: int, total: int):
    print("\n" + "=" * 80)
    print(f"{Style.BOLD}{Style.CYAN}Record {index+1}/{total} - ID: {rec.elastic_id}{Style.RESET}")
    print("=" * 80)
    
    # Question section
    print(f"{Style.UNDERLINE}{Style.GREEN}Question:{Style.RESET}")
    print(f"  {Style.YELLOW}Farsi:{Style.RESET} {rec.question.text.get('fa', 'N/A')}")
    print(f"  {Style.YELLOW}English:{Style.RESET} {rec.question.text.get('en', 'N/A')}")
    
    # Answers section
    print(f"\n{Style.UNDERLINE}{Style.GREEN}Answers:{Style.RESET}")
    if not rec.answers:
        print(f"  {Style.RED}No answers available.{Style.RESET}")
    else:
        for group_idx, ans_group in enumerate(rec.answers, 1):
            print(f"  {Style.MAGENTA}Answer Group {group_idx}:{Style.RESET}")
            for ans_idx, ans in enumerate(ans_group, 1):
                print(f"    {Style.BOLD}Answer {ans_idx}:{Style.RESET}")
                print(f"      {Style.YELLOW}Farsi:{Style.RESET} {ans.text.get('fa', 'N/A')}")
                print(f"      {Style.YELLOW}English:{Style.RESET} {ans.text.get('en', 'N/A')}")
                print("      ---")
    
    # Optional Metadata section (comment out if not needed)
    # print(f"\n{Style.UNDERLINE}{Style.GREEN}Metadata:{Style.RESET}")
    # for key, value in rec.metadata.model_dump(exclude_none=True).items():
    #     print(f"  {Style.YELLOW}{key.capitalize()}:{Style.RESET} {value}")
    
    print("=" * 80)

# Simple navigation loop
i = 0
while True:
    display_record(records[i], i, len(records))
    
    # Improved prompt with colors
    print(f"{Style.BOLD}Navigation:{Style.RESET} 'n' for next, 'p' for previous, 'q' to quit.")
    cmd = input("> ").strip().lower()
    
    if cmd == 'n':
        i = (i + 1) % len(records)
    elif cmd == 'p':
        i = (i - 1) % len(records)
    elif cmd == 'q':
        break
    else:
        print(f"{Style.RED}Invalid command. Try again.{Style.RESET}")