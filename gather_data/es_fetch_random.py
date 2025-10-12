import os
import json
from pathlib import Path
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from models import QARecord


load_dotenv()


ES_USER = os.getenv("ES_USER")
ES_PASSWORD = os.getenv("ES_PASSWORD")
ES_URL = os.getenv("ES_URL")

INDEX_NAME = "parsaqa_questions_003"
NUM_SHORTEST = 5
NUM_LONGEST = 95
SAMPLE_SIZE = 200000
    



def get_elasticsearch_client():
    """Create and return Elasticsearch client."""
    return Elasticsearch(
        hosts=[{"host": ES_URL, "port": 9200, "scheme": "https"}],
        basic_auth=(ES_USER, ES_PASSWORD),
        verify_certs=False,
        ssl_show_warn=False,
    )


def extract_length(doc_source):
    """Extract text length for sorting."""
    try:
        # this here returns the length of the question
        # return len(doc_source["question"]["text"]["fa"])  

        return len(doc_source["answers"][0]["text"]["fa"])
    except Exception:
        return 0


def validate_and_convert_records(hits):
    """Validate all hits using Pydantic models."""
    validated_records = []
    errors = []
    
    for hit in hits:
        try:
            # Add elastic_id to the source data
            source_data = hit["_source"].copy()
            source_data["elastic_id"] = hit["_id"]
            
            # Validate with Pydantic model
            qa_record = QARecord.model_validate(source_data)
            validated_records.append(qa_record)
            
        except Exception as e:
            errors.append({
                "elastic_id": hit.get("_id", "unknown"),
                "error": str(e)
            })
    
    return validated_records, errors


def save_records_to_file(records, filepath):
    """Save validated records to JSON file."""
    # Convert Pydantic models to dict for JSON serialization
    records_data = [record.model_dump() for record in records]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(records_data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(records)} records to {filepath}")


def load_records_from_file(filepath):
    """Load records from JSON file and convert back to Pydantic models."""
    with open(filepath, 'r', encoding='utf-8') as f:
        records_data = json.load(f)
    
    # Convert back to Pydantic models
    records = [QARecord.model_validate(data) for data in records_data]
    print(f"Loaded {len(records)} records from {filepath}")
    
    return records


def fetch_data_with_scroll(es, index_name, sample_size, source_filter):
    """Fetch data using scroll API to handle large result sets."""
    print("Fetching data from Elasticsearch using scroll API...")
    
    # Initial search with scroll
    response = es.search(
        index=index_name,
        size=1000,  # Process in batches of 1000
        query={"match_all": {}},
        _source=source_filter,
        scroll='5m',  # Keep scroll context alive for 5 minutes
    )
    
    scroll_id = response['_scroll_id']
    hits = response['hits']['hits']
    all_hits = hits.copy()
    
    # Continue scrolling until we have enough data or no more results
    while len(all_hits) < sample_size and hits:
        response = es.scroll(
            scroll_id=scroll_id,
            scroll='5m'
        )
        hits = response['hits']['hits']
        all_hits.extend(hits)
        
        if len(all_hits) % 5000 == 0:
            print(f"Fetched {len(all_hits)} documents so far...")
    
    # Clear scroll context
    es.clear_scroll(scroll_id=scroll_id)
    
    # Limit to requested sample size
    all_hits = all_hits[:sample_size]
    print(f"Fetched {len(all_hits)} documents total")
    
    return all_hits


def fetch_and_process_data():
    """Main function to fetch, validate, and save data."""
    es = get_elasticsearch_client()
    

    source_filter = {"excludes": ["*_vector"]}
    
    # Fetch data using scroll API
    hits = fetch_data_with_scroll(es, INDEX_NAME, SAMPLE_SIZE, source_filter)
    
    # Sort by text length
    scored = [(extract_length(h.get("_source", {})), h) for h in hits]
    scored.sort(key=lambda x: x[0])
    
    # Get shortest and longest
    shortest_hits = [h for _, h in scored[:NUM_SHORTEST]]
    longest_hits = [h for _, h in scored[-NUM_LONGEST:]] if NUM_LONGEST > 0 else []
    
    # Remove duplicates
    seen = set()
    combined_hits = []
    for h in shortest_hits + longest_hits:
        _id = h.get("_id")
        if _id not in seen:
            seen.add(_id)
            combined_hits.append(h)
    
    print(f"Selected {len(combined_hits)} unique records")
    
    # Validate all records
    print("Validating records...")
    validated_records, errors = validate_and_convert_records(combined_hits)
    
    if errors:
        print(f"Validation errors in {len(errors)} records:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - ID {error['elastic_id']}: {error['error']}")
    
    # Save to file
    output_file = Path("sample_es_data.json")
    save_records_to_file(validated_records, output_file)
    
    return validated_records, errors


if __name__ == "__main__":
    records, errors = fetch_and_process_data()
    print(f"\nProcessed {len(records)} valid records")
    if errors:
        print(f"Found {len(errors)} validation errors")

