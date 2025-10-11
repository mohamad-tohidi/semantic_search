"""
Simple script to query the Qdrant vector database.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointsList, ScoredPoint


def search_qa_database(query: str, limit: int = 5):
    """Search the QA database for similar content."""
    
    # Setup Qdrant client
    client = QdrantClient(path="./qdrant_db")
    collection_name = "parsa_003"
    
    # Load E5 model
    print("Loading E5 model...")
    model = SentenceTransformer("intfloat/e5-small")
    
    # Create query embedding
    query_embedding = model.encode([query], convert_to_tensor=False)[0]
    
    # Search in Qdrant
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=limit
    )
    
    return results


def main():
    """Main function to run queries."""
    print("QA Vector Database Search")
    print("=" * 30)
    
    while True:
        # Get user query
        query = input("\nEnter your search query (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a search query.")
            continue
        
        try:
            # Search the database
            print(f"\nSearching for: '{query}'")
            results : PointsList = search_qa_database(query, limit=5)


            print(results)

            if not results:
                print("No results found.")
                continue
            
            # Display results
            print(f"\nFound {len(results.points)} results:")
            print("-" * 50)
            
            for i, result in enumerate(results.points, 1):
                result : ScoredPoint
                print(f"{i}. Score: {result.score:.3f}")
                print(f"   Elastic ID: {result.payload['elastic_id']}")
                print(f"   Text: {result.payload['text']}")
                print()
                
        except Exception as e:
            print(f"Error searching database: {e}")


if __name__ == "__main__":
    main()