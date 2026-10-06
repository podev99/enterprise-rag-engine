import os
from typing import Dict, List
import chromadb

from document_loader import chunk_document, load_text_document
from embedding_service import embed_chunks, generate_embedding

# Paths and collection configuration
DB_PATH = "./chroma_db"
COLLECTION_NAME = "enterprise_policies"


def get_chroma_collection():
    """Initializes a local persistent ChromaDB client and gets/creates a collection."""
    client = chromadb.PersistentClient(path=DB_PATH)
    # Cosine distance space for vector similarity search
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    return collection


def add_chunks_to_vector_store(chunks: List[str]) -> None:
    """Embeds text chunks via Gemini and persists them into ChromaDB."""
    collection = get_chroma_collection()

    print("Generating Gemini Embeddings for chunks...")
    embedded_records = embed_chunks(chunks)

    ids = [f"chunk_{record['chunk_id']}" for record in embedded_records]
    embeddings = [record["embedding"] for record in embedded_records]
    documents = [record["text"] for record in embedded_records]
    metadatas = [
        {"source": "company_policy.txt", "chunk_id": record["chunk_id"]}
        for record in embedded_records
    ]

    print("Persisting vector records into ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    print(
        f"✓ Successfully stored {len(ids)} records in collection '{COLLECTION_NAME}'."
    )


def query_semantic_search(query_text: str, top_k: int = 3) -> Dict:
    """Converts input query into vector embedding and queries ChromaDB for top_k matches."""
    collection = get_chroma_collection()

    print(f"Generating embedding for user query: '{query_text}'...")
    query_vector = generate_embedding(query_text)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


if __name__ == "__main__":
    sample_file = "data/company_policy.txt"

    # Step 1: Load and ingest chunks into Vector DB
    print(f"Loading document: {sample_file}...")
    raw_text = load_text_document(sample_file)
    chunks = chunk_document(raw_text)

    print("\n=== INGESTION PHASE ===")
    add_chunks_to_vector_store(chunks)

    # Step 2: Test Semantic Search Query
    test_query = "What is the warranty policy for enterprise software and SLA for critical issues?"
    print(f"\n=== SEMANTIC SEARCH PHASE ===")
    print(f"Query: \"{test_query}\"\n")

    search_results = query_semantic_search(test_query, top_k=2)

    documents = search_results["documents"][0]
    distances = search_results["distances"][0]
    metadatas = search_results["metadatas"][0]

    print("--- TOP RELEVANT CONTEXTS FOUND ---")
    for idx, (doc, dist, meta) in enumerate(
        zip(documents, distances, metadatas), 1
    ):
        print(
            f"\n[Rank {idx}] (Cosine Distance: {dist:.4f} | Chunk ID: {meta.get('chunk_id')}):"
        )
        print(doc)
        print("-" * 55)