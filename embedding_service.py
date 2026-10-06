import os
from typing import List
from dotenv import load_dotenv
from google import genai

from document_loader import chunk_document, load_text_document

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def generate_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a given text using Gemini Embeddings API.
    Includes a model fallback chain for max compatibility.
    """
    models_to_try = [
        "models/text-embedding-004",
        "text-embedding-004",
        "gemini-embedding-001",
        "gemini-embedding-2",
    ]

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.embed_content(
                model=model_name,
                contents=text,
            )

            # Support different response attribute structures in google-genai SDK
            if hasattr(response, "embedding") and response.embedding:
                return response.embedding.values
            elif hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"All embedding models failed. Last error: {last_error}")


def embed_chunks(chunks: List[str]) -> List[dict]:
    """
    Processes a list of text chunks and returns a structured list containing
    chunk text, index, and its corresponding vector embedding.
    """
    embedded_data = []

    for idx, chunk in enumerate(chunks, 1):
        embedding = generate_embedding(chunk)
        embedded_data.append(
            {
                "chunk_id": idx,
                "text": chunk,
                "embedding": embedding,
                "dimensions": len(embedding),
            }
        )
        print(f"✓ Embedded Chunk {idx}/{len(chunks)} ({len(embedding)} dims)")

    return embedded_data


if __name__ == "__main__":
    sample_file = "data/company_policy.txt"
    print(f"Loading & Chunking: {sample_file}...")

    raw_text = load_text_document(sample_file)
    chunks = chunk_document(raw_text)

    print(
        f"\nGenerating Embeddings for {len(chunks)} chunks via Gemini Embeddings API...\n"
    )
    vector_records = embed_chunks(chunks)

    print("\n--- Embedding Verification Sample ---")
    first_record = vector_records[0]
    print(f"Chunk ID: {first_record['chunk_id']}")
    print(f"Text Preview: {first_record['text'][:60]}...")
    print(f"Vector Dimensions: {first_record['dimensions']}")
    print(f"First 5 Vector Values: {first_record['embedding'][:5]}")