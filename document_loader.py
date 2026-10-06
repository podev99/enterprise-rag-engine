import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_document(file_path: str) -> str:
    """Reads content from a text document."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_document(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into smaller chunks with overlap using RecursiveCharacterTextSplitter.
    
    - chunk_size: Target size of each text segment (characters).
    - chunk_overlap: Overlapping character count between consecutive chunks to retain context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "- ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    sample_file = "data/company_policy.txt"
    print(f"Loading document: {sample_file}...")
    
    raw_text = load_text_document(sample_file)
    chunks = chunk_document(raw_text)

    print(f"\n--- Total Chunks Generated: {len(chunks)} ---")
    for idx, chunk in enumerate(chunks, 1):
        print(f"\n[Chunk {idx}] ({len(chunk)} chars):")
        print(chunk)
        print("-" * 40)