import os
from typing import List, Tuple
from dotenv import load_dotenv
from google import genai
from google.genai import types

from vector_store import query_semantic_search

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_context_and_sources(user_query: str, top_k: int = 2) -> Tuple[str, List[dict]]:
    """Retrieves relevant context chunks and builds metadata sources from ChromaDB."""
    search_results = query_semantic_search(user_query, top_k=top_k)
    retrieved_docs = search_results["documents"][0]
    retrieved_metas = search_results["metadatas"][0]
    retrieved_dists = search_results["distances"][0]

    context_blocks = []
    sources = []

    for idx, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metas, retrieved_dists), 1):
        chunk_id = meta.get("chunk_id", idx)
        source_file = meta.get("source", "company_policy.txt")
        context_blocks.append(f"[Context {idx}]:\n{doc}")
        sources.append({
            "chunk_id": chunk_id,
            "source": source_file,
            "distance": dist
        })

    context_str = "\n\n".join(context_blocks)
    return context_str, sources


def generate_rag_response(user_query: str, top_k: int = 2) -> Tuple[str, List[dict]]:
    """
    Generates a context-grounded response using Gemini API:
    - Applies strict System Instructions for guardrails.
    - Sets temperature=0.0 for maximum factual precision.
    - Disables Automatic Function Calling (AFC) to prevent SDK warnings.
    - Returns response text alongside sources metadata.
    """
    # Step 1: Retrieve Context and Sources
    context_str, sources = get_context_and_sources(user_query, top_k=top_k)

    # Step 2: System Prompt & Strict Anti-Hallucination Guardrails
    system_instruction = (
        "You are an AI Customer Support Agent for TechCorp.\n"
        "STRICT RULES:\n"
        "1. Answer the user's question strictly based ONLY on the provided context.\n"
        "2. If the answer cannot be explicitly found in the provided context, state EXACTLY: "
        "'I do not have enough information in the company policy to answer this question.'\n"
        "3. Do NOT use any prior knowledge, make assumptions, or hallucinate facts."
    )

    augmented_prompt = (
        f"--- PROVIDED CONTEXT FROM KNOWLEDGE BASE ---\n"
        f"{context_str}\n\n"
        f"--- USER QUESTION ---\n"
        f"{user_query}"
    )

    # Step 3: Configure temperature=0.0 and disable AFC
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3-flash",
        "gemini-flash-latest",
    ]

    response_text = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=augmented_prompt,
                config=config,
            )
            response_text = response.text.strip()
            break
        except Exception as e:
            print(f"Warning: Model {model_name} failed: {e}")
            continue

    if not response_text:
        raise RuntimeError("All Gemini models failed to process the RAG request.")

    return response_text, sources


def format_rag_output(user_query: str, answer: str, sources: List[dict]) -> str:
    """Formats the answer with explicit Source Citations for terminal display."""
    output = []
    output.append(f"❓ QUESTION: {user_query}")
    output.append("-" * 60)
    output.append(f"🤖 ANSWER:\n{answer}")
    output.append("-" * 60)

    if "do not have enough information" in answer.lower():
        output.append("📌 SOURCES: None (Out-of-Scope / Refused)")
    else:
        output.append("📌 SOURCES USED:")
        for src in sources:
            output.append(
                f"   • File: {src['source']} | Chunk ID: {src['chunk_id']} (Distance: {src['distance']:.4f})"
            )

    return "\n".join(output)


if __name__ == "__main__":
    print("=== ENTERPRISE RAG ENGINE DEMO (WITH GUARDRAILS & SOURCES) ===\n")

    # Scenario A: In-Scope Query
    query_a = "What is the warranty period for TechCorp enterprise software and what is the SLA for Severity 1 issues?"
    print(">>> RUNNING SCENARIO A (In-Scope Query)...")
    answer_a, sources_a = generate_rag_response(query_a, top_k=2)
    print(format_rag_output(query_a, answer_a, sources_a))

    print("\n" + "=" * 70 + "\n")

    # Scenario B: Out-of-Scope Query (Anti-Hallucination Guardrail Test)
    query_b = "Does TechCorp offer special discount packages for university students?"
    print(">>> RUNNING SCENARIO B (Out-of-Scope Query)...")
    answer_b, sources_b = generate_rag_response(query_b, top_k=2)
    print(format_rag_output(query_b, answer_b, sources_b))