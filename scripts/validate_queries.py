import argparse
import json
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


DEFAULT_TESTS: List[Dict[str, object]] = [
    {
        "label": "performance",
        "query": "What performance improvements have I delivered in backend systems?",
        "expected_keywords": ["performance", "improved", "optimization", "latency"],
    },
    {
        "label": "leadership",
        "query": "Where did I lead engineers and what was the outcome?",
        "expected_keywords": ["led", "lead", "engineers", "team"],
    },
    {
        "label": "tech stack",
        "query": "What is my technology stack and frameworks experience?",
        "expected_keywords": ["asp.net", "azure", "api", "microservices", "sql"],
    },
    {
        "label": "achievements",
        "query": "What measurable achievements and impact metrics do I have?",
        "expected_keywords": ["100k", "100", "impact", "integrity", "users"],
    },
    {
        "label": "preferences",
        "query": "What are my work preferences and collaboration style?",
        "expected_keywords": ["preference", "collaboration", "style", "remote", "onsite"],
    },
]


def relevance_hint(results: List[str], expected_keywords: List[str]) -> int:
    lowered = [doc.lower() for doc in results]
    hits = 0
    for doc in lowered:
        if any(k.lower() in doc for k in expected_keywords):
            hits += 1
    return hits


def main():
    parser = argparse.ArgumentParser(description="Validate top-3 relevance for standard queries.")
    parser.add_argument("--chroma-path", required=True)
    parser.add_argument("--collection", default="work_background_v1")
    parser.add_argument("--model", default="intfloat/e5-small-v2")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    client = chromadb.PersistentClient(path=args.chroma_path)
    collection = client.get_collection(name=args.collection)

    print(f"Collection: {args.collection}")
    print(f"Top-K: {args.top_k}")

    for test in DEFAULT_TESTS:
        prefixed_query = f"query: {test['query']}"
        query_embedding = model.encode([prefixed_query], normalize_embeddings=True).tolist()[0]

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        hits = relevance_hint(docs, test["expected_keywords"])
        status = "PASS" if hits >= 1 else "WEAK"

        print("-" * 72)
        print(f"[{status}] {test['label']}: {test['query']}")
        print(f"Top-3 relevance hint hits: {hits}/{args.top_k}")

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
            title = meta.get("title", "")
            source_file = meta.get("source_file", "")
            chunk_index = meta.get("chunk_index", "")
            snippet = doc[:180].replace("\n", " ")
            print(f"  {i}. score={dist:.4f} | {source_file} | {title} | chunk={chunk_index}")
            print(f"     {snippet}...")

    print("-" * 72)
    print("If results are weak: reduce chunk size first, then increase overlap, then refine keywords.")


if __name__ == "__main__":
    main()
