import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)


def load_sessions(session_file: Path) -> Dict[str, Any]:
    if not session_file.exists():
        return {"sessions": {}}

    raw = session_file.read_text(encoding="utf-8").strip()
    if not raw:
        return {"sessions": {}}

    data = json.loads(raw)
    if "sessions" not in data or not isinstance(data["sessions"], dict):
        return {"sessions": {}}
    return data


def save_sessions(session_file: Path, data: Dict[str, Any]) -> None:
    ensure_parent_dir(session_file)
    session_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_or_create_session(data: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    sessions = data.setdefault("sessions", {})
    if session_id not in sessions:
        sessions[session_id] = {
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "turns": [],
        }
    return sessions[session_id]


def get_recent_turns(session: Dict[str, Any], max_turns: int) -> List[Dict[str, Any]]:
    turns = session.get("turns", [])
    if not isinstance(turns, list):
        return []
    if max_turns <= 0:
        return []
    return turns[-max_turns:]


def build_retrieval_query(question: str, recent_turns: List[Dict[str, Any]]) -> str:
    q = question.strip()
    explicit_keywords = [
        "company", "companies", "worked", "employer", "experience", "projects",
        "skills", "tech", "technology", "stack", "leadership", "achievement",
        "where", "which", "what", "who", "list",
    ]
    follow_up_markers = ["that", "those", "it", "them", "this", "summarize", "more", "elaborate"]

    lower_q = q.lower()
    has_explicit_intent = any(token in lower_q for token in explicit_keywords)
    has_follow_up_marker = any(token in lower_q for token in follow_up_markers)

    if has_explicit_intent and not has_follow_up_marker:
        return q

    if not recent_turns:
        return q

    recent_questions = [
        str(turn.get("question", "")).strip()
        for turn in recent_turns
        if str(turn.get("question", "")).strip()
    ]
    if not recent_questions:
        return q

    stitched = " | ".join(recent_questions[-2:])
    return f"Current question: {q} | Recent conversation context: {stitched}"


def _clean_company_name(raw: str) -> str:
    candidate = raw.strip().strip(".,;: ")
    candidate = re.sub(r"\s+", " ", candidate)
    candidate = re.sub(r"^the\s+", "", candidate, flags=re.IGNORECASE)
    return candidate


def is_company_question(question: str) -> bool:
    lower = question.lower()
    triggers = ["company", "companies", "employer", "employers", "worked for", "where did i work"]
    return any(trigger in lower for trigger in triggers)


def build_company_answer_from_facts(question: str, retrieval_result: Dict[str, Any]) -> str | None:
    if not is_company_question(question):
        return None

    metas = retrieval_result.get("metadatas", [[]])[0]
    documents = retrieval_result.get("documents", [[]])[0]

    company_to_fact: Dict[str, List[int]] = {}

    for idx, (meta, doc) in enumerate(zip(metas, documents), start=1):
        section = str(meta.get("section", "")).strip().lower()
        if section and section != "resume":
            continue

        candidates: List[str] = []

        title = str(meta.get("title", "")).strip()
        if "," in title:
            candidates.append(title.split(",")[-1])
        if " at " in title.lower():
            parts = re.split(r"\bat\b", title, flags=re.IGNORECASE)
            if len(parts) > 1:
                candidates.append(parts[-1])

        clean_doc = doc.removeprefix("passage: ")
        title_match = re.search(r"^Title:\s*(.+)$", clean_doc, flags=re.IGNORECASE | re.MULTILINE)
        if title_match:
            t = title_match.group(1)
            if "," in t:
                candidates.append(t.split(",")[-1])
            if " at " in t.lower():
                parts = re.split(r"\bat\b", t, flags=re.IGNORECASE)
                if len(parts) > 1:
                    candidates.append(parts[-1])

        for raw in candidates:
            company = _clean_company_name(raw)
            if not company:
                continue
            if len(company) < 2:
                continue
            lowered = company.lower()
            if lowered in {
                "developer", "engineer", "mentor", "backend", "full-stack", "fullstack",
                "asp.net core", "entity framework", "sql server", "azure", "aws developer",
            }:
                continue
            company_to_fact.setdefault(company, [])
            if idx not in company_to_fact[company]:
                company_to_fact[company].append(idx)

    if not company_to_fact:
        return "The retrieved facts do not clearly specify company names. Sources: [FACT 1]"

    companies = list(company_to_fact.keys())
    refs: List[str] = []
    for _, fact_indexes in company_to_fact.items():
        refs.extend([f"[FACT {i}]" for i in fact_indexes])

    dedup_refs: List[str] = []
    for ref in refs:
        if ref not in dedup_refs:
            dedup_refs.append(ref)

    companies_text = ", ".join(companies)
    return f"Based on retrieved facts, Amin has worked with: {companies_text}.\n\nSources: {', '.join(dedup_refs)}"


def embed_query(model: SentenceTransformer, query_text: str) -> List[float]:
    prefixed = f"query: {query_text.strip()}"
    return model.encode([prefixed], normalize_embeddings=True).tolist()[0]


def retrieve_facts(
    chroma_path: str,
    collection_name: str,
    query_embedding: List[float],
    top_k: int,
) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def retrieve_company_facts_direct(chroma_path: str, collection_name: str, top_k: int) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(name=collection_name)
    got = collection.get(
        where={"section": "Resume"},
        limit=top_k,
        include=["documents", "metadatas"],
    )

    documents = got.get("documents", [])
    metadatas = got.get("metadatas", [])
    distances = [0.0 for _ in documents]
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def build_context(result: Dict[str, Any]) -> str:
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    blocks: List[str] = []
    for idx, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        clean_doc = doc.removeprefix("passage: ").strip()
        excerpt = clean_doc[:420].rstrip()
        if len(clean_doc) > 420:
            excerpt += "..."
        block = (
            f"[FACT {idx}]\n"
            f"source_file: {meta.get('source_file', '')}\n"
            f"section: {meta.get('section', '')}\n"
            f"title: {meta.get('title', '')}\n"
            f"date_range: {meta.get('date_range', '')}\n"
            f"keywords: {meta.get('keywords', '')}\n"
            f"chunk_index: {meta.get('chunk_index', '')}\n"
            f"similarity_distance: {dist:.4f}\n"
            f"text: {excerpt}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_history_context(recent_turns: List[Dict[str, Any]]) -> str:
    if not recent_turns:
        return "(no prior turns)"

    lines: List[str] = []
    trimmed = recent_turns[-2:]
    for idx, turn in enumerate(trimmed, start=1):
        question = str(turn.get("question", "")).strip()
        if len(question) > 180:
            question = question[:180].rstrip() + "..."
        lines.append(f"[TURN {idx} USER] {question}")

    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a grounded assistant for a personal background agent. "
        "Use only the provided FACT blocks. "
        "Never invent achievements, dates, technologies, or employers not present in FACTS. "
        "If details are missing, say they are not available in the retrieved facts. "
        "Recent conversation turns are for tone and continuity only, not as factual sources. "
        "Keep the response concise and professional."
    )


def build_user_prompt(question: str, context: str, recent_turns: List[Dict[str, Any]]) -> str:
    history = build_history_context(recent_turns)
    return (
        "Recent user context:\n"
        f"{history}\n\n"
        "Current question:\n"
        f"{question}\n\n"
        "FACTS:\n"
        f"{context}\n\n"
        "Answer briefly using only FACTS. "
        "If a fact is missing, explicitly state what is missing. "
        "End with a short Sources list like: Sources: [FACT 1], [FACT 2]."
    )


def call_openai_compatible(
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    temperature: float,
    api_key: str,
) -> str:
    url = urljoin(base_url.rstrip("/") + "/", "v1/chat/completions")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 220,
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach OpenAI-compatible API at {url}. "
            "Confirm your server is running (e.g., http://127.0.0.1:8080)."
        ) from exc

    parsed = json.loads(body)
    choices = parsed.get("choices", [])
    if not choices:
        raise RuntimeError("LLM response did not include choices.")

    message = choices[0].get("message", {})
    text = str(message.get("content", "")).strip()
    if not text:
        raise RuntimeError("LLM returned an empty message content.")
    return text


def call_llama_cpp_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    base_url: str,
    temperature: float,
) -> str:
    url = urljoin(base_url.rstrip("/") + "/", "completion")
    merged_prompt = f"{system_prompt}\n\n{user_prompt}"
    payload = {
        "prompt": merged_prompt,
        "temperature": temperature,
        "n_predict": 220,
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach llama.cpp completion endpoint at {url}."
        ) from exc

    parsed = json.loads(body)
    text = str(parsed.get("content", "")).strip()
    if not text:
        raise RuntimeError("llama.cpp completion endpoint returned empty content.")
    return text


def call_ollama(model: str, prompt: str, url: str, temperature: float) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama API. Start Ollama and confirm the URL, e.g. http://localhost:11434/api/generate"
        ) from exc

    parsed = json.loads(body)
    text = parsed.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return text


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    llm_url: str,
    temperature: float,
    api_key: str,
) -> str:
    if provider == "openai-compatible":
        try:
            return call_openai_compatible(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_url=llm_url,
                temperature=temperature,
                api_key=api_key,
            )
        except RuntimeError as primary_error:
            try:
                return call_llama_cpp_completion(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    base_url=llm_url,
                    temperature=temperature,
                )
            except RuntimeError:
                raise primary_error

    if provider == "ollama":
        merged_prompt = f"{system_prompt}\n\n{user_prompt}"
        return call_ollama(
            model=model,
            prompt=merged_prompt,
            url=llm_url,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def print_retrieval_preview(result: Dict[str, Any], top_k: int) -> None:
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    print("Retrieved facts:")
    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        title = meta.get("title", "")
        source = meta.get("source_file", "")
        snippet = doc[:140].replace("\n", " ")
        print(f"  {idx}/{top_k}. score={dist:.4f} | {source} | {title}")
        print(f"     {snippet}...")


def append_turn(
    session: Dict[str, Any],
    question: str,
    answer: str,
    retrieval_result: Dict[str, Any],
) -> None:
    docs = retrieval_result.get("documents", [[]])[0]
    metas = retrieval_result.get("metadatas", [[]])[0]
    dists = retrieval_result.get("distances", [[]])[0]

    facts: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        facts.append(
            {
                "source_file": meta.get("source_file", ""),
                "section": meta.get("section", ""),
                "title": meta.get("title", ""),
                "date_range": meta.get("date_range", ""),
                "chunk_index": meta.get("chunk_index", ""),
                "distance": round(float(dist), 6),
                "snippet": doc[:220],
            }
        )

    session.setdefault("turns", []).append(
        {
            "timestamp_utc": utc_now_iso(),
            "question": question,
            "answer": answer,
            "facts": facts,
        }
    )
    session["updated_at"] = utc_now_iso()


def run_turn(
    question: str,
    args: argparse.Namespace,
    embedder: SentenceTransformer,
    session: Dict[str, Any],
) -> str:
    recent_turns = get_recent_turns(session, args.history_turns)
    retrieval_top_k = max(args.top_k, 5) if is_company_question(question) else args.top_k

    if is_company_question(question):
        result = retrieve_company_facts_direct(
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            top_k=retrieval_top_k,
        )
    else:
        retrieval_query = build_retrieval_query(question, recent_turns)
        q_embedding = embed_query(embedder, retrieval_query)
        result = retrieve_facts(
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            query_embedding=q_embedding,
            top_k=retrieval_top_k,
        )

    print_retrieval_preview(result, args.top_k)

    context = build_context(result)
    if args.facts_only:
        answer = f"FACTS ONLY MODE\n\n{context}"
        print("\n--- FACT CONTEXT ---\n")
        print(context)
    else:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(question, context, recent_turns)
        if args.debug_prompt:
            print("\n--- SYSTEM PROMPT ---\n")
            print(system_prompt)
            print("\n--- USER PROMPT (WITH FACTS) ---\n")
            print(user_prompt)
        try:
            deterministic_answer = build_company_answer_from_facts(question, result)
            if deterministic_answer is not None:
                answer = deterministic_answer
            else:
                answer = call_llm(
                    provider=args.llm_provider,
                    model=args.llm_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    llm_url=args.llm_url,
                    temperature=args.temperature,
                    api_key=args.api_key,
                )
            print("\n=== RAG ANSWER ===\n")
            print(answer)
        except RuntimeError as exc:
            if args.fallback_facts_on_llm_error:
                answer = (
                    "LLM unavailable. Returning retrieved facts only.\n\n"
                    f"Error: {exc}\n\n"
                    f"{context}"
                )
                print("\n=== LLM ERROR (FALLBACK TO FACTS) ===\n")
                print(str(exc))
                print("\n--- FACT CONTEXT ---\n")
                print(context)
            else:
                raise

    append_turn(session=session, question=question, answer=answer, retrieval_result=result)
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid RAG: facts from Chroma + phrasing by local model.")
    parser.add_argument("--question", default="", help="Single-turn question")
    parser.add_argument("--interactive", action="store_true", help="Run interactive multi-turn chat mode")
    parser.add_argument("--session-id", default="about_me_default", help="Session identifier")
    parser.add_argument(
        "--session-file",
        default="sessions/rag_sessions.json",
        help="Path to persisted chat session store JSON",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=4,
        help="How many recent turns to include for continuity and retrieval query rewriting",
    )
    parser.add_argument("--chroma-path", required=True, help="Path to persistent Chroma DB")
    parser.add_argument("--collection", default="work_background_v1", help="Collection name")
    parser.add_argument("--embed-model", default="intfloat/e5-small-v2", help="Embedding model")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--llm-provider",
        choices=["openai-compatible", "ollama"],
        default="openai-compatible",
        help="LLM backend provider",
    )
    parser.add_argument("--llm-model", default="llama", help="Model name exposed by your local server")
    parser.add_argument(
        "--llm-url",
        default="http://127.0.0.1:8080",
        help="For openai-compatible use base URL, for ollama use full /api/generate URL",
    )
    parser.add_argument("--api-key", default="", help="Optional API key for OpenAI-compatible servers")
    parser.add_argument("--temperature", type=float, default=0.2, help="Lower is better for factual grounding")
    parser.add_argument("--facts-only", action="store_true", help="Print retrieved facts without calling LLM")
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Print the exact system/user prompts sent to the model, including injected FACT blocks.",
    )
    parser.add_argument(
        "--fallback-facts-on-llm-error",
        action="store_true",
        default=True,
        help="If model endpoint is unavailable, keep chat alive and return facts-only for that turn.",
    )
    parser.add_argument(
        "--no-fallback-facts-on-llm-error",
        dest="fallback_facts_on_llm_error",
        action="store_false",
        help="Disable facts-only fallback when LLM calls fail.",
    )
    args = parser.parse_args()

    if not args.interactive and not args.question.strip():
        raise ValueError("Provide --question for single turn, or use --interactive mode.")

    session_file = Path(args.session_file)
    session_data = load_sessions(session_file)
    session = get_or_create_session(session_data, args.session_id)

    embedder = SentenceTransformer(args.embed_model)

    if args.interactive:
        print(f"Session: {args.session_id}")
        print("Interactive mode started. Type 'exit' to stop.")
        while True:
            try:
                question = input("\nYou: ").strip()
            except EOFError:
                print("\nStopped.")
                break

            if not question:
                continue

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            try:
                run_turn(question=question, args=args, embedder=embedder, session=session)
                save_sessions(session_file, session_data)
            except Exception as exc:
                print("\nError during turn:")
                print(str(exc))
                print("The session is still active. Ask another question or type 'exit'.")
    else:
        run_turn(question=args.question.strip(), args=args, embedder=embedder, session=session)
        save_sessions(session_file, session_data)


if __name__ == "__main__":
    main()
