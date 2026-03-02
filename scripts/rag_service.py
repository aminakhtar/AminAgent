import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from scripts import rag_answer as core


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="about_me_default", min_length=1, max_length=120)
    history_turns: int = Field(default=4, ge=0, le=20)
    top_k: int = Field(default=3, ge=1, le=10)
    llm_provider: str = Field(default="openai-compatible")
    llm_model: str = Field(default="llama")
    llm_url: str = Field(default="http://127.0.0.1:8080")
    api_key: str = Field(default="")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    facts_only: bool = Field(default=False)
    debug_prompt: bool = Field(default=False)


class SourceItem(BaseModel):
    fact: str
    source_file: str
    section: str
    title: str
    date_range: str
    chunk_index: Any
    distance: float


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    used_fallback: bool
    sources: List[SourceItem]
    latency_ms: int
    debug: Optional[Dict[str, str]] = None


CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", r"D:\private-rag-data\chroma_db")
COLLECTION = os.getenv("RAG_COLLECTION", "work_background_v1")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/e5-small-v2")
SESSION_FILE = Path(os.getenv("RAG_SESSION_FILE", "sessions/rag_sessions.json"))


app = FastAPI(title="Internal RAG Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_state_lock = threading.Lock()
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def build_sources(result: Dict[str, Any]) -> List[SourceItem]:
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    output: List[SourceItem] = []
    for idx, (_, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        output.append(
            SourceItem(
                fact=f"FACT {idx}",
                source_file=str(meta.get("source_file", "")),
                section=str(meta.get("section", "")),
                title=str(meta.get("title", "")),
                date_range=str(meta.get("date_range", "")),
                chunk_index=meta.get("chunk_index", ""),
                distance=float(dist),
            )
        )
    return output


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "chroma_path": CHROMA_PATH,
        "collection": COLLECTION,
        "embed_model": EMBED_MODEL,
        "embedder_loaded": _embedder is not None,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.time()

    if request.llm_provider not in {"openai-compatible", "ollama"}:
        raise HTTPException(status_code=400, detail="llm_provider must be 'openai-compatible' or 'ollama'")

    try:
        with _state_lock:
            session_data = core.load_sessions(SESSION_FILE)
            session = core.get_or_create_session(session_data, request.session_id)
            recent_turns = core.get_recent_turns(session, request.history_turns)

        retrieval_top_k = max(request.top_k, 5) if core.is_company_question(request.message) else request.top_k
        retrieval_query = ""
        if core.is_company_question(request.message):
            retrieval_result = core.retrieve_company_facts_direct(
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION,
                top_k=retrieval_top_k,
            )
        else:
            embedder = get_embedder()
            retrieval_query = core.build_retrieval_query(request.message, recent_turns)
            query_embedding = core.embed_query(embedder, retrieval_query)
            retrieval_result = core.retrieve_facts(
                chroma_path=CHROMA_PATH,
                collection_name=COLLECTION,
                query_embedding=query_embedding,
                top_k=retrieval_top_k,
            )

        context = core.build_context(retrieval_result)
        system_prompt = core.build_system_prompt()
        user_prompt = core.build_user_prompt(request.message, context, recent_turns)

        used_fallback = False
        if request.facts_only:
            answer = f"FACTS ONLY MODE\n\n{context}"
        else:
            deterministic_answer = core.build_company_answer_from_facts(request.message, retrieval_result)
            if deterministic_answer is not None:
                answer = deterministic_answer
            else:
                try:
                    answer = core.call_llm(
                        provider=request.llm_provider,
                        model=request.llm_model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        llm_url=request.llm_url,
                        temperature=request.temperature,
                        api_key=request.api_key,
                    )
                except RuntimeError as exc:
                    used_fallback = True
                    answer = (
                        "LLM unavailable. Returning retrieved facts only.\n\n"
                        f"Error: {exc}\n\n"
                        f"{context}"
                    )

        with _state_lock:
            core.append_turn(
                session=session,
                question=request.message,
                answer=answer,
                retrieval_result=retrieval_result,
            )
            core.save_sessions(SESSION_FILE, session_data)

        latency_ms = int((time.time() - started) * 1000)
        debug_block = None
        if request.debug_prompt:
            debug_block = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "retrieval_query": retrieval_query,
            }

        return ChatResponse(
            answer=answer,
            session_id=request.session_id,
            used_fallback=used_fallback,
            sources=build_sources(retrieval_result),
            latency_ms=latency_ms,
            debug=debug_block,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
