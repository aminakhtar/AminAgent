import os
import sys
import threading
import time
import subprocess
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
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


class RefreshRequest(BaseModel):
    source_root: str = Field(default="source_docs", description="Folder containing source docs")
    reset_collection: bool = Field(default=True, description="Delete and recreate collection")
    skip_validation: bool = Field(default=False, description="Skip retrieval validation after indexing")


class RefreshResponse(BaseModel):
    status: str
    message: str
    records_processed: Optional[int] = None
    chunks_upserted: Optional[int] = None
    validation_passed: Optional[bool] = None
    error: Optional[str] = None
    timestamp: str


CHROMA_PATH = os.getenv("RAG_CHROMA_PATH", r"D:\private-rag-data\chroma_db")
COLLECTION = os.getenv("RAG_COLLECTION", "work_background_v1")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/e5-small-v2")
SESSION_FILE = Path(os.getenv("RAG_SESSION_FILE", "sessions/rag_sessions.json"))

# Refresh configuration
SOURCE_ROOT = Path(os.getenv("RAG_SOURCE_ROOT", "source_docs"))
REFRESH_LOG_FILE = Path(os.getenv("RAG_REFRESH_LOG", "refresh.log"))

# Refresh state
_refresh_lock = threading.Lock()
_refresh_in_progress = False
_refresh_result: Optional[Dict[str, Any]] = None


app = FastAPI(title="Internal RAG Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:5231", "http://127.0.0.1:5231"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_state_lock = threading.Lock()
_embedder: Optional[SentenceTransformer] = None


def log_refresh(message: str):
    """Log refresh messages to file"""
    timestamp = datetime.now().isoformat()
    log_entry = f"[{timestamp}] {message}\n"
    with open(REFRESH_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(log_entry.strip())


def refresh_facts_internal(source_root: str = "source_docs", reset: bool = True, skip_validation: bool = False) -> Dict[str, Any]:
    """
    Internal function to refresh facts in Chrome DB.
    Calls index_chroma.py and validate_queries.py directly.
    """
    global _refresh_in_progress, _refresh_result
    
    # Use the current Python executable (from the venv)
    python_exe = sys.executable
    
    try:
        with _refresh_lock:
            _refresh_in_progress = True
        
        log_refresh("🚀 Starting facts refresh...")
        
        # Step 1: Index Chroma
        log_refresh("[1/2] Reindexing facts into Chroma...")
        index_cmd = [
            python_exe,
            "scripts/index_chroma.py",
            "--source-root", source_root,
            "--chroma-path", CHROMA_PATH,
            "--collection", COLLECTION,
        ]
        if reset:
            index_cmd.append("--reset")
        
        result = subprocess.run(
            index_cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            error_msg = f"Indexing failed with exit code {result.returncode}: {result.stderr}"
            log_refresh(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        log_refresh(f"✅ Indexing complete: {result.stdout}")
        
        # Extract stats from output
        records_processed = 0
        chunks_upserted = 0
        for line in result.stdout.split("\n"):
            if "Records:" in line:
                try:
                    records_processed = int(line.split("Records:")[1].strip())
                except:
                    pass
            if "Chunks upserted:" in line:
                try:
                    chunks_upserted = int(line.split("Chunks upserted:")[1].strip())
                except:
                    pass
        
        # Step 2: Validate (optional)
        validation_passed = True
        if not skip_validation:
            log_refresh("[2/2] Running retrieval validation...")
            validate_cmd = [
                python_exe,
                "scripts/validate_queries.py",
                "--chroma-path", CHROMA_PATH,
                "--collection", COLLECTION,
                "--top-k", "3",
            ]
            
            result = subprocess.run(
                validate_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                log_refresh(f"⚠️ Validation had issues: {result.stderr}")
                validation_passed = False
            else:
                log_refresh(f"✅ Validation passed: {result.stdout}")
        
        success_msg = "Done. Facts are refreshed in Chroma."
        log_refresh(success_msg)
        
        return {
            "status": "success",
            "message": success_msg,
            "records_processed": records_processed,
            "chunks_upserted": chunks_upserted,
            "validation_passed": validation_passed,
            "timestamp": datetime.now().isoformat(),
        }
    
    except Exception as exc:
        error_msg = f"Refresh failed: {str(exc)}"
        log_refresh(f"❌ {error_msg}")
        return {
            "status": "error",
            "message": error_msg,
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    
    finally:
        with _refresh_lock:
            _refresh_in_progress = False
            _refresh_result = None


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
        "refresh_in_progress": _refresh_in_progress,
    }


@app.post("/admin/refresh-facts", response_model=RefreshResponse)
def refresh_facts(request: RefreshRequest) -> RefreshResponse:
    """
    Synchronously refresh facts in Chrome DB.
    This blocks until indexing and validation complete.
    """
    global _refresh_in_progress
    
    with _refresh_lock:
        if _refresh_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Refresh already in progress. Please wait for it to complete."
            )
        _refresh_in_progress = True
    
    try:
        source_path = Path(request.source_root)
        if not source_path.exists():
            source_path = SOURCE_ROOT / request.source_root if not source_path.is_absolute() else source_path
        
        if not source_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Source directory not found: {request.source_root}"
            )
        
        result = refresh_facts_internal(
            source_root=str(source_path),
            reset=request.reset_collection,
            skip_validation=request.skip_validation
        )
        
        return RefreshResponse(
            status=result["status"],
            message=result["message"],
            records_processed=result.get("records_processed"),
            chunks_upserted=result.get("chunks_upserted"),
            validation_passed=result.get("validation_passed"),
            error=result.get("error"),
            timestamp=result["timestamp"],
        )
    
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    
    finally:
        with _refresh_lock:
            _refresh_in_progress = False


@app.post("/admin/refresh-facts-background")
def refresh_facts_background(request: RefreshRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Asynchronously refresh facts in Chrome DB.
    Returns immediately while refresh happens in background.
    """
    global _refresh_in_progress
    
    with _refresh_lock:
        if _refresh_in_progress:
            raise HTTPException(
                status_code=409,
                detail="Refresh already in progress. Please wait for it to complete."
            )
        _refresh_in_progress = True
    
    source_path = Path(request.source_root)
    if not source_path.exists():
        source_path = SOURCE_ROOT / request.source_root if not source_path.is_absolute() else source_path
    
    if not source_path.exists():
        with _refresh_lock:
            _refresh_in_progress = False
        raise HTTPException(
            status_code=400,
            detail=f"Source directory not found: {request.source_root}"
        )
    
    background_tasks.add_task(
        refresh_facts_internal,
        source_root=str(source_path),
        reset=request.reset_collection,
        skip_validation=request.skip_validation
    )
    
    return {
        "status": "background_task_queued",
        "message": "Refresh task queued. Check /admin/refresh-status for progress.",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/admin/refresh-status")
def refresh_status() -> Dict[str, Any]:
    """Get current refresh status"""
    return {
        "refresh_in_progress": _refresh_in_progress,
        "last_result": _refresh_result,
        "log_file": str(REFRESH_LOG_FILE),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/admin/refresh-logs")
def get_refresh_logs(lines: int = 50) -> Dict[str, Any]:
    """Get recent refresh logs"""
    try:
        if not REFRESH_LOG_FILE.exists():
            return {
                "logs": [],
                "total_lines": 0,
                "message": "No logs available yet",
            }
        
        with open(REFRESH_LOG_FILE, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
        
        recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": [line.rstrip() for line in recent_lines],
            "total_lines": len(all_lines),
            "displayed_lines": len(recent_lines),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(exc)}")


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
