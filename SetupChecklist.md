# Local RAG Setup Checklist (copy-this)

1. Create external Python env (outside repo):
   - Example path: `D:\private-rag-data\venv`
2. Install dependencies:
   - `sentence-transformers`, `chromadb`, `numpy`
3. Put source docs under one root folder with subfolders:
   - `resume/`, `projects/`, `achievements/`, `preferences/`
4. Ensure each record uses:
   - `Title`, `Date range`, `Context`, `What I did`, `Impact`, `Keywords`
5. Run indexing sequence:
   - load -> normalize -> chunk -> prefix -> embed -> upsert
6. Use e5 prefixes exactly:
   - document chunks: `passage: <chunk text>`
   - user queries: `query: <question>`
7. Use persistent Chroma path:
   - Example: `D:\private-rag-data\chroma_db`
8. Use one collection:
   - Example: `work_background_v1`
9. Use stable chunk IDs for safe reruns/upserts:
   - Example: `resume_2022_2024_003`
10. Run immediate validation with 5 query categories:
   - performance, leadership, tech stack, achievements, preferences
11. Relevance rule:
   - Check top-3 are relevant.
12. Tuning order if weak:
   - reduce chunk size -> increase overlap -> refine keywords

## Commands

### 1) Index

```powershell
python scripts/index_chroma.py \
  --source-root source_docs \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1
```

### 2) Validate

```powershell
python scripts/validate_queries.py \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 \
  --top-k 3
```

### 3) Hybrid RAG answer (facts + Llama phrasing)

```powershell
python scripts/rag_answer.py \\
   --question "What leadership impact did I deliver?" \\
   --chroma-path D:\\private-rag-data\\chroma_db \\
   --collection work_background_v1 \\
   --llm-provider openai-compatible \\
   --llm-url http://127.0.0.1:8080 \\
   --llm-model llama \\
   --session-id amin_about_me \\
   --session-file sessions/rag_sessions.json \\
   --top-k 3
```

### 4) Interactive multi-turn mode (session memory)

```powershell
python scripts/rag_answer.py \\
   --interactive \\
   --session-id amin_about_me \\
   --session-file sessions/rag_sessions.json \\
   --history-turns 4 \\
   --chroma-path D:\\private-rag-data\\chroma_db \\
   --collection work_background_v1 \\
   --llm-provider openai-compatible \\
   --llm-url http://127.0.0.1:8080 \\
   --llm-model llama \\
   --top-k 3
```

### 5) Debug injected FACT prompt

```powershell
python scripts/rag_answer.py \\
   --question "what is drink tracker?" \\
   --session-id amin_about_me \\
   --session-file sessions/rag_sessions.json \\
   --history-turns 4 \\
   --chroma-path D:\\private-rag-data\\chroma_db \\
   --collection work_background_v1 \\
   --llm-provider openai-compatible \\
   --llm-url http://127.0.0.1:8080 \\
   --llm-model llama \\
   --top-k 3 \\
   --debug-prompt
```

### 6) Internal HTTP RAG service (`/chat`)

Run internal service:

```powershell
python -m uvicorn scripts.rag_service:app --host 127.0.0.1 --port 8090
```

Health check:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8090/health" -Method Get
```

Chat request:

```powershell
$body = @{
   message = "What is Drink Tracker?"
   session_id = "amin_about_me"
   history_turns = 4
   top_k = 3
   llm_provider = "openai-compatible"
   llm_url = "http://127.0.0.1:8080"
   llm_model = "llama"
   debug_prompt = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8090/chat" -Method Post -ContentType "application/json" -Body $body
```

### 7) One-command fact refresh (reindex + validate)

```powershell
.\scripts\refresh_facts.ps1
```

Optional (skip validation):

```powershell
.\scripts\refresh_facts.ps1 -SkipValidation
```
