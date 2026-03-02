# Hybrid RAG: facts + Llama phrasing

Use Chroma retrieval for facts, then pass only retrieved facts to your local model at `http://127.0.0.1:8080` for answer wording.

## Why this works
- Retriever is responsible for factual grounding.
- Llama is responsible for phrasing and synthesis.
- Prompt constrains model to avoid hallucinations.

## Required rule (e5)
- Query embedding must use: `query: <question>`
- Chunk embeddings must use: `passage: <chunk text>`

Your existing indexer already stores `passage:` chunks, and `scripts/rag_answer.py` uses `query:`.

## Run

```powershell
& "D:\private-rag-data\venv\Scripts\python.exe" scripts/rag_answer.py \
  --question "What leadership impact did I deliver?" \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 \
  --llm-provider openai-compatible \
  --llm-url http://127.0.0.1:8080 \
  --llm-model llama \
  --session-id amin_about_me \
  --session-file sessions/rag_sessions.json \
  --top-k 3
```

If your server requires a token, add:

```powershell
  --api-key YOUR_TOKEN
```

## Facts-only debug mode

```powershell
& "D:\private-rag-data\venv\Scripts\python.exe" scripts/rag_answer.py \
  --question "What performance improvements have I delivered?" \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 \
  --session-id amin_about_me \
  --facts-only
```

## Interactive multi-turn mode

```powershell
& "D:\private-rag-data\venv\Scripts\python.exe" scripts/rag_answer.py \
  --interactive \
  --session-id amin_about_me \
  --session-file sessions/rag_sessions.json \
  --history-turns 4 \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 \
  --llm-provider openai-compatible \
  --llm-url http://127.0.0.1:8080 \
  --llm-model llama \
  --top-k 3
```

The session file stores turn-by-turn questions, answers, and retrieved fact snippets.

## Prompting pattern
- Include `[FACT n]` blocks with metadata and text.
- Instruct Llama: “Use only these facts; if missing, say missing.”
- Require answer to end with `Sources: [FACT n], ...`.

## Production guardrails
- Keep `temperature` low (`0.0` to `0.3`).
- Keep `top-k` small (`3` to `5`) for precision.
- Add fallback response when no relevant chunk is found.
- Log retrieved chunks for traceability.

## Notes on endpoint shape
- This script expects an OpenAI-compatible endpoint at `/v1/chat/completions`.
- For an Ollama backend, set `--llm-provider ollama` and pass `--llm-url http://localhost:11434/api/generate`.
