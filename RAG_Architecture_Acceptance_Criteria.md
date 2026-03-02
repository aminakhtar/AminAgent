# RAG Architecture + Acceptance Criteria

## Goal
Build a personal “about me” agent that answers using retrieved facts from your documents, while the local model is used only for phrasing.

## Scope and Principles
- No model weight changes are required.
- Retrieval supplies factual grounding; the model supplies wording.
- If facts are missing, the assistant must say so.
- If the model endpoint is unavailable, the system should degrade gracefully to facts-only output.

## Current Stack
- Source documents: `source_docs/` (resume, projects, achievements, preferences)
- Chunking and indexing: `scripts/index_chroma.py`
- Embeddings model: `intfloat/e5-small-v2`
- Vector DB: Chroma persistent store at `D:\private-rag-data\chroma_db`
- Retrieval validation: `scripts/validate_queries.py`
- Runtime RAG agent: `scripts/rag_answer.py`
- Local model endpoint: `http://127.0.0.1:8080`

## Reference Architecture
1. Ingestion + normalization
   - Parse markdown records with fields: Title, Date range, Context, What I did, Impact, Keywords.
   - Normalize whitespace; preserve headings and bullets.

2. Chunking
   - Primary split by section headers.
   - Chunk target: 500–700 chars.
   - Overlap: 100–150 chars.
   - Attach metadata: source_file, section, title, date_range, keywords, chunk_index.

3. Embedding + indexing
   - Embed each chunk as: `passage: <chunk text>`.
   - Use stable IDs for upsert-safe reruns.
   - Upsert into Chroma collection `work_background_v1`.

4. Query-time retrieval
   - Embed each user question as: `query: <question>`.
   - Retrieve top-k chunks (default 3).
   - Include metadata and distances for traceability.

5. Grounded generation
   - Inject retrieved FACT blocks into prompt.
   - Enforce rules: use only FACTS, no invention, say missing when unknown.
   - Return concise answer plus source references (`[FACT n]`).

6. Session memory
   - Persist turns in `sessions/rag_sessions.json`.
   - Use recent turns for conversational continuity and retrieval query rewriting.
   - Do not treat prior assistant output as source-of-truth; only retrieved FACTS are authoritative.

7. Resilience
   - Try OpenAI-compatible endpoint first (`/v1/chat/completions`).
   - Optionally fallback to llama.cpp `/completion`.
   - If endpoint unavailable, fallback to facts-only turn output and keep session alive.

## Prompt Contract (Runtime)
System intent:
- Grounded assistant for personal background.
- Facts-only answering from retrieved blocks.
- Explicitly report missing information.

User payload includes:
- Recent conversation snippets (for continuity only).
- Current question.
- FACT blocks with source metadata and chunk text.

## Acceptance Criteria

### A) Data + Indexing
- [ ] Every indexed chunk includes required metadata keys.
- [ ] Chunk text uses `passage:` prefix before embedding.
- [ ] Query text uses `query:` prefix before embedding.
- [ ] Collection upserts are stable across reruns (no duplicate drift).

### B) Retrieval Quality
- [ ] For each test category (performance, leadership, tech stack, achievements, preferences), top-3 includes at least one relevant chunk.
- [ ] Retrieved chunks include source metadata sufficient for citation.

### C) Grounded Answering
- [ ] Responses are based only on retrieved FACT blocks.
- [ ] If facts are insufficient, response states what is missing.
- [ ] Response includes source references (e.g., `Sources: [FACT 1], [FACT 2]`).

### D) Reliability and Failure Handling
- [ ] If model server is down, system does not crash interactive session.
- [ ] On model failure, facts-only fallback is returned for that turn.
- [ ] Session continues accepting subsequent user questions.

### E) Session Behavior
- [ ] Turns are persisted with timestamp, question, answer, and fact snippets.
- [ ] `--session-id` isolates conversation histories.
- [ ] `--history-turns` controls continuity without overriding factual grounding.

## Operational Commands
Index:
```powershell
& "D:\private-rag-data\venv\Scripts\python.exe" scripts/index_chroma.py \
  --source-root source_docs \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 --reset
```

Validate retrieval:
```powershell
& "D:\private-rag-data\venv\Scripts\python.exe" scripts/validate_queries.py \
  --chroma-path D:\private-rag-data\chroma_db \
  --collection work_background_v1 --top-k 3
```

Run interactive agent:
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

## Non-Goals
- No fine-tuning or weight updates in this phase.
- No autonomous web browsing or external fact injection.
- No replacing retrieval with long-context-only prompting.

## Change Control (Minimal)
When changing chunk sizes, overlap, embeddings model, or prompts:
1. Reindex from source.
2. Re-run 5-category retrieval validation.
3. Compare grounding behavior and fallback behavior.
4. Promote only if acceptance criteria remain satisfied.
