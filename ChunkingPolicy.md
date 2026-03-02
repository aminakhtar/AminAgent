# Chunking Policy (Step 1)

## Objective
Make each chunk:
- self-contained (understandable without reading other chunks),
- searchable (good keyword and semantic retrieval),
- small enough for reliable embeddings.

## Canonical Record Structure (required in every chunk)
Each chunk **must** preserve these fields:
1. Title
2. Date range
3. Context
4. What I did
5. Impact
6. Keywords

Use this exact shape in source docs:

```md
Title: ...
Date range: ...
Context: ...
What I did: ...
Impact: ...
Keywords: ...
```

## Splitting Strategy

### 1) Primary split: by section headers
- First split by Markdown section headers (`#`, `##`, `###`).
- Section examples: `Resume`, `Projects`, `Preferences`.
- If a file has no header, treat filename stem as section (e.g., `Resume.md` -> `Resume`).

### 2) Secondary split: by record boundary
Inside each section, split by record boundary:
- start a new record at each line matching `^Title:\s+`.

### 3) Tertiary split: by character window
For each record text:
- target chunk size: **500–700 characters**,
- hard target: **~600 characters**,
- overlap between adjacent chunks: **100–150 characters** (default **120**).

Use a sentence-aware sliding window where possible:
- prefer breaking at sentence end (`.`, `!`, `?`) or newline,
- if no clean break exists near boundary, hard-cut at the nearest limit.

## Chunk Construction Rules
1. Every chunk includes a small header block for self-containment:
   - `Title`, `Date range`, `Context` (even if truncated context).
2. Preserve original wording for factual claims.
3. Keep field labels in chunk text (do not remove `Title:` / `Impact:` etc.).
4. Maintain chronological text exactly as in source.
5. Do not merge two different `Title` records into one chunk.

## Metadata Schema (required per chunk)
Store these metadata keys exactly:

```json
{
  "source_file": "string", 
  "section": "string",
  "title": "string",
  "date_range": "string",
  "keywords": ["string"],
  "chunk_index": 0
}
```

### Metadata Rules
- `source_file`: relative filename (e.g., `Resume.md`, `Projects.md`).
- `section`: parent section header (`Resume`, `Projects`, etc.).
- `title`: value parsed from `Title:` line.
- `date_range`: value parsed from `Date range:` line.
- `keywords`: array split from `Keywords:` by comma, trimmed.
- `chunk_index`: 0-based index within a single `(source_file, section, title)` record.

## Recommended Defaults
- `chunk_target_chars = 600`
- `chunk_min_chars = 500`
- `chunk_max_chars = 700`
- `chunk_overlap_chars = 120`
- `split_on_sentence = true`
- `strip_empty_lines = true`

## Example (conceptual)
Given one record under section `Projects`:
- If record length is 1,450 chars:
  - chunk 0: chars 0–~620
  - chunk 1: starts at ~500 (120 overlap), ends ~1,120
  - chunk 2: starts at ~1,000 (120 overlap), ends at record end

Each chunk keeps the field labels and gets metadata with increasing `chunk_index`.

## Validation Checklist
Before indexing, verify:
- [ ] each chunk length is between 500–700 chars (or close for final chunk),
- [ ] overlap is 100–150 chars,
- [ ] all 6 record fields appear in chunk text,
- [ ] metadata has all required keys,
- [ ] `chunk_index` ordering is continuous per record.
