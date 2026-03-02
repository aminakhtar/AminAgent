import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
TITLE_RE = re.compile(r"^Title:\s*(.+)$", re.IGNORECASE)
DATE_RE = re.compile(r"^Date range:\s*(.+)$", re.IGNORECASE)
KEYWORDS_RE = re.compile(r"^Keywords:\s*(.+)$", re.IGNORECASE)


@dataclass
class Record:
    source_file: str
    section: str
    title: str
    date_range: str
    keywords: List[str]
    text: str


def normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    lines = [line for line in lines if line.strip()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def split_sections(text: str, fallback_section: str) -> List[Tuple[str, str]]:
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_section = fallback_section
    current_lines: List[str] = []

    for line in lines:
        header_match = HEADER_RE.match(line.strip())
        if header_match:
            if current_lines:
                sections.append((current_section, current_lines))
            current_section = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    return [(name, "\n".join(block).strip()) for name, block in sections if "\n".join(block).strip()]


def split_records(section_text: str) -> List[str]:
    lines = section_text.splitlines()
    records: List[List[str]] = []
    current: List[str] = []

    for line in lines:
        if TITLE_RE.match(line.strip()):
            if current:
                records.append(current)
            current = [line]
        elif current:
            current.append(line)

    if current:
        records.append(current)

    return ["\n".join(r).strip() for r in records if "\n".join(r).strip()]


def parse_record(source_file: str, section: str, record_text: str) -> Record:
    title = "Untitled"
    date_range = "Unknown"
    keywords: List[str] = []

    for raw_line in record_text.splitlines():
        line = raw_line.strip()
        title_match = TITLE_RE.match(line)
        date_match = DATE_RE.match(line)
        keywords_match = KEYWORDS_RE.match(line)

        if title_match:
            title = title_match.group(1).strip()
        elif date_match:
            date_range = date_match.group(1).strip()
        elif keywords_match:
            keywords = [part.strip() for part in keywords_match.group(1).split(",") if part.strip()]

    return Record(
        source_file=source_file,
        section=section,
        title=title,
        date_range=date_range,
        keywords=keywords,
        text=normalize_text(record_text),
    )


def sentence_aware_cut(text: str, start: int, target_end: int, max_end: int) -> int:
    candidate = text[start:max_end]
    if not candidate:
        return start

    preferred_breaks = [". ", "! ", "? ", "\n"]
    best_pos = -1

    for marker in preferred_breaks:
        pos = candidate.rfind(marker, 0, max(1, target_end - start))
        if pos > best_pos:
            best_pos = pos + len(marker)

    if best_pos == -1:
        return min(max_end, target_end)

    return start + best_pos


def chunk_record_text(text: str, min_chars: int, target_chars: int, max_chars: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    n = len(text)
    start = 0

    while start < n:
        target_end = min(n, start + target_chars)
        max_end = min(n, start + max_chars)

        if n - start <= max_chars:
            end = n
        else:
            end = sentence_aware_cut(text, start, target_end, max_end)
            if end - start < min_chars:
                end = min(n, start + min_chars)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(0, end - overlap)

    return chunks


def stable_id(record: Record, record_index: int, chunk_index: int) -> str:
    source_slug = re.sub(r"[^a-z0-9]+", "_", Path(record.source_file).stem.lower()).strip("_") or "doc"
    title_slug = re.sub(r"[^a-z0-9]+", "_", record.title.lower()).strip("_") or "untitled"
    years = re.findall(r"(19\d{2}|20\d{2}|21\d{2})", record.date_range)
    date_slug = "_".join(years) if years else "unknown"
    return f"{source_slug}_{date_slug}_{title_slug[:40]}_{record_index:03d}_{chunk_index:03d}"


def gather_records(source_root: Path) -> List[Record]:
    records: List[Record] = []
    files = sorted(source_root.rglob("*.md"))

    for file in files:
        text = file.read_text(encoding="utf-8")
        fallback = file.stem
        for section, section_text in split_sections(text, fallback_section=fallback):
            for record_text in split_records(section_text):
                record = parse_record(str(file.relative_to(source_root)).replace("\\", "/"), section, record_text)
                records.append(record)

    return records


def build_chunks(records: List[Record], min_chars: int, target_chars: int, max_chars: int, overlap: int):
    docs = []
    metadatas = []
    ids = []

    for record_index, record in enumerate(records):
        text_chunks = chunk_record_text(record.text, min_chars, target_chars, max_chars, overlap)
        for chunk_index, chunk_text in enumerate(text_chunks):
            prefixed = f"passage: {chunk_text}"
            docs.append(prefixed)
            metadatas.append(
                {
                    "source_file": record.source_file,
                    "section": record.section,
                    "title": record.title,
                    "date_range": record.date_range,
                    "keywords": json.dumps(record.keywords, ensure_ascii=False),
                    "chunk_index": chunk_index,
                }
            )
            ids.append(stable_id(record, record_index, chunk_index))

    return docs, metadatas, ids


def main():
    parser = argparse.ArgumentParser(description="Chunk markdown docs and upsert into Chroma with e5 prefixes.")
    parser.add_argument("--source-root", required=True, help="Folder containing source docs.")
    parser.add_argument("--chroma-path", required=True, help="Persistent Chroma DB folder.")
    parser.add_argument("--collection", default="work_background_v1", help="Chroma collection name.")
    parser.add_argument("--model", default="intfloat/e5-small-v2", help="SentenceTransformer model.")
    parser.add_argument("--chunk-min", type=int, default=500)
    parser.add_argument("--chunk-target", type=int, default=600)
    parser.add_argument("--chunk-max", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection before upsert.")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    records = gather_records(source_root)
    if not records:
        raise RuntimeError("No records found. Ensure each entry starts with 'Title:'.")

    docs, metadatas, ids = build_chunks(
        records,
        min_chars=args.chunk_min,
        target_chars=args.chunk_target,
        max_chars=args.chunk_max,
        overlap=args.overlap,
    )

    model = SentenceTransformer(args.model)
    embeddings = model.encode(docs, normalize_embeddings=True, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=args.chroma_path)
    if args.reset:
        try:
            client.delete_collection(name=args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    print(f"Records: {len(records)}")
    print(f"Chunks upserted: {len(ids)}")
    print(f"Collection: {args.collection}")
    print(f"DB path: {args.chroma_path}")


if __name__ == "__main__":
    main()
