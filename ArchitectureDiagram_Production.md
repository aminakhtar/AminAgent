# Amin Personal Agent - Production / Online Deployment Architecture

```mermaid
flowchart TD
    Internet[Internet Users / Client Apps] --> WAF[Cloud WAF / Rate Limit / TLS]
    WAF --> ASP[Public ASP.NET Core API\n/api/chat\nAuth + Validation + Logging]

    ASP --> RAG[Private Python RAG Service\nFastAPI /chat\nInternal Network Only]

    RAG --> RET[Retriever\nE5 query embedding\nquery: <question>]
    RET --> CDB[(Private Chroma DB Volume\ncollection: work_background_v1)]

    CDB --> FACTS[Top-K FACT Chunks + Metadata]
    FACTS --> PROMPT[Prompt Builder\nGrounding Rules\nCitations]

    PROMPT --> LLM[Private Local Llama Server\n127.0.0.1:8080 or internal host]
    LLM --> RAG
    RAG --> ASP
    ASP --> Internet

    RAG --> SESS[(Session Store\nJSON/SQLite/Postgres)]

    CICD[CI/CD] --> ASP
    CICD --> RAG

    SRC[(source_docs/*.md)] --> IDX[Index Job\nindex_chroma.py\npassage: <chunk>]
    IDX --> CDB

    MON[Monitoring + Alerts\nlatency/errors/fallback rate] -.-> ASP
    MON -.-> RAG

    SECRETS[Secret Manager\nAPI keys/config] -.-> ASP
    SECRETS -.-> RAG

    classDef public fill:#fff4e5,stroke:#d18b00,color:#6d4500;
    classDef private fill:#eef6ff,stroke:#6aa0ff,color:#1f3b73;

    class Internet,WAF,ASP public;
    class RAG,RET,CDB,FACTS,PROMPT,LLM,SESS,IDX,SECRETS,MON private;
```

## Notes
- Keep `RAG`, `Chroma`, and `LLM` private (not internet-exposed).
- Publicly expose only ASP.NET API behind TLS + auth + rate limiting.
- Continue using e5 prefixes (`passage:` for chunks, `query:` for questions).
- Run indexing as a controlled internal job after source updates.
