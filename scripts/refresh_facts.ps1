param(
    [string]$PythonExe = "D:\private-rag-data\venv\Scripts\python.exe",
    [string]$SourceRoot = "source_docs",
    [string]$ChromaPath = "D:\private-rag-data\chroma_db",
    [string]$Collection = "work_background_v1",
    [int]$TopK = 3,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at: $PythonExe"
}

Write-Host "[1/2] Reindexing facts into Chroma..." -ForegroundColor Cyan
& $PythonExe scripts/index_chroma.py `
    --source-root $SourceRoot `
    --chroma-path $ChromaPath `
    --collection $Collection `
    --reset

if ($LASTEXITCODE -ne 0) {
    throw "Indexing failed with exit code $LASTEXITCODE"
}

if (-not $SkipValidation) {
    Write-Host "[2/2] Running retrieval validation..." -ForegroundColor Cyan
    & $PythonExe scripts/validate_queries.py `
        --chroma-path $ChromaPath `
        --collection $Collection `
        --top-k $TopK

    if ($LASTEXITCODE -ne 0) {
        throw "Validation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Done. Facts are refreshed in Chroma." -ForegroundColor Green
