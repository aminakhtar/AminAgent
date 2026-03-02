# AminPersonalAgentApi

ASP.NET Core API wrapper around the internal Python RAG service.

## Architecture
- Public API: this ASP.NET Core service
- Internal RAG API: `http://127.0.0.1:8091`
- Local LLM server: `http://127.0.0.1:8080`
- Vector DB: Chroma path managed by Python service

## Endpoints
- `GET /health` -> API health + internal RAG health
- `POST /api/chat` -> forwards chat request to internal RAG `/chat`

## Run order
1. Start llama server on port `8080`
2. Start Python RAG service on port `8091`
3. Start this ASP.NET API

## Run this API
```powershell
cd AminPersonalAgentApi
dotnet run
```

## Test
```powershell
Invoke-RestMethod -Uri "http://localhost:5231/health" -Method Get
```

```powershell
$body = @{
  message = "What is Drink Tracker?"
  sessionId = "amin_api_test"
  historyTurns = 4
  topK = 3
  llmProvider = "openai-compatible"
  llmUrl = "http://127.0.0.1:8080"
  llmModel = "llama"
  debugPrompt = $false
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5231/api/chat" -Method Post -ContentType "application/json" -Body $body
```
