import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription, timeout } from 'rxjs';

import { environment } from '../environments/environment';

interface ChatRequest {
  message: string;
  sessionId: string;
  historyTurns: number;
  topK: number;
  llmProvider: string;
  llmUrl: string;
  llmModel: string;
  temperature: number;
  factsOnly: boolean;
  debugPrompt: boolean;
}

interface SourceItem {
  fact: string;
  source_file: string;
  section: string;
  title: string;
  date_range: string;
  chunk_index: number;
  distance: number;
}

interface ChatResponse {
  answer: string;
  session_id: string;
  used_fallback: boolean;
  sources: SourceItem[];
  latency_ms: number;
}

interface UiMessage {
  role: 'user' | 'assistant';
  text: string;
  sources?: SourceItem[];
  fallback?: boolean;
  latencyMs?: number;
}

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = 'Amin Personal Agent';

  apiBaseUrl = environment.apiBaseUrl;
  directRagBaseUrl = 'http://127.0.0.1:8091';
  useDirectRagApi = true;
  messageInput = '';
  sessionId = 'amin_web_session';
  historyTurns = 2;
  topK = 2;
  requestTimeoutMs = 45000;
  loading = false;
  error = '';
  private activeRequest?: Subscription;

  messages: UiMessage[] = [
    {
      role: 'assistant',
      text: 'Ask me anything about your background and projects. I will answer using your RAG facts.'
    }
  ];

  constructor(private readonly http: HttpClient) {}

  sendMessage(): void {
    const text = this.messageInput.trim();
    if (!text || this.loading) {
      return;
    }

    this.error = '';
    this.messages.push({ role: 'user', text });

    const payload: ChatRequest = {
      message: text,
      sessionId: this.sessionId.trim() || 'amin_web_session',
      historyTurns: Math.max(0, Math.min(this.historyTurns, 6)),
      topK: Math.max(1, Math.min(this.topK, 6)),
      llmProvider: 'openai-compatible',
      llmUrl: 'http://127.0.0.1:8080',
      llmModel: 'llama',
      temperature: 0.2,
      factsOnly: false,
      debugPrompt: false
    };

    this.loading = true;
    this.messageInput = '';

    const primaryEndpoint = this.useDirectRagApi ? `${this.directRagBaseUrl}/chat` : `${this.apiBaseUrl}/api/chat`;
    const fallbackEndpoint = this.useDirectRagApi ? `${this.apiBaseUrl}/api/chat` : `${this.directRagBaseUrl}/chat`;
    let usedRetry = false;

    const execute = (endpoint: string): void => {
      this.activeRequest = this.http
        .post<ChatResponse>(endpoint, payload)
        .pipe(timeout({ first: this.requestTimeoutMs }))
        .subscribe({
          next: (response) => {
            this.messages.push({
              role: 'assistant',
              text: response.answer,
              sources: response.sources,
              fallback: response.used_fallback,
              latencyMs: response.latency_ms
            });
            this.loading = false;
          },
          error: (err) => {
            if (!usedRetry && this.shouldRetryAlternate(err)) {
              usedRetry = true;
              execute(fallbackEndpoint);
              return;
            }

            if (err?.name === 'TimeoutError') {
              this.error = `Request timed out after ${Math.round(this.requestTimeoutMs / 1000)}s. Try shorter question, lower history/topK, or switch direct/API mode.`;
            } else {
              this.error = err?.error?.detail || err?.error?.error || err?.message || 'Request failed. Check API and RAG services.';
            }
            this.loading = false;
          }
        });
    };

    execute(primaryEndpoint);
  }

  private shouldRetryAlternate(err: unknown): boolean {
    const candidate = err as { name?: string; status?: number };
    if (candidate?.name === 'TimeoutError') {
      return true;
    }

    const status = Number(candidate?.status ?? 0);
    return status === 0 || status >= 500;
  }

  cancelRequest(): void {
    if (!this.loading || !this.activeRequest) {
      return;
    }
    this.activeRequest.unsubscribe();
    this.loading = false;
    this.error = 'Request cancelled.';
  }
}
