import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ChangeDetectorRef, Component, NgZone } from '@angular/core';
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
  personaOnly: boolean;
  debugPrompt: boolean;
}

interface DirectChatRequest {
  message: string;
  session_id: string;
  history_turns: number;
  top_k: number;
  llm_provider: string;
  llm_url: string;
  llm_model: string;
  temperature: number;
  facts_only: boolean;
  persona_only: boolean;
  debug_prompt: boolean;
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
  protected readonly suggestedQuestions = [
    'What leadership impact did I deliver?',
    'What is Drink Tracker and why did I build it?',
    'What technologies do I use the most?',
    'What achievements best represent my work?',
    'Summarize my recent experience in one paragraph.',
    'What kinds of projects do I enjoy building?'
  ];

  apiBaseUrl = environment.apiBaseUrl;
  directRagBaseUrl = 'http://127.0.0.1:8090';
  useDirectRagApi = true;
  personaOnly = true;
  messageInput = '';
  sessionId = 'amin_web_session';
  historyTurns = 2;
  topK = 2;
  requestTimeoutMs = 45000;
  loading = false;
  error = '';
  private activeRequest?: Subscription;
  private questionHistory: string[] = [];
  private historyCursor = -1;
  private historyDraft = '';

  messages: UiMessage[] = [
    {
      role: 'assistant',
      text: 'Ask me anything about your background and projects. I will answer using your RAG facts.'
    }
  ];

  constructor(
    private readonly http: HttpClient,
    private readonly zone: NgZone,
    private readonly cdr: ChangeDetectorRef,
  ) {}

  private updateUiSafely(update: () => void): void {
    this.zone.run(() => {
      update();
      this.cdr.detectChanges();
    });
  }

  useSuggestedQuestion(question: string): void {
    if (this.loading) {
      return;
    }

    this.messageInput = question;
    this.resetHistoryNavigation();
  }

  onComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.sendMessage();
      return;
    }

    const target = event.target as HTMLTextAreaElement | null;
    if (!target || event.altKey || event.ctrlKey || event.metaKey) {
      return;
    }

    if (event.key === 'ArrowUp') {
      const atStart = target.selectionStart === 0 && target.selectionEnd === 0;
      if (atStart) {
        event.preventDefault();
        this.navigateHistory(-1, target);
      }
      return;
    }

    if (event.key === 'ArrowDown') {
      const atEnd = target.selectionStart === target.value.length && target.selectionEnd === target.value.length;
      if (atEnd) {
        event.preventDefault();
        this.navigateHistory(1, target);
      }
    }
  }

  sendMessage(): void {
    const text = this.messageInput.trim();
    if (!text || this.loading) {
      return;
    }

    if (this.questionHistory.length === 0 || this.questionHistory[this.questionHistory.length - 1] !== text) {
      this.questionHistory.push(text);
    }
    this.resetHistoryNavigation();

    this.error = '';
    this.messages.push({ role: 'user', text });

    const sessionId = this.sessionId.trim() || 'amin_web_session';
    const historyTurns = Math.max(0, Math.min(this.historyTurns, 6));
    const topK = Math.max(1, Math.min(this.topK, 6));

    const apiPayload: ChatRequest = {
      message: text,
      sessionId,
      historyTurns,
      topK,
      llmProvider: 'openai-compatible',
      llmUrl: 'http://127.0.0.1:8080',
      llmModel: 'llama',
      temperature: 0.2,
      factsOnly: false,
      personaOnly: this.personaOnly,
      debugPrompt: false
    };

    const directPayload: DirectChatRequest = {
      message: text,
      session_id: sessionId,
      history_turns: historyTurns,
      top_k: topK,
      llm_provider: 'openai-compatible',
      llm_url: 'http://127.0.0.1:8080',
      llm_model: 'llama',
      temperature: 0.2,
      facts_only: false,
      persona_only: this.personaOnly,
      debug_prompt: false
    };

    this.loading = true;
    this.messageInput = '';

    const primaryUsesDirect = this.useDirectRagApi;
    const primaryEndpoint = primaryUsesDirect ? `${this.directRagBaseUrl}/chat` : `${this.apiBaseUrl}/api/chat`;
    const fallbackEndpoint = primaryUsesDirect ? `${this.apiBaseUrl}/api/chat` : `${this.directRagBaseUrl}/chat`;
    let usedRetry = false;

    const execute = (endpoint: string, body: ChatRequest | DirectChatRequest): void => {
      this.activeRequest = this.http
        .post<ChatResponse>(endpoint, body)
        .pipe(timeout({ first: this.requestTimeoutMs }))
        .subscribe({
          next: (response) => {
            this.updateUiSafely(() => {
              this.messages.push({
                role: 'assistant',
                text: response.answer,
                sources: response.sources,
                fallback: response.used_fallback,
                latencyMs: response.latency_ms
              });
              this.loading = false;
            });
          },
          error: (err) => {
            if (!usedRetry && this.shouldRetryAlternate(err)) {
              usedRetry = true;
              execute(fallbackEndpoint, primaryUsesDirect ? apiPayload : directPayload);
              return;
            }

            this.updateUiSafely(() => {
              if (err?.name === 'TimeoutError') {
                this.error = `Request timed out after ${Math.round(this.requestTimeoutMs / 1000)}s. Try shorter question, lower history/topK, or switch direct/API mode.`;
              } else {
                this.error = err?.error?.detail || err?.error?.error || err?.message || 'Request failed. Check API and RAG services.';
              }
              this.loading = false;
            });
          }
        });
    };

    execute(primaryEndpoint, primaryUsesDirect ? directPayload : apiPayload);
  }

  private navigateHistory(direction: -1 | 1, textarea: HTMLTextAreaElement): void {
    if (this.questionHistory.length === 0) {
      return;
    }

    if (this.historyCursor === -1) {
      this.historyCursor = this.questionHistory.length;
      this.historyDraft = this.messageInput;
    }

    const nextCursor = this.historyCursor + direction;
    if (nextCursor < 0 || nextCursor > this.questionHistory.length) {
      return;
    }

    this.historyCursor = nextCursor;
    this.messageInput = nextCursor === this.questionHistory.length
      ? this.historyDraft
      : this.questionHistory[nextCursor];

    setTimeout(() => {
      const pos = this.messageInput.length;
      textarea.setSelectionRange(pos, pos);
    });
  }

  private resetHistoryNavigation(): void {
    this.historyCursor = -1;
    this.historyDraft = '';
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
    this.updateUiSafely(() => {
      this.loading = false;
      this.error = 'Request cancelled.';
    });
  }
}
