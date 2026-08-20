// 同源 REST 客户端。页面挂在 /app 下，API 在根路径（绝对路径即可同源命中）。
import type {
  ApprovalCase,
  ConversationMeta,
  RunRecord,
  SendMessageResponse,
} from './types'
import { uuid } from './types'

const TOKEN_KEY = 'org_token'
const DEFAULT_TOKEN = 'dev-token-demo'

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) || DEFAULT_TOKEN
  } catch {
    return DEFAULT_TOKEN
  }
}

export function setToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_KEY, token)
  } catch {
    // localStorage 不可用时静默忽略（仅影响持久化）
  }
}

export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function errText(e: unknown): string {
  if (e instanceof Error) return e.message
  return String(e)
}

async function request<T>(
  path: string,
  init: { method?: string; body?: unknown; headers?: Record<string, string> } = {},
): Promise<T> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${getToken()}`,
    ...init.headers,
  }
  let body: string | undefined
  if (init.body !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(init.body)
  }
  const res = await fetch(path, { method: init.method ?? 'GET', headers, body })
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.text()).slice(0, 300)
    } catch {
      // 忽略 body 读取失败
    }
    throw new ApiError(res.status, `HTTP ${res.status}${detail ? `：${detail}` : ''}`)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  listConversations(): Promise<ConversationMeta[]> {
    return request<ConversationMeta[]>('/conversations')
  },

  createConversation(title?: string): Promise<{ conversation_id: string; title: string | null }> {
    return request('/conversations', {
      method: 'POST',
      body: title === undefined ? {} : { title },
    })
  },

  listMessages(cid: string): Promise<RunRecord[]> {
    return request<RunRecord[]>(`/conversations/${encodeURIComponent(cid)}/messages`)
  },

  sendMessage(
    cid: string,
    body: { user_message: string; intent: string; automation_tier: string },
  ): Promise<SendMessageResponse> {
    return request<SendMessageResponse>(`/conversations/${encodeURIComponent(cid)}/messages`, {
      method: 'POST',
      body,
      headers: { 'Idempotency-Key': uuid() },
    })
  },

  listApprovals(): Promise<ApprovalCase[]> {
    return request<ApprovalCase[]>('/approvals')
  },

  decideApproval(caseRef: string, decision: 'approved' | 'rejected'): Promise<unknown> {
    return request(`/approvals/${encodeURIComponent(caseRef)}`, {
      method: 'POST',
      body: { decision, reviewer_id: 'console' },
    })
  },
}
