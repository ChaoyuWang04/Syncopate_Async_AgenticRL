// 后端契约类型 + 前端本地视图模型

export type Behavior = 'tool_call' | 'answer' | 'defer' | 'clarify' | 'reject'

export const BEHAVIOR_VALUES: readonly Behavior[] = [
  'tool_call',
  'answer',
  'defer',
  'clarify',
  'reject',
]

export interface RunResult {
  behavior: Behavior
  answer: Record<string, unknown>
}

export interface ConversationMeta {
  conversation_id: string
  title: string | null
  runs: number
  last_activity: string | null
}

/** GET /conversations/{cid}/messages 的历史条目 */
export interface RunRecord {
  run_id: string
  user_message: string
  status: string
  intent: string | null
  automation_tier: string | null
  result: { behavior?: string; answer?: unknown } | null
  error: string | null
  created_at: string | null
}

export interface SendMessageResponse {
  run_id: string
  status: string
  [k: string]: unknown
}

export interface ApprovalCase {
  case_ref: string
  run_id: string
  action_type: string
  proposed_params: unknown
  rationale: string
  trigger_reason: string
  status: string
  // 档位判定的证据（tier / tier_reason / triggers）—— 人看的是证据不是结论
  evidence?: { tier?: string; tier_reason?: string; triggers?: string[] } | null
}

// ---------------- 前端视图模型 ----------------

export type RunStatus =
  | 'pending'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'
  | 'waiting_for_user'

export type ConnState = 'connected' | 'reconnecting'

export interface StepEntry {
  key: string
  kind: 'info' | 'tool' | 'retrieval' | 'degraded' | 'thinking'
  text: string
  ok: boolean
}

/** 线程里的一条消息：user 气泡或 assistant 回合（一次 run） */
export interface ChatItem {
  id: string
  role: 'user' | 'assistant'
  runId: string | null
  text: string
  status: RunStatus
  intent?: string
  tier?: string
  steps: StepEntry[]
  result?: RunResult | null
  error?: string | null
  caseRef?: string | null
  approval?: ApprovalCase | null
  approvalDecision?: 'approved' | 'rejected' | null
  conn?: ConnState | null
  createdAt?: string | null
}

export const INTENT_OPTIONS = [
  { value: 'I01', label: 'I01 查指标' },
  { value: 'I07', label: 'I07 归因' },
  { value: 'I09', label: 'I09 扩量' },
  { value: 'I11', label: 'I11 素材' },
] as const

export const TIER_OPTIONS = ['A', 'B', 'C', 'D'] as const

export function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v)
}

export function isBehavior(v: unknown): v is Behavior {
  return typeof v === 'string' && (BEHAVIOR_VALUES as readonly string[]).includes(v)
}

/** 把终态/历史里的 result 归一成 {behavior, answer} */
export function normalizeResult(data: unknown): RunResult {
  if (!isRecord(data)) return { behavior: 'answer', answer: { value: data ?? null } }
  const behavior = isBehavior(data['behavior']) ? data['behavior'] : 'answer'
  const rawAnswer = data['answer']
  let answer: Record<string, unknown>
  if (isRecord(rawAnswer)) {
    answer = rawAnswer
  } else if (rawAnswer !== undefined) {
    answer = { answer: rawAnswer }
  } else {
    answer = {}
    for (const [k, v] of Object.entries(data)) {
      if (k !== 'behavior') answer[k] = v
    }
  }
  return { behavior, answer }
}

export function uuid(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
}
