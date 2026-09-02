// 聊天控制器：会话列表 / 历史回放 / 发消息 / SSE 事件流 / 审批裁决。
// 所有服务端事件都汇聚成 ChatItem（一次 run = 一条 assistant 消息）。
import { useCallback, useEffect, useReducer, useRef } from 'react'
import { api, errText } from '../lib/api'
import { streamRunEvents } from '../lib/sse'
import type {
  ApprovalCase,
  ChatItem,
  ConversationMeta,
  RunRecord,
  RunStatus,
  StepEntry,
} from '../lib/types'
import { isRecord, normalizeResult, uuid } from '../lib/types'

interface State {
  conversations: ConversationMeta[]
  activeCid: string | null
  items: ChatItem[]
  banner: string | null
}

type Action =
  | { type: 'conversations'; list: ConversationMeta[] }
  | { type: 'select'; cid: string | null; items: ChatItem[] }
  | { type: 'append'; items: ChatItem[] }
  | { type: 'patch'; id: string; patch: Partial<ChatItem> }
  | { type: 'step'; id: string; step: StepEntry }
  | { type: 'banner'; text: string | null }

function reducer(state: State, a: Action): State {
  switch (a.type) {
    case 'conversations':
      return { ...state, conversations: a.list }
    case 'select':
      return { ...state, activeCid: a.cid, items: a.items }
    case 'append':
      return { ...state, items: [...state.items, ...a.items] }
    case 'patch':
      return {
        ...state,
        items: state.items.map((i) => (i.id === a.id ? { ...i, ...a.patch } : i)),
      }
    case 'step':
      return {
        ...state,
        items: state.items.map((i) => {
          if (i.id !== a.id) return i
          if (i.steps.some((s) => s.key === a.step.key)) return i // 按事件 id 去重（续传重放）
          return { ...i, steps: [...i.steps, a.step] }
        }),
      }
    case 'banner':
      return { ...state, banner: a.text }
  }
}

function parseJson(text: string): unknown {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

function historyStatus(s: string): RunStatus {
  switch (s) {
    case 'succeeded':
    case 'failed':
    case 'cancelled':
    case 'waiting_for_user':
    case 'pending':
      return s
    default:
      return 'running'
  }
}

function summarize(item: Pick<ChatItem, 'result' | 'error'>): string {
  if (item.error) return item.error
  if (item.result) {
    try {
      return `[${item.result.behavior}] ${JSON.stringify(item.result.answer)}`
    } catch {
      return `[${item.result.behavior}]`
    }
  }
  return '（运行中）'
}

export interface ChatController {
  conversations: ConversationMeta[]
  activeCid: string | null
  items: ChatItem[]
  banner: string | null
  busy: boolean
  dismissBanner: () => void
  refreshConversations: () => Promise<void>
  selectConversation: (cid: string) => Promise<void>
  newConversation: (model?: string) => Promise<void>
  send: (text: string) => Promise<void>
  decide: (itemId: string, decision: 'approved' | 'rejected') => Promise<void>
}

export function useChatController(): ChatController {
  const [state, dispatch] = useReducer(reducer, {
    conversations: [],
    activeCid: null,
    items: [],
    banner: null,
  })

  // itemId -> 流控制/续传游标（跨渲染保存）
  const streamsRef = useRef(new Map<string, AbortController>())
  const lastEventIdRef = useRef(new Map<string, string>())
  const selectSeqRef = useRef(0)
  const itemsRef = useRef<ChatItem[]>(state.items)
  useEffect(() => {
    itemsRef.current = state.items
  }, [state.items])

  const abortAllStreams = useCallback(() => {
    for (const ctrl of streamsRef.current.values()) ctrl.abort()
    streamsRef.current.clear()
  }, [])

  useEffect(() => abortAllStreams, [abortAllStreams])

  const refreshConversations = useCallback(async () => {
    try {
      dispatch({ type: 'conversations', list: await api.listConversations() })
    } catch (e) {
      dispatch({ type: 'banner', text: `会话列表加载失败：${errText(e)}` })
    }
  }, [])

  useEffect(() => {
    void refreshConversations()
  }, [refreshConversations])

  const attachApproval = useCallback(async (itemId: string, runId: string, caseRef: string | null) => {
    try {
      const list = await api.listApprovals()
      const found =
        list.find((a) => a.run_id === runId && a.status === 'pending') ??
        (caseRef ? list.find((a) => a.case_ref === caseRef) : undefined) ??
        list.find((a) => a.run_id === runId) ??
        null
      if (found) {
        dispatch({ type: 'patch', id: itemId, patch: { approval: found, caseRef: found.case_ref } })
      } else {
        dispatch({ type: 'banner', text: `未在 /approvals 中找到 run ${runId} 对应的审批单` })
      }
    } catch (e) {
      dispatch({ type: 'banner', text: `审批单加载失败：${errText(e)}` })
    }
  }, [])

  const openStream = useCallback(
    (itemId: string, runId: string, lastEventId?: string) => {
      streamsRef.current.get(itemId)?.abort()
      const ctrl = new AbortController()
      streamsRef.current.set(itemId, ctrl)

      streamRunEvents(runId, {
        lastEventId,
        signal: ctrl.signal,
        onConnectionChange: (s) => {
          dispatch({ type: 'patch', id: itemId, patch: { conn: s } })
        },
        onDone: () => {
          if (streamsRef.current.get(itemId) === ctrl) streamsRef.current.delete(itemId)
        },
        onEvent: (ev) => {
          if (ev.id !== undefined) lastEventIdRef.current.set(itemId, ev.id)
          const data = parseJson(ev.data)
          const d = isRecord(data) ? data : {}
          const stepKey = ev.id !== undefined ? `id:${ev.id}` : uuid()

          switch (ev.event) {
            case 'run.started': {
              const attempt = typeof d['attempt'] === 'number' ? d['attempt'] : 1
              // ★ 2026-08-20：档位改由**动作**推导（tier_policy）⇒ run 上通常没有声明值，
              //   这里此前显示「档位 ?」（我改推导时留下的）。没有声明就不提档位 ——
              //   真正的档位判定会在**审批卡**上带理由显示（那才是它有意义的地方）。
              const tier = typeof d['automation_tier'] === 'string' ? d['automation_tier'] : null
              dispatch({ type: 'patch', id: itemId, patch: { status: 'running' } })
              dispatch({
                type: 'step',
                id: itemId,
                step: {
                  key: stepKey,
                  kind: 'info',
                  ok: true,
                  text: tier
                    ? `开始执行（第 ${attempt} 次尝试 · 声明档位 ${tier}）`
                    : `开始执行（第 ${attempt} 次尝试）`,
                },
              })
              break
            }
            case 'model.thinking': {
              const text = typeof d['text'] === 'string' ? d['text'] : ''
              if (text) {
                dispatch({
                  type: 'step',
                  id: itemId,
                  step: { key: stepKey, kind: 'thinking', ok: true, text },
                })
              }
              break
            }
            case 'tool.result': {
              const ok = d['ok'] !== false
              const tool = typeof d['tool'] === 'string' ? d['tool'] : 'tool'
              const replayed = d['replayed'] === true ? '（重放）' : ''
              const err = typeof d['error'] === 'string' && d['error'] ? ` — ${d['error']}` : ''
              dispatch({
                type: 'step',
                id: itemId,
                step: {
                  key: stepKey,
                  kind: 'tool',
                  ok,
                  text: `${tool} ${ok ? '✓' : '✗'}${replayed}${err}`,
                },
              })
              break
            }
            case 'retrieval.result': {
              const tool = typeof d['tool'] === 'string' ? d['tool'] : 'retrieval'
              const hits = typeof d['hits'] === 'number' ? d['hits'] : '?'
              const latency = typeof d['latency_ms'] === 'number' ? `${d['latency_ms']}ms` : '?'
              const status =
                typeof d['status'] === 'string' && d['status'] !== 'ok' ? ` · ${d['status']}` : ''
              dispatch({
                type: 'step',
                id: itemId,
                step: {
                  key: stepKey,
                  kind: 'retrieval',
                  ok: true,
                  text: `${tool}（检索）· ${hits} 命中 · ${latency}${status}`,
                },
              })
              break
            }
            case 'run.degraded': {
              const reason = typeof d['reason'] === 'string' ? d['reason'] : '未知原因'
              dispatch({
                type: 'step',
                id: itemId,
                step: { key: stepKey, kind: 'degraded', ok: false, text: `已降级：${reason}` },
              })
              break
            }
            case 'run.completed': {
              const result = normalizeResult(data)
              dispatch({
                type: 'patch',
                id: itemId,
                patch: {
                  status: 'succeeded',
                  result,
                  error: null,
                  conn: null,
                  text: summarize({ result, error: null }),
                },
              })
              break
            }
            case 'run.failed':
            case 'run.cancelled': {
              const error =
                typeof d['error'] === 'string' && d['error']
                  ? d['error']
                  : ev.event === 'run.failed'
                    ? '运行失败'
                    : '运行已取消'
              dispatch({
                type: 'patch',
                id: itemId,
                patch: {
                  status: ev.event === 'run.failed' ? 'failed' : 'cancelled',
                  error,
                  conn: null,
                  text: error,
                },
              })
              break
            }
            case 'run.waiting_for_user': {
              const caseRef = typeof d['case_ref'] === 'string' ? d['case_ref'] : null
              const triggers = d['triggers']
              dispatch({
                type: 'patch',
                id: itemId,
                patch: { status: 'waiting_for_user', caseRef, conn: null, approvalDecision: null },
              })
              dispatch({
                type: 'step',
                id: itemId,
                step: {
                  key: stepKey,
                  kind: 'info',
                  ok: true,
                  text: `触发人工审批${triggers !== undefined && triggers !== null ? `（${JSON.stringify(triggers)}）` : ''}`,
                },
              })
              void attachApproval(itemId, runId, caseRef)
              break
            }
            default:
              break // 未知事件忽略
          }
        },
      })
    },
    [attachApproval],
  )

  const selectConversation = useCallback(
    async (cid: string) => {
      const seq = ++selectSeqRef.current
      abortAllStreams()
      dispatch({ type: 'select', cid, items: [] })
      try {
        const runs = await api.listMessages(cid)
        if (selectSeqRef.current !== seq) return // 已切走
        const items: ChatItem[] = []
        const live: { itemId: string; runId: string }[] = []
        const waiting: { itemId: string; runId: string }[] = []
        for (const run of runs) {
          items.push(userItemFromRun(run))
          const a = assistantItemFromRun(run)
          items.push(a)
          if (a.runId) {
            if (a.status === 'running' || a.status === 'pending') {
              live.push({ itemId: a.id, runId: a.runId })
            } else if (a.status === 'waiting_for_user') {
              waiting.push({ itemId: a.id, runId: a.runId })
            }
          }
        }
        dispatch({ type: 'select', cid, items })
        for (const l of live) openStream(l.itemId, l.runId)
        for (const w of waiting) void attachApproval(w.itemId, w.runId, null)
      } catch (e) {
        if (selectSeqRef.current !== seq) return
        dispatch({ type: 'banner', text: `历史加载失败：${errText(e)}` })
      }
    },
    [abortAllStreams, attachApproval, openStream],
  )

  const newConversation = useCallback(async (model?: string) => {
    try {
      const created = await api.createConversation(undefined, model)
      await refreshConversations()
      await selectConversation(created.conversation_id)
    } catch (e) {
      dispatch({ type: 'banner', text: `新建会话失败：${errText(e)}` })
    }
  }, [refreshConversations, selectConversation])

  const send = useCallback(
    async (text: string) => {
      const cid = state.activeCid
      if (!cid) return
      const userItem: ChatItem = {
        id: uuid(),
        role: 'user',
        runId: null,
        text,
        status: 'succeeded',
        steps: [],
      }
      const asstItem: ChatItem = {
        id: uuid(),
        role: 'assistant',
        runId: null,
        text: '（运行中）',
        status: 'pending',
        steps: [],
      }
      dispatch({ type: 'append', items: [userItem, asstItem] })
      try {
        // ★ 意图与档位都不再由前端给：菜单是全量 30 个（模型自选），
        //   档位由动作推导（tier_policy）—— 后端两个字段都已改成可选。
        const res = await api.sendMessage(cid, { user_message: text })
        dispatch({ type: 'patch', id: userItem.id, patch: { runId: res.run_id } })
        dispatch({ type: 'patch', id: asstItem.id, patch: { runId: res.run_id, status: 'running' } })
        openStream(asstItem.id, res.run_id)
        void refreshConversations()
      } catch (e) {
        dispatch({
          type: 'patch',
          id: asstItem.id,
          patch: { status: 'failed', error: `发送失败：${errText(e)}` },
        })
      }
    },
    [state.activeCid, openStream, refreshConversations],
  )

  const decide = useCallback(
    async (itemId: string, decision: 'approved' | 'rejected') => {
      const item = itemsRef.current.find((i) => i.id === itemId)
      if (!item || !item.caseRef || !item.runId) return
      dispatch({ type: 'patch', id: itemId, patch: { approvalDecision: decision } })
      try {
        await api.decideApproval(item.caseRef, decision)
        const approval: ApprovalCase | null = item.approval
          ? { ...item.approval, status: decision }
          : null
        dispatch({ type: 'patch', id: itemId, patch: { status: 'running', approval } })
        // 裁决后继续对同一 run 的 SSE，从上次 seq 续传直到下一个终态
        openStream(itemId, item.runId, lastEventIdRef.current.get(itemId))
      } catch (e) {
        dispatch({ type: 'patch', id: itemId, patch: { approvalDecision: null } })
        dispatch({ type: 'banner', text: `审批提交失败：${errText(e)}` })
      }
    },
    [openStream],
  )

  const dismissBanner = useCallback(() => dispatch({ type: 'banner', text: null }), [])

  const busy = state.items.some(
    (i) =>
      i.role === 'assistant' &&
      (i.status === 'pending' || i.status === 'running' || i.status === 'waiting_for_user'),
  )

  return {
    conversations: state.conversations,
    activeCid: state.activeCid,
    items: state.items,
    banner: state.banner,
    busy,
    dismissBanner,
    refreshConversations,
    selectConversation,
    newConversation,
    send,
    decide,
  }
}

function userItemFromRun(run: RunRecord): ChatItem {
  return {
    id: `${run.run_id}:user`,
    role: 'user',
    runId: run.run_id,
    text: run.user_message,
    status: 'succeeded',
    intent: run.intent ?? undefined,
    tier: run.automation_tier ?? undefined,
    steps: [],
    createdAt: run.created_at,
  }
}

function assistantItemFromRun(run: RunRecord): ChatItem {
  const status = historyStatus(run.status)
  const result = run.result ? normalizeResult(run.result) : null
  const error = run.error ?? null
  return {
    id: `${run.run_id}:assistant`,
    role: 'assistant',
    runId: run.run_id,
    text: summarize({ result, error }),
    status,
    steps: [],
    result,
    error,
    createdAt: run.created_at,
  }
}
