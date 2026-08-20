// fetch + ReadableStream 手解 SSE。
// EventSource 用不了（带不了 Authorization 头），所以自己解析
// `id:` / `event:` / `data:` 行、空行分隔的事件，忽略 `:` 注释（keepalive）。
// 断线自动重连，并带 Last-Event-ID 续传；收到终态事件即关流。
import { getToken } from './api'

export interface SseEvent {
  id?: string
  event: string
  data: string
}

const TERMINAL_EVENTS = new Set([
  'run.succeeded',
  'run.failed',
  'run.cancelled',
  'run.waiting_for_user',
])

export function isTerminalEvent(name: string): boolean {
  return TERMINAL_EVENTS.has(name)
}

export interface StreamRunOptions {
  lastEventId?: string | undefined
  signal: AbortSignal
  onEvent: (ev: SseEvent) => void
  onConnectionChange?: (state: 'connected' | 'reconnecting') => void
  /** 流正常结束（收到终态）或被 abort 后调用一次 */
  onDone?: () => void
}

/** 消费 /runs/{runId}/events，直到终态事件或外部 abort。内部自动重连。 */
export function streamRunEvents(runId: string, opts: StreamRunOptions): void {
  void runLoop(runId, opts).finally(() => {
    opts.onDone?.()
  })
}

async function runLoop(runId: string, opts: StreamRunOptions): Promise<void> {
  const { signal } = opts
  let lastEventId = opts.lastEventId
  let attempt = 0

  while (!signal.aborted) {
    try {
      const headers: Record<string, string> = {
        Authorization: `Bearer ${getToken()}`,
        Accept: 'text/event-stream',
      }
      if (lastEventId !== undefined) headers['Last-Event-ID'] = lastEventId

      const res = await fetch(`/runs/${encodeURIComponent(runId)}/events`, { headers, signal })
      if (!res.ok || !res.body) {
        throw new Error(`SSE HTTP ${res.status}`)
      }
      opts.onConnectionChange?.('connected')
      attempt = 0

      const gotTerminal = await readSseStream(res.body, (ev) => {
        if (ev.id !== undefined) lastEventId = ev.id
        opts.onEvent(ev)
        return isTerminalEvent(ev.event)
      })
      if (gotTerminal) return // 终态：关流，不再重连
      // 流被服务端关闭但没到终态 → 走重连
    } catch {
      if (signal.aborted) return
    }

    if (signal.aborted) return
    opts.onConnectionChange?.('reconnecting')
    attempt += 1
    const delay = Math.min(10_000, 1000 * 2 ** Math.min(attempt - 1, 3))
    const aborted = await sleep(delay, signal)
    if (aborted) return
  }
}

/** 逐行解析 SSE 流；onEvent 返回 true 表示终态，停止读取。返回是否收到终态。 */
async function readSseStream(
  body: ReadableStream<Uint8Array>,
  onEvent: (ev: SseEvent) => boolean,
): Promise<boolean> {
  const reader = body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buf = ''
  let dataLines: string[] = []
  let eventName = ''
  let eventId: string | undefined

  const dispatch = (): boolean => {
    if (dataLines.length === 0 && eventName === '' && eventId === undefined) return false
    const ev: SseEvent = {
      event: eventName || 'message',
      data: dataLines.join('\n'),
    }
    if (eventId !== undefined) ev.id = eventId
    dataLines = []
    eventName = ''
    eventId = undefined
    return onEvent(ev)
  }

  const handleLine = (line: string): boolean => {
    if (line === '') return dispatch()
    if (line.startsWith(':')) return false // 注释行（如 ": keepalive"）忽略
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    let value = colon === -1 ? '' : line.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'data') dataLines.push(value)
    else if (field === 'event') eventName = value
    else if (field === 'id') eventId = value
    return false
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      let nl: number
      while ((nl = buf.indexOf('\n')) !== -1) {
        let line = buf.slice(0, nl)
        buf = buf.slice(nl + 1)
        if (line.endsWith('\r')) line = line.slice(0, -1)
        if (handleLine(line)) return true
      }
    }
    // 流结束：冲掉残余行与未派发的事件
    buf += decoder.decode()
    let tail = buf
    if (tail.endsWith('\r')) tail = tail.slice(0, -1)
    if (tail !== '' && handleLine(tail)) return true
    if (dispatch()) return true
    return false
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // reader 可能已因 abort 释放
    }
  }
}

/** 返回是否因 abort 提前醒来 */
function sleep(ms: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve(true)
      return
    }
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort)
      resolve(false)
    }, ms)
    const onAbort = () => {
      clearTimeout(timer)
      resolve(true)
    }
    signal.addEventListener('abort', onAbort, { once: true })
  })
}
