// assistant 回合：工具步骤行 → （审批卡）→ 终态结果卡
import type { Behavior, ChatItem, StepEntry } from '../lib/types'

const BEHAVIOR_STYLE: Record<Behavior, { label: string; card: string; badge: string }> = {
  answer: {
    label: '结论',
    card: 'border-slate-200 bg-white',
    badge: 'bg-slate-100 text-slate-700',
  },
  tool_call: {
    label: '执行完成',
    card: 'border-slate-200 bg-white',
    badge: 'bg-indigo-100 text-indigo-700',
  },
  defer: {
    label: '建议等待',
    card: 'border-amber-300 bg-amber-50',
    badge: 'bg-amber-100 text-amber-800',
  },
  clarify: {
    label: '需要澄清',
    card: 'border-sky-300 bg-sky-50',
    badge: 'bg-sky-100 text-sky-800',
  },
  reject: {
    label: '拒绝执行',
    card: 'border-rose-300 bg-rose-50',
    badge: 'bg-rose-100 text-rose-800',
  },
}

function stepClass(s: StepEntry): string {
  switch (s.kind) {
    case 'degraded':
      return 'text-amber-700'
    case 'tool':
      return s.ok ? 'text-slate-500' : 'text-rose-600'
    case 'retrieval':
      return 'text-slate-500'
    case 'info':
      return 'text-slate-400'
    case 'thinking':
      return 'text-neutral-500'
  }
}

function stepIcon(s: StepEntry): string {
  switch (s.kind) {
    case 'degraded':
      return '⚠'
    case 'tool':
      return s.ok ? '·' : '·'
    case 'retrieval':
      return '·'
    case 'info':
      return '›'
    case 'thinking':
      return '▸'
  }
}

function ValueView({ value }: { value: unknown }) {
  if (value === null || value === undefined) {
    return <span className="text-slate-400">—</span>
  }
  if (typeof value === 'string') {
    return <span className="whitespace-pre-wrap break-words">{value}</span>
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return <span className="font-mono">{String(value)}</span>
  }
  return (
    <pre className="mt-1 max-h-64 overflow-auto rounded bg-slate-900/5 p-2 font-mono text-xs leading-relaxed">
      {JSON.stringify(value, null, 2)}
    </pre>
  )
}

function AnswerKV({ answer }: { answer: Record<string, unknown> }) {
  const entries = Object.entries(answer)
  if (entries.length === 0) return <div className="text-sm text-slate-400">（无内容）</div>
  // ★ 2026-08-20：结论契约改成「机器字段 + 人话字段并列」（decider.DEFAULT_ANSWER_FIELDS）。
  //   `reply` 是给人读的那一句 ⇒ 当正文突出显示；其余键退为附属明细。
  const reply = typeof answer.reply === 'string' ? answer.reply : null
  const rest = entries.filter(([k]) => k !== 'reply')
  return (
    <dl className="space-y-2">
      {reply && (
        <div className="mb-1 text-[15px] leading-relaxed text-slate-900">{reply}</div>
      )}
      {rest.map(([k, v]) => (
        <div key={k}>
          <dt className="text-xs font-medium text-slate-500">{k}</dt>
          <dd className="text-sm text-slate-800">
            <ValueView value={v} />
          </dd>
        </div>
      ))}
    </dl>
  )
}

function ResultCard({ item }: { item: ChatItem }) {
  const result = item.result
  if (!result) return null
  const style = BEHAVIOR_STYLE[result.behavior]
  return (
    <div className={`rounded-xl border p-4 shadow-sm ${style.card}`}>
      <span
        className={`mb-2 inline-block rounded-full px-2 py-0.5 text-xs font-medium ${style.badge}`}
      >
        {style.label}
      </span>
      {result.behavior === 'defer' && (
        <p className="mb-2 text-sm text-amber-800">Agent 建议当前不执行操作，先等待。</p>
      )}
      {result.behavior === 'reject' && (
        <p className="mb-2 text-sm text-rose-800">Agent 拒绝执行该请求。</p>
      )}
      <AnswerKV answer={result.answer} />
    </div>
  )
}

function ErrorCard({ item }: { item: ChatItem }) {
  const cancelled = item.status === 'cancelled'
  return (
    <div className="rounded-xl border border-rose-300 bg-rose-50 p-4 shadow-sm">
      <span className="mb-2 inline-block rounded-full bg-rose-100 px-2 py-0.5 text-xs font-medium text-rose-800">
        {cancelled ? '已取消' : '运行失败'}
      </span>
      <p className="whitespace-pre-wrap break-words font-mono text-sm text-rose-800">
        {item.error ?? (cancelled ? '运行已取消' : '运行失败')}
      </p>
    </div>
  )
}

function ApprovalCard({
  item,
  onDecide,
}: {
  item: ChatItem
  onDecide: (itemId: string, decision: 'approved' | 'rejected') => void
}) {
  const a = item.approval
  const deciding = item.approvalDecision != null
  return (
    <div className="rounded-xl border border-violet-300 bg-violet-50 p-4 shadow-sm">
      <span className="mb-2 inline-block rounded-full bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-800">
        等待人工审批
      </span>
      {a ? (
        <div className="space-y-2">
          <div>
            <div className="text-xs font-medium text-slate-500">操作类型</div>
            <div className="font-mono text-sm text-slate-800">{a.action_type}</div>
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500">拟执行参数</div>
            <pre className="mt-1 max-h-64 overflow-auto rounded bg-slate-900/5 p-2 font-mono text-xs leading-relaxed">
              {JSON.stringify(a.proposed_params, null, 2)}
            </pre>
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500">理由</div>
            <div className="whitespace-pre-wrap text-sm text-slate-800">{a.rationale}</div>
          </div>
          <div>
            <div className="text-xs font-medium text-slate-500">触发原因</div>
            <div className="whitespace-pre-wrap text-sm text-slate-800">{a.trigger_reason}</div>
          </div>
          {a.evidence?.tier_reason && (
            <div>
              {/* ★ 档位不再由人选，而是由动作推导 ⇒ 界面要如实说明"为什么判成这一档" */}
              <div className="text-xs font-medium text-slate-500">
                档位判定{a.evidence.tier ? `（${a.evidence.tier} 档）` : ''}
              </div>
              <div className="whitespace-pre-wrap text-sm text-slate-800">
                {a.evidence.tier_reason}
              </div>
            </div>
          )}
          <div className="flex gap-2 pt-1">
            <button
              type="button"
              disabled={deciding}
              onClick={() => onDecide(item.id, 'approved')}
              className="rounded-lg bg-emerald-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
            >
              批准
            </button>
            <button
              type="button"
              disabled={deciding}
              onClick={() => onDecide(item.id, 'rejected')}
              className="rounded-lg bg-rose-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-rose-700 disabled:opacity-50"
            >
              拒绝
            </button>
            {deciding && <span className="self-center text-xs text-slate-500">提交中…</span>}
          </div>
        </div>
      ) : (
        <div className="text-sm text-slate-500">正在加载审批单…</div>
      )}
    </div>
  )
}

function RunningDots() {
  return (
    <div className="flex items-center gap-1.5 py-1 text-slate-400" aria-label="运行中">
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:0ms]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:150ms]" />
      <span className="h-2 w-2 animate-bounce rounded-full bg-slate-400 [animation-delay:300ms]" />
    </div>
  )
}

export function AssistantTurn({
  item,
  onDecide,
}: {
  item: ChatItem
  onDecide: (itemId: string, decision: 'approved' | 'rejected') => void
}) {
  const running = item.status === 'pending' || item.status === 'running'
  return (
    <div className="mb-6 flex justify-start">
      <div className="w-full max-w-[46rem]">
        {item.steps.length > 0 && (
          <ol className="mb-2 space-y-0.5">
            {item.steps.map((s) =>
              s.kind === 'thinking' ? (
                <li key={s.key} className="text-xs">
                  <details className="group rounded-md bg-neutral-50 ring-1 ring-neutral-200">
                    <summary className="flex cursor-pointer select-none items-center gap-1.5 px-2 py-1 text-neutral-500 hover:text-neutral-700">
                      <span className="inline-block w-3 text-center transition-transform group-open:rotate-90">▸</span>
                      <span className="italic">思考过程（{s.text.length} 字）</span>
                    </summary>
                    <pre className="max-h-64 overflow-y-auto whitespace-pre-wrap px-3 pb-2 pt-1 font-mono text-[11px] leading-relaxed text-neutral-600">
                      {s.text}
                    </pre>
                  </details>
                </li>
              ) : (
                <li key={s.key} className={`font-mono text-xs ${stepClass(s)}`}>
                  <span className="mr-1.5 inline-block w-3 text-center">{stepIcon(s)}</span>
                  {s.text}
                </li>
              ),
            )}
          </ol>
        )}

        {item.conn === 'reconnecting' && (
          <div className="mb-2 inline-flex items-center gap-1.5 rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-700 ring-1 ring-amber-200">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-500" />
            事件流断开，正在重连…
          </div>
        )}

        {running && <RunningDots />}

        {item.status === 'waiting_for_user' && <ApprovalCard item={item} onDecide={onDecide} />}

        {item.approvalDecision != null && item.status !== 'waiting_for_user' && (
          <div
            className={`mb-2 text-xs ${item.approvalDecision === 'approved' ? 'text-emerald-600' : 'text-rose-600'}`}
          >
            审批已{item.approvalDecision === 'approved' ? '批准' : '拒绝'}
            {running ? '，继续执行…' : ''}
          </div>
        )}

        {item.status === 'succeeded' && <ResultCard item={item} />}
        {(item.status === 'failed' || item.status === 'cancelled') && <ErrorCard item={item} />}
      </div>
    </div>
  )
}
