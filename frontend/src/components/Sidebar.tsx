import { useState } from 'react'
import type { ConversationMeta, ModelTag } from '../lib/types'

const MODEL_LABEL: Record<string, string> = { rl: 'RL', sft: 'SFT', base: 'Base' }
const MODEL_BADGE: Record<string, string> = {
  rl: 'bg-emerald-100 text-emerald-700',
  sft: 'bg-sky-100 text-sky-700',
  base: 'bg-slate-200 text-slate-600',
}

function formatTime(iso: string | null): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('zh-CN', { hour12: false })
}

export function Sidebar({
  conversations,
  activeCid,
  onSelect,
  onNew,
}: {
  conversations: ConversationMeta[]
  activeCid: string | null
  onSelect: (cid: string) => void
  onNew: (model: ModelTag) => void
}) {
  const [model, setModel] = useState<ModelTag>('rl')
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="space-y-2 p-3">
        {/* dev mode：新会话的模型选择（创建即锁定；已有会话不可改） */}
        <div className="flex rounded-lg bg-slate-200 p-0.5 text-xs">
          {(['rl', 'sft', 'base'] as ModelTag[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setModel(m)}
              className={`flex-1 rounded-md px-2 py-1 font-medium transition-colors ${
                model === m ? 'bg-white text-slate-800 shadow-sm' : 'text-slate-500 hover:text-slate-700'
              }`}
              title={m === 'rl' ? 'RL 最优点（v14.5 s12）' : m === 'sft' ? 'SFT 最佳（v14.5-e3）' : '裸底座 Qwen3-4B'}
            >
              {MODEL_LABEL[m]}
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => onNew(model)}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          ＋ 新建会话（{MODEL_LABEL[model]}）
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 pb-3">
        {conversations.length === 0 && (
          <p className="px-2 py-4 text-center text-xs text-slate-400">暂无会话</p>
        )}
        <ul className="space-y-1">
          {conversations.map((c) => {
            const active = c.conversation_id === activeCid
            return (
              <li key={c.conversation_id}>
                <button
                  type="button"
                  onClick={() => onSelect(c.conversation_id)}
                  className={`w-full rounded-lg px-3 py-2 text-left transition-colors ${
                    active ? 'bg-blue-100 text-blue-900' : 'text-slate-700 hover:bg-slate-200/60'
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="min-w-0 truncate text-sm font-medium">
                      {c.title || `会话 ${c.conversation_id.slice(0, 8)}`}
                    </span>
                    <span className={`shrink-0 rounded px-1 py-px text-[10px] font-semibold ${MODEL_BADGE[c.model ?? 'rl']}`}>
                      {MODEL_LABEL[c.model ?? 'rl']}
                    </span>
                  </div>
                  <div className="mt-0.5 flex justify-between text-[11px] text-slate-400">
                    <span>{c.runs} 次运行</span>
                    <span>{formatTime(c.last_activity)}</span>
                  </div>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>
    </aside>
  )
}
