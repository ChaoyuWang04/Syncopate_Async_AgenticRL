import type { ConversationMeta } from '../lib/types'

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
  onNew: () => void
}) {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-200 bg-slate-50">
      <div className="p-3">
        <button
          type="button"
          onClick={onNew}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700"
        >
          ＋ 新建会话
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
                  <div className="truncate text-sm font-medium">
                    {c.title || `会话 ${c.conversation_id.slice(0, 8)}`}
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
