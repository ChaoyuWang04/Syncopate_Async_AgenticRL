import type { ChatItem } from '../lib/types'
import { INTENT_OPTIONS } from '../lib/types'

function intentLabel(value: string | undefined): string | null {
  if (!value) return null
  const found = INTENT_OPTIONS.find((o) => o.value === value)
  return found ? found.label : value
}

export function UserBubble({ item }: { item: ChatItem }) {
  const intent = intentLabel(item.intent)
  return (
    <div className="mb-6 flex flex-col items-end">
      <div className="max-w-[40rem] rounded-2xl rounded-br-sm bg-blue-600 px-4 py-2.5 text-sm text-white shadow-sm">
        <span className="whitespace-pre-wrap break-words">{item.text}</span>
      </div>
      {(intent || item.tier) && (
        <div className="mt-1 flex gap-1.5 text-[11px] text-slate-400">
          {intent && <span className="rounded bg-slate-100 px-1.5 py-0.5">{intent}</span>}
          {item.tier && <span className="rounded bg-slate-100 px-1.5 py-0.5">档位 {item.tier}</span>}
        </div>
      )}
    </div>
  )
}
