// 主聊天区：assistant-ui（@assistant-ui/react）的 ExternalStoreRuntime 接自家 controller，
// Thread/Composer 用无样式 primitives + Tailwind 自排版。
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AssistantRuntimeProvider,
  ComposerPrimitive,
  ThreadPrimitive,
  useExternalStoreRuntime,
  type AppendMessage,
  type ThreadMessageLike,
} from '@assistant-ui/react'
import type { ChatController } from '../state/controller'
import type { ChatItem } from '../lib/types'
import { INTENT_OPTIONS, TIER_OPTIONS } from '../lib/types'
import { AssistantTurn } from './AssistantTurn'
import { UserBubble } from './UserBubble'

// assistant-ui 需要的最小消息形状；实际渲染完全走我们自己的 ChatItem
function convertMessage(item: ChatItem): ThreadMessageLike {
  return {
    id: item.id,
    role: item.role,
    content: [{ type: 'text', text: item.text || '…' }],
  }
}

function appendMessageText(m: AppendMessage): string {
  return m.content
    .map((p) => (p.type === 'text' ? p.text : ''))
    .join('\n')
    .trim()
}

export function ChatView({ controller }: { controller: ChatController }) {
  const { items, activeCid, busy, send, decide } = controller

  const [intent, setIntent] = useState<string>(INTENT_OPTIONS[0].value)
  const [tier, setTier] = useState<string>('C')
  // onNew 的闭包要读到最新选择，走 ref（effect 同步，事件总在渲染后发生）
  const intentRef = useRef(intent)
  const tierRef = useRef(tier)
  useEffect(() => {
    intentRef.current = intent
    tierRef.current = tier
  }, [intent, tier])

  const itemsById = useMemo(() => new Map(items.map((i) => [i.id, i])), [items])

  const onNew = useCallback(
    async (m: AppendMessage) => {
      const text = appendMessageText(m)
      if (!text) return
      await send(text, intentRef.current, tierRef.current)
    },
    [send],
  )

  const runtime = useExternalStoreRuntime<ChatItem>({
    messages: items,
    isRunning: busy,
    isSendDisabled: busy || !activeCid,
    onNew,
    convertMessage,
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Root className="flex min-h-0 flex-1 flex-col">
        <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto max-w-3xl">
            {!activeCid ? (
              <div className="py-24 text-center text-sm text-slate-400">
                从左侧选择一个会话，或新建会话开始。
              </div>
            ) : (
              <ThreadPrimitive.Empty>
                <div className="py-24 text-center text-sm text-slate-400">
                  还没有消息，输入第一条指令试试。
                </div>
              </ThreadPrimitive.Empty>
            )}
            <ThreadPrimitive.Messages>
              {({ message }) => {
                const item = itemsById.get(message.id)
                if (!item) return null
                return item.role === 'user' ? (
                  <UserBubble item={item} />
                ) : (
                  <AssistantTurn item={item} onDecide={decide} />
                )
              }}
            </ThreadPrimitive.Messages>
          </div>
        </ThreadPrimitive.Viewport>

        <div className="border-t border-slate-200 bg-white px-6 py-3">
          <div className="mx-auto max-w-3xl">
            <div className="mb-2 flex items-center gap-3 text-xs text-slate-500">
              <label className="flex items-center gap-1.5">
                意图
                <select
                  value={intent}
                  onChange={(e) => setIntent(e.target.value)}
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
                >
                  {INTENT_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-1.5">
                档位
                <select
                  value={tier}
                  onChange={(e) => setTier(e.target.value)}
                  className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-700"
                >
                  {TIER_OPTIONS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </label>
              {busy && <span className="text-amber-600">当前运行未结束，发送已禁用…</span>}
              {!activeCid && <span>请先选择或新建会话</span>}
            </div>
            <ComposerPrimitive.Root className="flex items-end gap-2 rounded-xl border border-slate-300 bg-white p-2 focus-within:border-blue-400">
              <ComposerPrimitive.Input
                rows={1}
                placeholder="输入消息，Enter 发送…"
                className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-slate-800 outline-none placeholder:text-slate-400"
              />
              <ComposerPrimitive.Send className="rounded-lg bg-blue-600 px-4 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40">
                发送
              </ComposerPrimitive.Send>
            </ComposerPrimitive.Root>
          </div>
        </div>
      </ThreadPrimitive.Root>
    </AssistantRuntimeProvider>
  )
}
