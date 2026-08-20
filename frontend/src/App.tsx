import { useState } from 'react'
import { getToken, setToken } from './lib/api'
import { useChatController } from './state/controller'
import { Sidebar } from './components/Sidebar'
import { ChatView } from './components/ChatView'

function TokenBox() {
  const [token, setTokenState] = useState(getToken())
  return (
    <label className="flex items-center gap-2 text-xs text-slate-500">
      Org Token
      <input
        value={token}
        onChange={(e) => {
          setTokenState(e.target.value)
          setToken(e.target.value)
        }}
        spellCheck={false}
        className="w-44 rounded-md border border-slate-300 bg-white px-2 py-1 font-mono text-xs text-slate-700 outline-none focus:border-blue-400"
      />
    </label>
  )
}

export default function App() {
  const controller = useChatController()
  const activeTitle =
    controller.conversations.find((c) => c.conversation_id === controller.activeCid)?.title ??
    (controller.activeCid ? `会话 ${controller.activeCid.slice(0, 8)}` : null)

  return (
    <div className="flex h-screen bg-white text-slate-900">
      <Sidebar
        conversations={controller.conversations}
        activeCid={controller.activeCid}
        onSelect={(cid) => void controller.selectConversation(cid)}
        onNew={() => void controller.newConversation()}
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-200 px-6 py-3">
          <div className="min-w-0">
            <h1 className="text-sm font-semibold">投放 Agent 控制台</h1>
            {activeTitle && <p className="truncate text-xs text-slate-400">{activeTitle}</p>}
          </div>
          <TokenBox />
        </header>

        {controller.banner && (
          <div className="flex items-center justify-between border-b border-rose-200 bg-rose-50 px-6 py-2 text-sm text-rose-700">
            <span className="truncate">{controller.banner}</span>
            <button
              type="button"
              onClick={controller.dismissBanner}
              className="ml-4 shrink-0 text-xs text-rose-500 hover:text-rose-700"
            >
              关闭
            </button>
          </div>
        )}

        <ChatView controller={controller} />
      </main>
    </div>
  )
}
