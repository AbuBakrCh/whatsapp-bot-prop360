import React, { useRef, useEffect, useState } from 'react'
import { Send, MessagesSquare } from 'lucide-react'
import { getClientConfig, toggleClientBot } from '../api'

export default function ChatWindow({ selected, messages = [], onSend }) {
  const bottomRef = useRef(null)
  const [text, setText] = useState('')
  const [botEnabledForClient, setBotEnabledForClient] = useState(true)

  // Scroll to bottom whenever messages or selection changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, selected])

  // Fetch bot config when a new chat is selected
  useEffect(() => {
    if (!selected) return
    const fetchConfig = async () => {
      try {
        const data = await getClientConfig(selected)
        setBotEnabledForClient(data.botEnabled !== false) // default true
      } catch (err) {
        console.error("Failed to fetch client config:", err)
        setBotEnabledForClient(true)
      }
    }
    fetchConfig()
  }, [selected])

  const handleSend = () => {
    if (text.trim()) {
      onSend(text)
      setText('')
    }
  }

  const handleToggle = async (enabled) => {
    setBotEnabledForClient(enabled)
    try {
      await toggleClientBot(selected, enabled)
    } catch (err) {
      console.error("Failed to update bot toggle:", err)
    }
  }

  if (!selected) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-slate-100 p-4">
        <MessagesSquare className="w-20 h-20 text-slate-300 mb-4" />
        <div className="text-slate-500 text-lg">No conversation selected</div>
        <div className="text-slate-400 text-sm">
          Select a contact to start chatting
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full bg-slate-100">
      {/* Chat Header */}
      <div className="p-4 bg-white sticky top-0 z-10 border-b border-slate-200 flex items-center gap-4 shadow-sm">
        <div className="w-11 h-11 rounded-full bg-gradient-to-br from-green-500 to-green-600 flex items-center justify-center text-white font-medium text-xl flex-shrink-0 ring-4 ring-white">
          {selected.slice(0, 2).toUpperCase()}
        </div>
        <div className="text-lg font-semibold text-gray-800 truncate">
          {selected}
        </div>
      </div>

      {/* Bot Toggle */}
      <div className="flex items-center justify-end gap-2 p-3 bg-slate-50 border-b border-slate-200">
        <span className="text-sm text-gray-600">Bot Enabled</span>
        <input
          type="checkbox"
          checked={botEnabledForClient}
          onChange={(e) => handleToggle(e.target.checked)}
          className="w-5 h-5 cursor-pointer"
        />
      </div>

      {/* Scrollable Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.direction === 'incoming' ? 'justify-start' : 'justify-end'}`}
          >
            <div
              className={`max-w-[70%] p-3 rounded-xl shadow-md
                ${
                  m.direction === 'incoming'
                    ? 'bg-white text-gray-800 rounded-bl-sm'
                    : 'bg-gradient-to-br from-green-500 to-green-600 text-white rounded-br-sm'
                }`}
            >
              <div className="text-base break-words">{m.message}</div>
              <div
                className={`text-xs mt-1 text-right ${
                  m.direction === 'incoming'
                    ? 'text-slate-400'
                    : 'text-green-100 opacity-80'
                }`}
              >
                {m.timestamp
                  ? new Date(m.timestamp).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })
                  : ''}
              </div>
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Fixed Input Box */}
      <div className="p-4 bg-white border-t border-slate-200 flex-none">
        <div className="flex gap-3">
          <input
            className="flex-1 p-3 px-5 bg-slate-100 rounded-full focus:outline-none focus:bg-white focus:ring-2 focus:ring-green-500 transition-all duration-300 ease-in-out"
            placeholder="Type a message..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSend() }}
          />
          <button
            className={`bg-gradient-to-br from-green-500 to-green-600 text-white p-3 rounded-full flex items-center justify-center
                        transform transition-all duration-300 ease-in-out
                        ${(!text.trim() || !botEnabledForClient)
                          ? 'opacity-60 saturate-50 cursor-not-allowed'
                          : 'hover:brightness-110 hover:scale-105 hover:shadow-lg hover:shadow-green-500/40'}`}
            onClick={handleSend}
            disabled={!text.trim() || !botEnabledForClient}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  )
}
