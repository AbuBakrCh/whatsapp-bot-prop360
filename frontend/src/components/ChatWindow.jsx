import React, { useRef, useEffect, useState } from 'react'
import { Send, MessagesSquare } from 'lucide-react'

export default function ChatWindow({ selected, messages = [], onSend }) {
  const bottomRef = useRef(null)
  const [text, setText] = useState('')

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, selected])

  const handleSend = () => {
    if (text.trim()) {
      onSend(text)
      setText('')
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

      {/* Scrollable Messages */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-3">
        {messages.map((m, i) => {
          let bubbleClasses = ''
          let timestampClasses = ''

          if (m.direction === 'incoming') {
            bubbleClasses = 'bg-white text-gray-800 rounded-bl-sm'
            timestampClasses = 'text-slate-400'
          } else if (m.outgoingSender === 'admin') {
            bubbleClasses = 'bg-gradient-to-br from-green-500 to-green-600 text-white rounded-br-sm'
            timestampClasses = 'text-green-100 opacity-80'
          } else if (m.outgoingSender === 'bot') {
            bubbleClasses = 'bg-green-300 text-gray-900 italic rounded-br-sm'
            timestampClasses = 'text-green-700 opacity-90 italic'
          }

          return (
            <div
              key={i}
              className={`flex ${m.direction === 'incoming' ? 'justify-start' : 'justify-end'}`}
            >
              <div className={`max-w-[70%] p-3 rounded-xl shadow-md ${bubbleClasses}`}>
                <div className="text-base break-words">{m.message}</div>
                <div className={`text-xs mt-1 text-right ${timestampClasses}`}>
                  {m.timestamp
                    ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    : ''}
                  {m.outgoingSender === 'bot' && (
                    <span className="ml-2 text-xs text-gray-600">bot</span>
                  )}
                  {m.outgoingSender === 'admin' && (
                    <span className="ml-2 text-xs text-white-600">admin</span>
                  )}
                </div>
              </div>
            </div>
          )
        })}
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
                        ${!text.trim()
                          ? 'opacity-60 saturate-50 cursor-not-allowed'
                          : 'hover:brightness-110 hover:scale-105 hover:shadow-lg hover:shadow-green-500/40'}`}
            onClick={handleSend}
            disabled={!text.trim()}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  )
}
