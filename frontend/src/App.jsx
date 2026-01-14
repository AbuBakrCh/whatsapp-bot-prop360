import React, { useEffect, useState, useRef } from 'react'
import { io } from 'socket.io-client'
import { getConversations, getChat, sendMessage } from './api'
import ChatList from './components/ChatList'
import ChatWindow from './components/ChatWindow'
import { Link } from "react-router-dom"

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://127.0.0.1:8000'

export default function App() {
  const [conversations, setConversations] = useState([])
  const [selected, setSelected] = useState(null)
  const [messages, setMessages] = useState([])
  const [loadingConversations, setLoadingConversations] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)

  const socketRef = useRef(null)
  const selectedRef = useRef(null)

  /* keep selected ref in sync */
  useEffect(() => {
    selectedRef.current = selected
  }, [selected])

  /* initial load + socket */
  useEffect(() => {
    loadConversations(true)

    socketRef.current = io(SOCKET_URL, { transports: ['websocket'] })
    socketRef.current.on('connect', () => console.log('socket connected'))

    socketRef.current.on('new_message', (msg) => {
      if (msg.outgoingSender === 'admin') return

      setConversations(prev => {
        const existing = prev.find(c => c.clientNumber === msg.clientNumber)
        const others = prev.filter(c => c.clientNumber !== msg.clientNumber)

        return [{
          clientNumber: msg.clientNumber,
          lastMessage: msg.message,
          direction: msg.direction,
          outgoingSender: msg.outgoingSender,
          lastTimestamp: msg.timestamp,
          clientName: existing?.clientName || msg.clientNumber
        }, ...others]
      })

      if (selectedRef.current === msg.clientNumber) {
        setMessages(prev => [...prev, msg])
      }
    })

    return () => socketRef.current?.disconnect()
  }, [])

  /* pagination loader */
  async function loadConversations(reset = false) {
    if (!hasMore && !reset) return

    setLoadingConversations(true)
    try {
      const nextPage = reset ? 1 : page
      const data = await getConversations(nextPage, 20)

      setConversations(prev =>
        reset ? data.items : [...prev, ...data.items]
      )

      setHasMore(data.items.length === 20)
      setPage(nextPage + 1)
    } catch (e) {
      console.error(e)
    } finally {
      setLoadingConversations(false)
    }
  }

  async function openChat(clientNumber) {
    setSelected(clientNumber)
    setLoadingMessages(true)
    try {
      const res = await getChat(clientNumber)
      setMessages(res.messages || [])
    } catch (e) {
      console.error(e)
    } finally {
      setLoadingMessages(false)
    }
  }

  async function handleSend(text) {
    if (!selected || !text) return

    try {
      await sendMessage(selected, text)
      setMessages(prev => [...prev, {
        clientNumber: selected,
        message: text,
        direction: 'outgoing',
        outgoingSender: 'admin',
        timestamp: new Date().toISOString()
      }])
    } catch (e) {
      console.error(e)
      alert('Failed to send message')
    }
  }

return (
  <div className="h-screen flex flex-col">
    {/* Full-width Header */}
    <div className="h-[70px] px-4 bg-white border-b sticky top-0 z-30 flex justify-between items-center shadow-sm">
      <h1 className="text-lg font-semibold text-green-600">
        Kostas' Dashboard
      </h1>

      <div className="flex items-center gap-3">
        <Link
          to="/utilities"
          className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md text-sm font-medium transition"
        >
          Utilities
        </Link>

        <Link
          to="/important-links"
          className="px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md text-sm font-medium transition"
        >
          Important links
        </Link>

        <span className="text-xs text-gray-400">
          v{__APP_VERSION__}
        </span>
      </div>
    </div>

    {/* Main content */}
    <div className="flex flex-1 h-full">
      {/* Chat List */}
      <div className="w-1/3 bg-green-50 border-r border-green-200 flex flex-col h-full">
        <ChatList
          conversations={conversations}
          selected={selected}
          onSelect={openChat}
          onLoadMore={loadConversations}
          loading={loadingConversations}
          hasMore={hasMore}
          page={page}
        />
      </div>

      {/* Chat Window */}
      <div className="flex-1 bg-white">
        {loadingMessages ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            Loading messages…
          </div>
        ) : (
          <ChatWindow
            selected={selected}
            selectedChat={conversations.find(c => c.clientNumber === selected)}
            messages={messages}
            onSend={handleSend}
            onDetailsUpdated={() => loadConversations(true)}
          />
        )}
      </div>
    </div>
  </div>
)
}
