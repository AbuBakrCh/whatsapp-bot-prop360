import React, { useEffect, useState, useRef } from 'react'
import { io } from 'socket.io-client'
import { getConversations, getChat, sendMessage } from './api'
import ChatList from './components/ChatList'
import ChatWindow from './components/ChatWindow'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || 'http://127.0.0.1:8000'

export default function App() {
  const [conversations, setConversations] = useState([])
  const [selected, setSelected] = useState(null)
  const [messages, setMessages] = useState([])
  const [loadingConversations, setLoadingConversations] = useState(true)
  const [loadingMessages, setLoadingMessages] = useState(false)
  const socketRef = useRef(null)

  useEffect(() => {
    loadConversations()

    // connect socket
    socketRef.current = io(SOCKET_URL, { transports: ['websocket'] })
    socketRef.current.on('connect', () => console.log('socket connected'))
    socketRef.current.on('new_message', (msg) => {
        if (msg.outgoingSender === 'admin') return  // ignore own messages

        setConversations(prev => {
          const existing = prev.find(c => c.clientNumber === msg.clientNumber)
          const others = prev.filter(c => c.clientNumber !== msg.clientNumber)

          const newItem = {
            clientNumber: msg.clientNumber,
            lastMessage: msg.message,
            direction: msg.direction,
            outgoingSender: msg.outgoingSender,
            lastTimestamp: msg.timestamp,
            clientName: existing?.clientName || msg.clientNumber
          }

          return [newItem, ...others]
        })

        if (selected === msg.clientNumber) {
          setMessages(prev => [...prev, msg])
        }
    })

    return () => socketRef.current && socketRef.current.disconnect()
  }, [selected])

  async function loadConversations(){
    setLoadingConversations(true)
    try {
      const data = await getConversations()
      setConversations(data.map(d => ({
        clientNumber: d.clientNumber,
        lastMessage: d.lastMessage,
        outgoingSender: d.outgoingSender,
        lastTimestamp: d.lastTimestamp,
        clientName: d.clientName
      })))
    } catch (e) {
      console.error(e)
    } finally {
      setLoadingConversations(false)
    }
  }

  async function openChat(clientNumber){
    setSelected(clientNumber)
    setLoadingMessages(true)
    try {
      const res = await getChat(clientNumber)
      setMessages(res.messages || [])
    } catch(e) {
      console.error(e)
    } finally {
      setLoadingMessages(false)
    }
  }

  async function handleSend(text){
    if(!selected || !text) return
    try {
      await sendMessage(selected, text)
      const msg = { clientNumber: selected, message: text, direction: 'outgoing', outgoingSender: 'admin', timestamp: new Date().toISOString() }
      setMessages(prev => [...prev, msg])
    } catch(e) {
      console.error(e)
      alert('Failed to send message')
    }
  }

  // Loader Component
  const Loader = ({ text }) => (
    <div className="flex items-center justify-center h-full text-gray-500">
      <div className="flex items-center space-x-2">
        <svg
          className="animate-spin h-5 w-5 text-green-500"
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          ></circle>
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8v4l3-3-3-3v4a8 8 0 018 8h-4l3 3-3 3h4a8 8 0 01-8 8v-4l-3 3 3 3v-4a8 8 0 01-8-8z"
          ></path>
        </svg>
        <span>{text}</span>
      </div>
    </div>
  )

  return (
    <div className="h-screen flex">
      {/* Chat List */}
      <div className="w-1/3 bg-green-50 border-r border-green-200 flex flex-col h-screen">
        {/* Fixed header */}
        <div className="p-4 border-b bg-white sticky top-0 z-20 flex justify-between items-center">
          <h1 className="text-lg font-semibold text-green-600">Kostas' Dashboard</h1>
          <span className="text-xs text-gray-400">v{__APP_VERSION__}</span>
        </div>

        {/* Scrollable chat list */}
        <div className="flex-1 overflow-y-auto p-3 scrollbar-hide">
          {loadingConversations ? (
            <Loader text="Loading chats..." />
          ) : (
            <ChatList conversations={conversations} onSelect={openChat} selected={selected} />
          )}
        </div>
      </div>

      {/* Chat Window */}
      <div className="flex-1 bg-white">
        {loadingMessages ? (
          <Loader text="Loading messages..." />
        ) : (
          <ChatWindow
            selected={selected}
            selectedChat={conversations.find(c => c.clientNumber === selected)}
            messages={messages}
            onSend={handleSend}
            onDetailsUpdated={loadConversations}
          />
        )}
      </div>
    </div>
  )
}
