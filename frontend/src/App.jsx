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
  const socketRef = useRef(null)

  useEffect(() => {
    loadConversations()
    // connect socket
    socketRef.current = io(SOCKET_URL, { transports: ['websocket'] })
    socketRef.current.on('connect', () => console.log('socket connected'))
    socketRef.current.on('new_message', (msg) => {
        if (msg.outgoingSender === 'admin') return  // ignore own messages
        setConversations(prev => {
          const others = prev.filter(c => c.clientNumber !== msg.clientNumber)
          const newItem = {
            clientNumber: msg.clientNumber,
            lastMessage: msg.message,
            direction: msg.direction,
            outgoingSender: msg.outgoingSender,
            lastTimestamp: msg.timestamp
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
    }
  }

  async function openChat(clientNumber){
    setSelected(clientNumber)
    try {
      const res = await getChat(clientNumber)
      setMessages(res.messages || [])
    } catch(e) {
      console.error(e)
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

  return (
    <div className="h-screen flex">
      <div className="w-1/3 bg-green-50 border-r border-green-200 flex flex-col">
        {/* Fixed header */}
        <div className="p-4 border-b bg-white sticky top-0 z-20">
          <h1 className="text-lg font-semibold text-green-600">Kostas' Dashboard</h1>
        </div>
        {/* Scrollable chat list */}
        <ChatList conversations={conversations} onSelect={openChat} selected={selected} />
      </div>

      <div className="flex-1 bg-white">
        <ChatWindow
          selected={selected}
          messages={messages}
          onSend={handleSend}
          onDetailsUpdated={loadConversations}
        />
      </div>
    </div>
  )
}
