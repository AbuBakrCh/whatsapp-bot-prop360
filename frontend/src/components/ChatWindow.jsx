import React, { useRef, useEffect, useState } from 'react'
import { Send, MessagesSquare } from 'lucide-react'
import { getClientConfig, toggleClientBot, getDetails, updateDetails } from '../api';

export default function ChatWindow({ selected, selectedChat, messages = [], onSend, onDetailsUpdated }) {
  const bottomRef = useRef(null)
  const [text, setText] = useState('')
  const [botEnabledForClient, setBotEnabledForClient] = useState(true)
  const [showDetailsModal, setShowDetailsModal] = useState(false)
  const [contactName, setContactName] = useState('')
  const [contactInfo, setContactInfo] = useState('')
  const [isEditing, setIsEditing] = useState(false)
  const [loadingDetails, setLoadingDetails] = useState(false)

  // ← NEW: loader state for messages
  const [loadingMessages, setLoadingMessages] = useState(false)

  // Fetch existing details when modal opens
  useEffect(() => {
    const fetchDetails = async () => {
        if (!selected || !showDetailsModal) return;
        setLoadingDetails(true);
        try {
          const data = await getDetails(selected);
          setContactName(data.name || '');
          setContactInfo(data.info || '');
        } catch (err) {
          console.error('Error fetching details:', err);
        } finally {
          setLoadingDetails(false);
          setIsEditing(false);
        }
      };
    fetchDetails();
    }, [showDetailsModal, selected]);

  const handleSaveDetails = async () => {
      try {
        await updateDetails(selected, contactName, contactInfo);
        setShowDetailsModal(false);
        if (onDetailsUpdated) {
            onDetailsUpdated();
        }
      } catch (err) {
        console.error('Failed to update details:', err);
      }
    };


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
        setBotEnabledForClient(data.botEnabled !== false)
      } catch (err) {
        console.error('Failed to fetch client config:', err)
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
      console.error('Failed to update bot toggle:', err)
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
            {selectedChat?.clientName || selected}
        </div>

        {/* 🟢 Details Button */}
        <button
          onClick={() => setShowDetailsModal(true)}
          className="ml-auto px-3 py-1.5 text-sm font-medium bg-green-500 hover:bg-green-600 text-white rounded-md shadow-sm transition"
        >
          Details
        </button>
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
        {/* ← NEW: show loader if messages are loading */}
        {loadingMessages ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            Loading messages...
          </div>
        ) : (
          messages.map((m, i) => {
            const isIncoming = m.direction === 'incoming'
            let bubbleClass = ''
            if (isIncoming) {
              bubbleClass = 'bg-white text-gray-800 rounded-bl-sm shadow-md'
            } else if (m.outgoingSender === 'bot') {
              bubbleClass = 'bg-gradient-to-br from-green-500 to-green-600 text-white rounded-br-sm shadow-md'
            } else {
              bubbleClass = 'bg-blue-500 text-white rounded-br-sm shadow-md'
            }

            return (
              <div
                key={i}
                className={`flex flex-col ${isIncoming ? 'items-start' : 'items-end'}`}
              >
                {!isIncoming && m.outgoingSender && (
                  <div className="mb-1 text-xs px-2 py-0.5 rounded bg-green-100 text-green-800 select-none">
                    {m.outgoingSender === 'bot' ? 'Bot' : 'Admin'}
                  </div>
                )}
                <div className={`max-w-[70%] p-3 rounded-xl ${bubbleClass}`}>
                  <div className="text-base break-words">{m.message}</div>
                  {m.outgoingSender === 'bot' && m.context && (
                    <details className="mt-2 text-xs bg-green-700/10 p-2 rounded-md border border-green-500/30 whitespace-pre-wrap">
                      <summary className="cursor-pointer text-green-700 font-medium">🧩 Show context</summary>
                      <div className="mt-1 text-green-900">{m.context}</div>
                    </details>
                  )}
                  <div
                    className={`text-xs mt-1 text-right ${
                      isIncoming
                        ? 'text-slate-400'
                        : m.outgoingSender === 'bot'
                          ? 'text-green-100 opacity-80'
                          : 'text-blue-100 opacity-80'
                    }`}
                  >
                    {m.timestamp
                        ? new Date(
                            m.timestamp.endsWith('Z') ? m.timestamp : m.timestamp + 'Z'
                          ).toLocaleString([], {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                          })
                      : ''}

                  </div>
                </div>
              </div>
            )
          })
        )}
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
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSend()
            }}
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

      {/* 🪟 Details Modal */}
      {showDetailsModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg w-full max-w-md p-6 relative">
            <h2 className="text-lg font-semibold mb-4 text-gray-800">Contact Details</h2>

            {loadingDetails ? (
              <div className="text-gray-500 text-sm">Loading details...</div>
            ) : (
              <div className="space-y-3">
                <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">Name</label>
                  <input
                    type="text"
                    value={contactName}
                    onChange={(e) => setContactName(e.target.value)}
                    readOnly={!isEditing}
                    className={`w-full p-2 border rounded-md ${
                      isEditing
                        ? 'focus:ring-2 focus:ring-green-500 focus:outline-none'
                        : 'bg-gray-100 cursor-not-allowed text-gray-600'
                    }`}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-600 mb-1">
                    Other Information
                  </label>
                  <textarea
                    value={contactInfo}
                    onChange={(e) => setContactInfo(e.target.value)}
                    readOnly={!isEditing}
                    className={`w-full p-2 border rounded-md resize-none ${
                      isEditing
                        ? 'focus:ring-2 focus:ring-green-500 focus:outline-none'
                        : 'bg-gray-100 cursor-not-allowed text-gray-600'
                    }`}
                    rows={4}
                  />
                </div>
              </div>
            )}

            <div className="mt-5 flex justify-between items-center">
              <button
                onClick={() => setIsEditing((prev) => !prev)}
                className="px-3 py-1.5 text-sm font-medium text-green-600 hover:text-green-800"
              >
                {isEditing ? 'Cancel Edit' : 'Edit'}
              </button>

              <div className="flex gap-3">
                <button
                  onClick={() => setShowDetailsModal(false)}
                  className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800"
                >
                  Close
                </button>
                {isEditing && (
                  <button
                    onClick={handleSaveDetails}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white text-sm rounded-md"
                  >
                    OK
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
