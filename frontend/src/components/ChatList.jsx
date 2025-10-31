import React from 'react'

export default function ChatList({ conversations = [], onSelect, selected }) {
  return (
    <div className="h-full flex flex-col bg-green-50 border-r border-green-200">
      {/* Header */}
      <div className="p-4 text-lg font-semibold text-gray-800 border-b border-green-200">
        Chats
      </div>

      {/* Scrollable conversations */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-hide">
        {conversations.length === 0 && (
          <div className="text-gray-400 text-center mt-10">No chats yet</div>
        )}

        {conversations.map(c => (
          <div
            key={c.clientNumber}
            onClick={() => onSelect(c.clientNumber)}
            className={`p-3 rounded-2xl cursor-pointer transition-all duration-200 flex flex-col
              ${selected === c.clientNumber
                ? 'bg-green-500 text-white shadow-lg'
                : 'bg-white hover:shadow-md hover:scale-[1.02]'}`
            }
          >
            <div className="flex justify-between items-center">
                <div className="font-medium text-sm truncate">{c.clientName || c.clientNumber}</div>
              <div className="text-xs text-gray-400">
                {c.lastTimestamp
                  ? new Date(
                      c.lastTimestamp.endsWith('Z') ? c.lastTimestamp : c.lastTimestamp + 'Z'
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
            <div className="text-xs text-gray-700 mt-1 truncate">
              {c.lastMessage || 'No messages yet'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
