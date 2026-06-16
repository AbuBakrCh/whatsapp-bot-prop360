import React from 'react'
import { Link } from 'react-router-dom'
import { isMessengerEnabled } from './config/messenger'

const sections = [
  { to: '/utilities', label: 'Utilities', description: 'Email, contacts, properties, jobs, and more' },
  { to: '/important-links', label: 'Important links', description: 'Quick access to external tools' },
  { to: '/spitogatos', label: 'Spitogatos', description: 'Property listings and crawler' },
  { to: '/ledger', label: 'Ledger', description: 'Property cashflow and finances' },
  { to: '/groups', label: 'Groups', description: 'Manage and share with contact groups' },
]

const messengerSection = {
  to: '/messenger',
  label: 'WhatsApp Messenger',
  description: 'View and reply to WhatsApp conversations',
}

export default function App() {
  const visibleSections = isMessengerEnabled()
    ? [messengerSection, ...sections]
    : sections

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <div className="h-[70px] px-4 bg-white border-b sticky top-0 z-30 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">
          Kostas' Dashboard
        </h1>

        <span className="text-xs text-gray-400">
          v{__APP_VERSION__}
        </span>
      </div>

      <div className="flex-1 p-8 max-w-3xl mx-auto w-full">
        <h2 className="text-2xl font-semibold text-gray-800 mb-2">Welcome</h2>
        <p className="text-gray-500 mb-8">Select a section to get started.</p>

        <div className="grid gap-4">
          {visibleSections.map(({ to, label, description }) => (
            <Link
              key={to}
              to={to}
              className="block p-5 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md hover:border-green-300 transition"
            >
              <div className="text-lg font-medium text-green-600">{label}</div>
              <div className="text-sm text-gray-500 mt-1">{description}</div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
