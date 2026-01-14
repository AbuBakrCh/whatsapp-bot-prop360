import React from 'react'
import { Link } from 'react-router-dom'

export default function ImportantLinks() {
  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="p-4 bg-white border-b border-green-200 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">Important Links</h1>
        <Link
          to="/"
          className="px-3 py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg text-sm transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      {/* Content Section */}
      <div className="flex-1 p-6 space-y-8">
        {/* OpenAI Section */}
        <div className="bg-white rounded-lg shadow-sm border p-4">
          <h2 className="text-lg font-medium text-gray-700 mb-3">
            OpenAI
          </h2>

          <a
            href="https://platform.openai.com/settings/organization/billing/overview"
            target="_blank"
            rel="noopener noreferrer"
            className="text-green-600 hover:underline font-medium"
          >
            OpenAI Platform Dashboard
          </a>
        </div>

        {/* Add more sections here in the future */}
      </div>
    </div>
  )
}
