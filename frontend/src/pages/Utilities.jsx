import React from "react";
import { Link } from "react-router-dom";
import SendBulkEmail from "./SendBulkEmail"; // use your correct component name
import ProcessBankStatementsDrive from "./ProcessBankStatementsDrive";

export default function Utilities() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Header */}
      <div className="p-4 bg-white border-b border-green-200 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">Utilities</h1>
        <Link
          to="/"
          className="px-3 py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg text-sm transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      {/* Content Section */}
      <div className="flex-1 p-6 space-y-8">
        <SendBulkEmail />
        <ProcessBankStatementsDrive />
      </div>
    </div>
  );
}
