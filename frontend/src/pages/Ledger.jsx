import React from "react";
import { Link } from "react-router-dom";
import PropertyCashflowLedger from "./PropertyCashflowLedger";

export default function Ledger() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <div className="p-4 bg-white border-b border-green-200 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">Ledger</h1>
        <Link
          to="/"
          className="px-3 py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg text-sm transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      <div className="flex-1 p-6">
        <PropertyCashflowLedger />
      </div>
    </div>
  );
}
