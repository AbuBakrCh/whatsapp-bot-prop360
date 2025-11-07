import React, { useState } from "react";
import { processBankStatementsFromDrive } from "../api";

export default function ProcessBankStatementsDrive() {
  const [driveLink, setDriveLink] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const handleProcess = async () => {
    if (!driveLink.trim()) {
      setStatusMessage("❌ Please provide a Google Drive *folder* link.");
      return;
    }

    setLoading(true);
    setStatusMessage("⏳ Processing bank statements...");

    try {
      const result = await processBankStatementsFromDrive(driveLink.trim());

      setStatusMessage(`✅ Processed: ${result.processed.length} statements.`);
    } catch (err) {
      console.error(err);
      setStatusMessage(
        `❌ Error: ${err.response?.data?.detail || err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Process Bank Statements</h2>

      <p className="mb-4 text-gray-700">
        Paste a <b>public Google Drive folder link</b> containing bank statement images.
      </p>

      <input
        type="text"
        placeholder="Google Drive folder link"
        value={driveLink}
        onChange={(e) => setDriveLink(e.target.value)}
        className="w-full mb-4 border p-2 rounded-md focus:ring-2 focus:ring-green-400"
      />

      {statusMessage && <div className="mb-4 text-gray-800">{statusMessage}</div>}

      <button
        onClick={handleProcess}
        disabled={loading}
        className={`w-full px-5 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Processing..." : "Process Statements"}
      </button>
    </div>
  );
}
