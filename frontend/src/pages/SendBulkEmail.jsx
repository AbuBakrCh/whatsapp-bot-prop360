import React, { useState } from "react";
import { sendBulkEmailFile } from "../api"; // handles FormData POST internally

export default function SendBulkEmailDrive() {
  const [driveLink, setDriveLink] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const handleSendEmails = async () => {
    if (!driveLink.trim()) {
      setStatusMessage("❌ Please provide a Google Drive link.");
      return;
    }

    setLoading(true);
    setStatusMessage("⏳ Sending emails... Please wait.");

    try {
      // Pass the raw string; api.js will convert to FormData
      const result = await sendBulkEmailFile(driveLink.trim());

      setStatusMessage(
        `✅ Emails sent: ${result.sent}. ❌ Failed: ${result.failed.length}`
      );
    } catch (err) {
      console.error(err);
      setStatusMessage(
        `❌ Error sending emails: ${err.response?.data?.detail || err.message}`
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Send Bulk Email</h2>

      <p className="mb-4 text-gray-700">
        Paste a <b>public Google Drive link</b> to your Excel file containing email data.
      </p>

      <input
        type="text"
        placeholder="Google Drive link"
        value={driveLink}
        onChange={(e) => setDriveLink(e.target.value)}
        className="w-full mb-4 border p-2 rounded-md focus:ring-2 focus:ring-green-400"
      />

      <details className="mb-4 text-gray-600 text-sm">
        <summary className="cursor-pointer font-medium">File Format Instructions</summary>
        <ul className="mt-2 list-disc list-inside">
          <li><b>Columns:</b> subject | content | attachments | recipients | cc | bcc</li>
          <li><b>attachments:</b> comma-separated public Drive links (optional)</li>
          <li><b>recipients, cc, bcc:</b> comma-separated emails</li>
        </ul>
        <p className="mt-1 text-xs italic">
          Example row: Welcome | Hello! | link1,link2 | a@example.com,b@example.com | cc@example.com | bcc@example.com
        </p>
      </details>

      {statusMessage && <div className="mb-4 text-gray-800">{statusMessage}</div>}

      <button
        onClick={handleSendEmails}
        disabled={loading}
        className={`w-full px-5 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Sending..." : "Send Emails"}
      </button>
    </div>
  );
}
