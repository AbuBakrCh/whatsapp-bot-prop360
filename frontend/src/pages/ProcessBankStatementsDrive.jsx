import React, { useState } from "react";
import { processBankStatementsFromDrive } from "../api";

const PROP_FIELDS = [
  { id: "field-1757604618017-q2mtvmiqp", label: "Trx Ref. No" },
  { id: "field-1757605069078-p5plna7qr", label: "Trx Value Date" },
  { id: "field-1757605079803-icw8ykc19", label: "Transaction Date" },
  { id: "field-1757605194423-ofbqnqfso", label: "Transaction Bank" },
  { id: "field-1757605219384-fikyy3d7u", label: "Transaction Description" },
  { id: "field-1757605508754-uea4iadqd", label: "Trx Amount" },
  { id: "field-1757605632930-2hfg96qgr", label: "Invest Greece Notes" },
  { id: "field-1757605718340-ue95ozr9u", label: "Debit/Credit" },
  { id: "field-1758478644281-ct8pck6lk", label: "Trx Paym. Method" },
  { id: "field-1758478917909-jqxoz2s3h", label: "Trx Bank Statement" },
  { id: "field-1758699529035-xp7lumgx5", label: "Document Type" },
  { id: "field-1759392145178-qbx06ungl", label: "Month" },
  { id: "field-1759392151218-bmx5iidn6", label: "Year" },
];

export default function ProcessBankStatementsDrive() {
  const [driveLink, setDriveLink] = useState("");
  const [authToken, setAuthToken] = useState("");
  const [mapping, setMapping] = useState({});
  const [tempInputs, setTempInputs] = useState({});
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  // Keep controlled inputs
  const handleInputChange = (fieldId, value) => {
    setTempInputs((prev) => ({ ...prev, [fieldId]: value }));
  };

  const handleMappingBlur = (fieldId) => {
    const value = tempInputs[fieldId] || "";
    const cleaned = value
      .split(",")
      .map((h) => h.trim())
      .filter(Boolean);

    setMapping((prev) => {
      const newMap = { ...prev };
      // Remove old entries for this field
      Object.keys(newMap).forEach((k) => {
        if (newMap[k] === fieldId) delete newMap[k];
      });
      // Add new entries
      cleaned.forEach((header) => {
        newMap[header] = fieldId;
      });
      return newMap;
    });
  };

  const handleProcess = async () => {
    if (!driveLink.trim()) return setStatusMessage("❌ Enter folder link.");
    if (!authToken.trim()) return setStatusMessage("❌ Enter auth token.");
    if (!Object.keys(mapping).length) return setStatusMessage("❌ Provide at least one mapping.");

    setLoading(true);
    setStatusMessage("⏳ Processing...");

    try {
      const result = await processBankStatementsFromDrive(
        driveLink.trim(),
        authToken.trim(),
        mapping
      );
      setStatusMessage(`✅ Processed: ${result.processed.length} entries.`);
    } catch (err) {
      setStatusMessage(`❌ Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Process Documents</h2>

      <input
        type="text"
        placeholder="Google Drive Folder Link"
        value={driveLink}
        onChange={(e) => setDriveLink(e.target.value)}
        className="w-full mb-4 border p-2 rounded-md focus:ring-2 focus:ring-green-400"
      />

      <input
        type="text"
        placeholder="Auth Token"
        value={authToken}
        onChange={(e) => setAuthToken(e.target.value)}
        className="w-full mb-4 border p-2 rounded-md focus:ring-2 focus:ring-green-400"
      />

      <h3 className="text-lg font-semibold mt-4 mb-2 text-gray-700">Field Mapping</h3>
      <p className="text-sm text-gray-600 mb-3">
        Enter CSV column name(s) that map to each Prop360 field. Use comma-separated headers if needed.
      </p>

      <div className="grid grid-cols-2 gap-3 max-h-80 overflow-y-auto mb-4">
        {PROP_FIELDS.map((field) => (
          <div key={field.id} className="flex gap-2 items-center">
            <label className="w-1/2 text-sm text-gray-800">{field.label}</label>
            <input
              type="text"
              placeholder="CSV header(s)"
              value={tempInputs[field.id] ?? Object.keys(mapping).filter((k) => mapping[k] === field.id).join(", ")}
              onChange={(e) => handleInputChange(field.id, e.target.value)}
              onBlur={() => handleMappingBlur(field.id)}
              className="w-1/2 border p-1 rounded-md text-sm"
            />
          </div>
        ))}
      </div>

      {statusMessage && <div className="mb-4 text-gray-800">{statusMessage}</div>}

      <button
        onClick={handleProcess}
        disabled={loading}
        className={`w-full px-5 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Processing..." : "Process Documents"}
      </button>
    </div>
  );
}
