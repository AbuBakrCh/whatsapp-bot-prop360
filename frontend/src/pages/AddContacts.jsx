import React, { useState } from "react";
import { addContacts } from "../api";

export default function AddingContacts() {
  const [sourceMerchantEmail, setSourceMerchantEmail] = useState("");
  const [targetMerchantEmails, setTargetMerchantEmails] = useState("");

  // New toggles
  const [searchForProperty, setSearchForProperty] = useState(false);
  const [doesHeHaveProperty, setDoesHeHaveProperty] = useState(false);

  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const handleProcess = async () => {
    if (!sourceMerchantEmail || !targetMerchantEmails) {
      setResponseMsg("Source Merchant Email and Target Merchant Emails are required.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const payload = {
        sourceMerchantEmail,
        targetMerchantEmails: targetMerchantEmails
          .split(",")
          .map((email) => email.trim())
          .filter(Boolean),
        filters: {
          searchForProperty,
          doesHeHaveProperty,
        },
      };

      const res = await addContacts(payload);

      if (res.error) {
        setResponseMsg(res.error);
      } else if (res.message) {
        setResponseMsg(res.message);
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(err.response?.data?.error || "Failed to process contacts.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Adding Contacts</h2>

      {/* Main Inputs */}
      <div className="flex flex-col gap-4 mb-6">
        <input
          type="text"
          value={sourceMerchantEmail}
          onChange={(e) => setSourceMerchantEmail(e.target.value)}
          placeholder="Source Merchant Email"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <textarea
          value={targetMerchantEmails}
          onChange={(e) => setTargetMerchantEmails(e.target.value)}
          placeholder="Target Merchant Emails (comma separated)"
          className="border border-slate-300 rounded-md px-3 py-2 h-24 resize-none"
        />
      </div>

      {/* Filters */}
      <h3 className="text-xl font-semibold text-gray-700 mb-2">Filters</h3>

      <div className="flex flex-col gap-4 mb-6">
        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={searchForProperty}
            onChange={(e) => setSearchForProperty(e.target.checked)}
            className="w-4 h-4"
          />
          Search For Property?
        </label>

        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={doesHeHaveProperty}
            onChange={(e) => setDoesHeHaveProperty(e.target.checked)}
            className="w-4 h-4"
          />
          Does He Have Property?
        </label>
      </div>

      {/* Process Button */}
      <button
        onClick={handleProcess}
        disabled={loading}
        className={`w-full px-4 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Processing..." : "Process"}
      </button>

      {/* Response Message */}
      {responseMsg && (
        <p
          className={`mt-4 ${
            responseMsg.includes("success") ? "text-green-600" : "text-red-600"
          }`}
        >
          {responseMsg}
        </p>
      )}
    </div>
  );
}
