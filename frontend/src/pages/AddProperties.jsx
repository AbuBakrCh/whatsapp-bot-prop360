import React, { useState } from "react";
import { addProperties, deleteProperties } from "../api";

export default function AddingProperties() {
  const [sourceMerchantEmail, setSourceMerchantEmail] = useState("");
  const [targetMerchantEmails, setTargetMerchantEmails] = useState("");

  // Filters
  const [city, setCity] = useState("");
  const [division, setDivision] = useState("");
  const [published, setPublished] = useState(false);

  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const handleProcess = async (actionType) => {
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
          city,
          division,
          published,
        },
      };


      let res;

      if (actionType === "add") {
        res = await addProperties(payload);
      } else if (actionType === "delete") {
        res = await deleteProperties(payload);
      }

      if (res.error) {
        setResponseMsg(res.error);
      } else if (res.message) {
        setResponseMsg(res.message);
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(err.response?.data?.error || "Failed to process properties.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Manage Properties</h2>

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
        <input
          type="text"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          placeholder="City"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <input
          type="text"
          value={division}
          onChange={(e) => setDivision(e.target.value)}
          placeholder="Division"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={published}
            onChange={(e) => setPublished(e.target.checked)}
            className="w-4 h-4"
          />
          Is Published?
        </label>
      </div>

      {/* Buttons */}
      <div className="flex gap-4">
        <button
          onClick={() => handleProcess("add")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Processing..." : "Share Property"}
        </button>

        <button
          onClick={() => handleProcess("delete")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading ? "bg-red-300 cursor-not-allowed" : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading ? "Processing..." : "Unshare Property"}
        </button>
      </div>

      {/* Response Message */}
      {responseMsg && (
        <p
          className={`mt-4 ${
            responseMsg.toLowerCase().includes("success")
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {responseMsg}
        </p>
      )}
    </div>
  );
}
