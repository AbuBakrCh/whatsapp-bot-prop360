import React, { useState } from "react";
import { addCashflows, deleteCashflows } from "../api";

const YES_NO_OPTIONS = ["Yes", "No"];

const emptyFilters = () => ({
  toBeShared: "",
});

function YesNoSelect({ label, value, onChange }) {
  return (
    <label className="flex flex-col gap-1 text-gray-700">
      <span className="text-sm font-medium">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border border-slate-300 rounded-md px-3 py-2 bg-white"
      >
        <option value="">Any</option>
        {YES_NO_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function ManageCashflows() {
  const [sourceMerchantEmail, setSourceMerchantEmail] = useState("");
  const [targetMerchantEmails, setTargetMerchantEmails] = useState("");
  const [filters, setFilters] = useState(emptyFilters);

  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleProcess = async (actionType) => {
    if (!targetMerchantEmails) {
      setResponseMsg("Target Merchant Emails are required.");
      return;
    }

    if (actionType === "add" && !sourceMerchantEmail) {
      setResponseMsg("Source Merchant Email is required for adding cashflows.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const payload = {
        sourceMerchantEmail: sourceMerchantEmail.trim().toLowerCase(),
        targetMerchantEmails: targetMerchantEmails
          .split(",")
          .map((email) => email.trim().toLowerCase())
          .filter(Boolean),
        filters: {
          toBeShared: filters.toBeShared || undefined,
        },
      };

      let res;

      if (actionType === "add") {
        res = await addCashflows(payload);
      } else if (actionType === "delete") {
        res = await deleteCashflows(payload);
      }

      if (res?.error) {
        setResponseMsg(res.error);
      } else if (res?.message) {
        setResponseMsg(
          res.updated !== undefined
            ? `${res.message} (updated: ${res.updated})`
            : res.message
        );
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(
        err.response?.data?.error || "Failed to process cashflows."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Manage Cashflows
      </h2>

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
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xl font-semibold text-gray-700">Filters</h3>
        <button
          type="button"
          onClick={() => setFilters(emptyFilters())}
          className="text-sm text-slate-500 hover:text-slate-700 underline"
        >
          Reset filters
        </button>
      </div>
      <p className="text-sm text-slate-500 mb-4">
        Leave filters empty / on Any to ignore them. Selected filters are
        combined with AND.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <YesNoSelect
          label="To Be Shared"
          value={filters.toBeShared}
          onChange={(v) => updateFilter("toBeShared", v)}
        />
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={() => handleProcess("add")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading
              ? "bg-green-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Processing..." : "Share Cashflow"}
        </button>

        <button
          onClick={() => handleProcess("delete")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading
              ? "bg-red-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading ? "Processing..." : "Unshare Cashflow"}
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
