import React, { useState } from "react";
import { createContactFromInvoice } from "../api";

export default function CreateContactFromInvoice() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a PDF or image invoice.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const data = await createContactFromInvoice(file);
      setResult(data);
    } catch (err) {
      const detail =
        err.response?.data?.detail ||
        err.response?.data?.error ||
        err.message ||
        "Failed to create contact from invoice.";
      setError(typeof detail === "string" ? detail : JSON.stringify(detail));
    } finally {
      setLoading(false);
    }
  };

  const matchedByLabel =
    result?.matchedBy === "tax_number"
      ? "Tax Number / AFM"
      : result?.matchedBy === "name"
        ? "Company name"
        : null;

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">
        Create Contact from Invoice
      </h2>
      <p className="text-sm text-slate-600 mb-6">
        Upload a PDF or image invoice. The issuer is matched by Tax Number/AFM
        first, then company name. A new contact is created only if none exists.
      </p>

      <form onSubmit={handleSubmit} className="flex flex-col gap-4 mb-6">
        <input
          type="file"
          accept=".pdf,image/*,.png,.jpg,.jpeg"
          onChange={(e) => {
            setFile(e.target.files?.[0] || null);
            setError("");
            setResult(null);
          }}
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <button
          type="submit"
          disabled={loading || !file}
          className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 text-white font-medium rounded-md px-4 py-2 transition-colors"
        >
          {loading ? "Processing…" : "Create Contact"}
        </button>
      </form>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 text-red-800 px-4 py-3 mb-4 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div
          className={`rounded-lg border px-4 py-4 text-sm ${
            result.status === "created"
              ? "border-emerald-200 bg-emerald-50 text-emerald-900"
              : "border-amber-200 bg-amber-50 text-amber-900"
          }`}
        >
          <p className="font-semibold text-base mb-3">
            {result.status === "created"
              ? "Contact created"
              : "Existing contact found"}
          </p>

          {matchedByLabel && (
            <p className="mb-2">
              Matched by: <span className="font-medium">{matchedByLabel}</span>
            </p>
          )}

          <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 mb-3">
            <div>
              <dt className="text-slate-500">Issuer</dt>
              <dd className="font-medium">
                {result.extracted?.issuer || result.contact?.fullName || "—"}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Tax Number / AFM</dt>
              <dd className="font-medium">
                {result.extracted?.taxId || result.contact?.taxNumber || "—"}
              </dd>
            </div>
            <div>
              <dt className="text-slate-500">Profession</dt>
              <dd className="font-medium">
                {result.extracted?.profession || result.contact?.profession || "—"}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-slate-500">Address</dt>
              <dd className="font-medium">
                {result.extracted?.address || result.contact?.address || "—"}
              </dd>
            </div>
            <div className="sm:col-span-2">
              <dt className="text-slate-500">Contact ID</dt>
              <dd className="font-medium break-all">
                {result.contactId ? (
                  <a
                    href={`https://prop360.pro/dashboard/forms/contacts/${result.contactId}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline hover:no-underline"
                  >
                    {result.contactId}
                  </a>
                ) : (
                  "—"
                )}
              </dd>
            </div>
          </dl>
        </div>
      )}
    </div>
  );
}
