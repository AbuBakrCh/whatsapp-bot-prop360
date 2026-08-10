import React, { useMemo, useState } from "react";
import {
  buildInvoiceNumber,
  downloadInvoicePdf,
  formatAmount,
  formatOrdinalDate,
  parseLocalDate,
} from "../utils/invoicePdf";

function todayIso() {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

export default function GenerateInvoice() {
  const [invoiceDate, setInvoiceDate] = useState(todayIso);
  const [amount, setAmount] = useState("");
  const [error, setError] = useState("");
  const [lastDownload, setLastDownload] = useState(null);
  const [busy, setBusy] = useState(false);

  const preview = useMemo(() => {
    if (!invoiceDate) return null;
    try {
      const date = parseLocalDate(invoiceDate);
      if (Number.isNaN(date.getTime())) return null;
      return {
        invoiceNumber: buildInvoiceNumber(date),
        invoiceDateLabel: formatOrdinalDate(date),
      };
    } catch {
      return null;
    }
  }, [invoiceDate]);

  const handleDownload = (e) => {
    e.preventDefault();
    setError("");
    setLastDownload(null);

    if (!invoiceDate) {
      setError("Please select an invoice date.");
      return;
    }
    const amountNum = Number(amount);
    if (!amount || Number.isNaN(amountNum) || amountNum <= 0) {
      setError("Please enter a valid amount greater than zero.");
      return;
    }

    setBusy(true);
    try {
      const result = downloadInvoicePdf({
        invoiceDate,
        amount: amountNum,
      });
      setLastDownload(result);
    } catch (err) {
      setError(err.message || "Failed to generate invoice PDF.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-2">Generate Invoice</h2>
      <p className="text-sm text-slate-600 mb-6">
        Create a consultant invoice PDF. Enter the date and amount; the invoice
        number is generated from the year and month. All other details match the
        standard template.
      </p>

      <form onSubmit={handleDownload} className="flex flex-col gap-4 mb-6">
        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-slate-700">
            Invoice Date
          </span>
          <input
            type="date"
            value={invoiceDate}
            onChange={(e) => {
              setInvoiceDate(e.target.value);
              setError("");
              setLastDownload(null);
            }}
            className="border border-slate-300 rounded-md px-3 py-2"
            required
          />
        </label>

        <label className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-slate-700">
            Total Amount (USD)
          </span>
          <input
            type="number"
            min="0.01"
            step="0.01"
            value={amount}
            onChange={(e) => {
              setAmount(e.target.value);
              setError("");
              setLastDownload(null);
            }}
            placeholder="1700.00"
            className="border border-slate-300 rounded-md px-3 py-2"
            required
          />
        </label>

        {preview && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 space-y-1">
            <div>
              <span className="font-medium">Invoice Number:</span>{" "}
              {preview.invoiceNumber}
            </div>
            <div>
              <span className="font-medium">Invoice Date:</span>{" "}
              {preview.invoiceDateLabel}
            </div>
            {amount && Number(amount) > 0 && (
              <div>
                <span className="font-medium">Total:</span>{" "}
                {formatAmount(amount)}
              </div>
            )}
          </div>
        )}

        <button
          type="submit"
          disabled={busy}
          className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 text-white font-medium rounded-md px-4 py-2 transition-colors"
        >
          {busy ? "Generating…" : "Download PDF"}
        </button>
      </form>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 text-red-800 px-4 py-3 mb-4 text-sm">
          {error}
        </div>
      )}

      {lastDownload && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 text-emerald-900 px-4 py-3 text-sm">
          Downloaded <span className="font-medium">{lastDownload.filename}</span>{" "}
          ({lastDownload.invoiceNumber}).
        </div>
      )}
    </div>
  );
}
