import React, { useState } from "react";
import { ExternalLink } from "lucide-react";
import { getCashflowLedger, exportCashflowLedger } from "../api";

const ACTIVITY_PAGE_SIZE = 10;

function ActivitiesTable({ activities, onPageChange, loading }) {
  const rows = activities?.rows || [];
  const page = activities?.page ?? 1;
  const totalPages = activities?.totalPages ?? 1;
  const totalCount = activities?.totalCount ?? 0;

  return (
    <div className="w-full mt-8">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-semibold text-blue-700">Activities</h3>
        {totalCount > 0 && (
          <span className="text-sm text-gray-500">{totalCount} total</span>
        )}
      </div>
      <div className="overflow-x-auto border border-slate-200 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-slate-100">
            <tr>
              <th className="text-left px-3 py-2 font-medium text-gray-700 w-40">Date</th>
              <th className="text-center px-3 py-2 font-medium text-gray-700 w-12">
                <span className="sr-only">Open</span>
              </th>
              <th className="text-left px-3 py-2 font-medium text-gray-700">
                Activity Description
              </th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={3} className="px-3 py-6 text-center text-gray-400">
                  Loading activities...
                </td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-6 text-center text-gray-400">
                  No activities
                </td>
              </tr>
            ) : (
              rows.map((row, index) => (
                <tr key={row.id || `${row.date}-${index}`} className="border-t border-slate-100">
                  <td className="px-3 py-2 whitespace-nowrap align-top">{row.date}</td>
                  <td className="px-3 py-2 text-center align-top">
                    {row.url ? (
                      <a
                        href={row.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Open activity in Prop360"
                        className="inline-flex text-blue-600 hover:text-blue-800"
                      >
                        <ExternalLink className="w-4 h-4" aria-hidden="true" />
                      </a>
                    ) : (
                      "—"
                    )}
                  </td>
                  <td className="px-3 py-2 whitespace-pre-wrap">{row.description || "—"}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-4">
          <button
            type="button"
            onClick={() => onPageChange(page - 1)}
            disabled={loading || page <= 1}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-sm"
          >
            Prev
          </button>
          <span className="text-sm text-gray-600">
            Page {page} of {totalPages}
          </span>
          <button
            type="button"
            onClick={() => onPageChange(page + 1)}
            disabled={loading || page >= totalPages}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-sm"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function LedgerTable({ title, section, accentClass }) {
  const rows = section?.transactions || [];
  const sum = section?.sum ?? 0;

  return (
    <div className="flex-1 min-w-0">
      <h3 className={`text-lg font-semibold mb-3 ${accentClass}`}>{title}</h3>
      <div className="overflow-x-auto border border-slate-200 rounded-lg">
        <table className="w-full text-sm">
          <thead className="bg-slate-100">
            <tr>
              <th className="text-left px-3 py-2 font-medium text-gray-700">Date</th>
              <th className="text-left px-3 py-2 font-medium text-gray-700">
                Transaction Description
              </th>
              <th className="text-right px-3 py-2 font-medium text-gray-700">Amount</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr>
                <td colSpan={3} className="px-3 py-6 text-center text-gray-400">
                  No transactions
                </td>
              </tr>
            ) : (
              rows.map((row, index) => (
                <tr key={`${row.date}-${index}`} className="border-t border-slate-100">
                  <td className="px-3 py-2 whitespace-nowrap">{row.date}</td>
                  <td className="px-3 py-2">{row.description || "—"}</td>
                  <td className="px-3 py-2 text-right whitespace-nowrap">
                    {Number(row.amount).toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </td>
                </tr>
              ))
            )}
            <tr className="border-t-2 border-slate-300 bg-slate-50 font-semibold">
              <td className="px-3 py-2" colSpan={2}>
                Total
              </td>
              <td className="px-3 py-2 text-right">
                {Number(sum).toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function PropertyCashflowLedger() {
  const [propertyId, setPropertyId] = useState("");
  const [ledger, setLedger] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activitiesLoading, setActivitiesLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const loadLedger = async (trimmedId, activityPage = 1, { fullLoad = false } = {}) => {
    if (fullLoad) {
      setLoading(true);
      setLedger(null);
    } else {
      setActivitiesLoading(true);
    }
    setStatusMessage("");

    try {
      const data = await getCashflowLedger(trimmedId, {
        activityPage,
        activityPageSize: ACTIVITY_PAGE_SIZE,
      });
      if (data?.error) {
        setStatusMessage(data.error);
        return;
      }
      setLedger(data);
      if (fullLoad) {
        setStatusMessage(
          `Loaded ${data.transactionCount ?? 0} transaction(s) and ${data.activityCount ?? 0} activity(ies) for property ${data.propertyId}.`
        );
      }
    } catch (err) {
      setStatusMessage(
        err.response?.data?.detail || err.message || "Failed to load ledger."
      );
    } finally {
      setLoading(false);
      setActivitiesLoading(false);
    }
  };

  const handleLoad = async () => {
    const trimmedId = propertyId.trim();
    if (!trimmedId) {
      setStatusMessage("Property ID is required.");
      return;
    }
    await loadLedger(trimmedId, 1, { fullLoad: true });
  };

  const handleActivityPageChange = async (nextPage) => {
    const trimmedId = propertyId.trim() || ledger?.propertyId;
    if (!trimmedId || nextPage < 1) return;
    await loadLedger(trimmedId, nextPage, { fullLoad: false });
  };

  const handleExport = async () => {
    const trimmedId = propertyId.trim();
    if (!trimmedId) {
      setStatusMessage("Property ID is required.");
      return;
    }

    setExporting(true);
    setStatusMessage("");

    try {
      await exportCashflowLedger(trimmedId);
      setStatusMessage("Excel file downloaded.");
    } catch (err) {
      setStatusMessage(
        err.response?.data?.detail || err.message || "Failed to export ledger."
      );
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Property Cashflow Ledger</h2>
      <p className="text-sm text-gray-600 mb-4">
        Enter a property ID (the value after the pipe in the property field) to build a
        debit/credit ledger from cashflow records.
      </p>

      <div className="flex flex-col sm:flex-row gap-3 mb-4">
        <input
          type="text"
          value={propertyId}
          onChange={(e) => setPropertyId(e.target.value)}
          placeholder="Property ID (e.g. 7279970513734244)"
          className="flex-1 border border-slate-300 rounded-md px-3 py-2"
        />
        <button
          onClick={handleLoad}
          disabled={loading}
          className={`px-5 py-2 rounded-md text-white font-medium transition ${
            loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Loading..." : "Load Ledger"}
        </button>
        <button
          onClick={handleExport}
          disabled={exporting}
          className={`px-5 py-2 rounded-md text-white font-medium transition ${
            exporting ? "bg-blue-300 cursor-not-allowed" : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {exporting ? "Exporting..." : "Export Excel"}
        </button>
      </div>

      {statusMessage && (
        <p
          className={`mb-4 text-sm ${
            statusMessage.toLowerCase().includes("fail") ||
            statusMessage.toLowerCase().includes("required")
              ? "text-red-600"
              : "text-gray-700"
          }`}
        >
          {statusMessage}
        </p>
      )}

      {ledger && (
        <div className="mb-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm">
          <p className="text-gray-800">
            <span className="font-medium text-gray-600">Property ID:</span>{" "}
            {ledger.propertyId}
          </p>
          <p className="mt-1 text-gray-800">
            <span className="font-medium text-gray-600">Property Name:</span>{" "}
            {ledger.propertyName || "—"}
          </p>
        </div>
      )}

      {ledger && (
        <div className="flex flex-col lg:flex-row gap-6">
          <LedgerTable
            title="Debit"
            section={ledger.debit}
            accentClass="text-red-700"
          />
          <LedgerTable
            title="Credit"
            section={ledger.credit}
            accentClass="text-green-700"
          />
        </div>
      )}

      {ledger && (
        <ActivitiesTable
          activities={ledger.activities}
          onPageChange={handleActivityPageChange}
          loading={activitiesLoading}
        />
      )}
    </div>
  );
}
