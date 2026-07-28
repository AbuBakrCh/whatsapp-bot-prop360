import React, { useState } from "react";
import { Link } from "react-router-dom";
import { ExternalLink } from "lucide-react";
import { getAccrualPaymentMatches } from "../api";

const PAGE_SIZE = 20;

function formatAmount(amount) {
  return Number(amount).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function CashflowLink({ row, label }) {
  if (!row?.url) {
    return <span className="text-gray-800 font-medium">{label}</span>;
  }
  return (
    <a
      href={row.url}
      target="_blank"
      rel="noopener noreferrer"
      title="Open in Prop360"
      className="inline-flex items-center gap-1.5 text-blue-600 hover:text-blue-800 font-medium"
    >
      <span>{label}</span>
      <ExternalLink className="w-3.5 h-3.5 shrink-0" aria-hidden="true" />
    </a>
  );
}

function MetaLine({ row }) {
  const parts = [
    row.documentType,
    row.description,
    row.contact,
    row.property,
    row.matchingNumber ? `Match #${row.matchingNumber}` : "",
  ].filter(Boolean);

  if (parts.length === 0) {
    return <p className="text-sm text-gray-400 mt-1">No extra details</p>;
  }

  return (
    <p className="text-sm text-gray-600 mt-1 break-words">
      {parts.join(" · ")}
    </p>
  );
}

function DateRangeFields({
  title,
  description,
  startValue,
  endValue,
  onStartChange,
  onEndChange,
}) {
  return (
    <div className="border border-slate-200 rounded-xl p-4 bg-slate-50/60">
      <h3 className="text-sm font-semibold text-gray-800">{title}</h3>
      {description && (
        <p className="text-sm text-gray-600 mt-1 mb-3">{description}</p>
      )}
      <div className="flex flex-col sm:flex-row gap-4">
        <label className="flex flex-col gap-1 text-gray-700 flex-1">
          Start
          <input
            type="date"
            value={startValue}
            onChange={(e) => onStartChange(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 bg-white"
          />
        </label>
        <label className="flex flex-col gap-1 text-gray-700 flex-1">
          End
          <input
            type="date"
            value={endValue}
            onChange={(e) => onEndChange(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 bg-white"
          />
        </label>
      </div>
    </div>
  );
}

function CandidateList({ candidates }) {
  if (!candidates || candidates.length === 0) {
    return (
      <p className="text-sm text-amber-700 bg-amber-50 border border-amber-100 rounded-md px-3 py-2">
        No matching payments
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {candidates.map((payment) => (
        <li
          key={payment.id}
          className="border border-slate-200 rounded-md px-3 py-2 bg-white"
        >
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <CashflowLink
              row={payment}
              label={`Payment ${payment.id?.slice(-8) || ""}`}
            />
            <span className="text-sm font-semibold text-gray-800">
              {formatAmount(payment.amount)}
            </span>
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-500 mt-1">
            <span>Created: {payment.createdAt || "—"}</span>
          </div>
          <MetaLine row={payment} />
        </li>
      ))}
    </ul>
  );
}

export default function AccrualPaymentMatching() {
  const [accrualStartDate, setAccrualStartDate] = useState("");
  const [accrualEndDate, setAccrualEndDate] = useState("");
  const [paymentStartDate, setPaymentStartDate] = useState("");
  const [paymentEndDate, setPaymentEndDate] = useState("");
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const loadMatches = async (nextPage = 1) => {
    if (
      !accrualStartDate ||
      !accrualEndDate ||
      !paymentStartDate ||
      !paymentEndDate
    ) {
      setError("Please fill in both accrual and payment date ranges.");
      return;
    }
    if (accrualEndDate < accrualStartDate) {
      setError("Accrual end date must be on or after its start date.");
      return;
    }
    if (paymentEndDate < paymentStartDate) {
      setError("Payment end date must be on or after its start date.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const data = await getAccrualPaymentMatches({
        accrualStartDate,
        accrualEndDate,
        paymentStartDate,
        paymentEndDate,
        page: nextPage,
        pageSize: PAGE_SIZE,
      });
      setPage(data?.pagination?.page || nextPage);
      setResult(data);
    } catch (err) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to load accrual–payment matches.";
      setError(detail);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const handleLoad = async (e) => {
    e.preventDefault();
    await loadMatches(1);
  };

  const summary = result?.summary;
  const pagination = result?.pagination;
  const accruals = result?.accruals || [];
  const totalPages = pagination?.totalPages || 1;
  const currentPage = pagination?.page || page;

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <div className="p-4 bg-white border-b border-green-200 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">
          Accrual–Payment Matching
        </h1>
        <Link
          to="/"
          className="px-3 py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg text-sm transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      <div className="flex-1 max-w-4xl mx-auto w-full p-6">
        <form
          onSubmit={handleLoad}
          className="bg-white rounded-2xl shadow-lg p-6 border border-slate-200 mb-6"
        >
          <p className="text-sm text-gray-600 mb-4">
            Dates are in Greece time. Matching is by equal amount only — nothing
            is saved.
          </p>

          <div className="flex flex-col gap-4 mb-4">
            <DateRangeFields
              title="Accrual period"
              description="Choose dates for accruals."
              startValue={accrualStartDate}
              endValue={accrualEndDate}
              onStartChange={setAccrualStartDate}
              onEndChange={setAccrualEndDate}
            />
            <DateRangeFields
              title="Payment period"
              description="Choose dates for matching payment identification."
              startValue={paymentStartDate}
              endValue={paymentEndDate}
              onStartChange={setPaymentStartDate}
              onEndChange={setPaymentEndDate}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? "Loading…" : "Load matches"}
          </button>

          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        </form>

        {summary && (
          <div className="flex flex-wrap gap-3 mb-6 text-sm">
            <span className="bg-white border border-slate-200 rounded-lg px-3 py-2">
              Accruals: <strong>{summary.accrualCount}</strong>
            </span>
            <span className="bg-white border border-slate-200 rounded-lg px-3 py-2">
              Payments on this page: <strong>{summary.paymentCount}</strong>
            </span>
            <span className="bg-white border border-slate-200 rounded-lg px-3 py-2">
              With candidates: <strong>{summary.withCandidates}</strong>
            </span>
            <span className="bg-white border border-slate-200 rounded-lg px-3 py-2">
              No match: <strong>{summary.withoutCandidates}</strong>
            </span>
          </div>
        )}

        {result && accruals.length === 0 && !loading && (
          <p className="text-center text-gray-500 py-10">
            No active accruals in this period.
          </p>
        )}

        <div className="space-y-4">
          {accruals.map((accrual) => (
            <div
              key={accrual.id}
              className="bg-white rounded-xl border border-slate-200 shadow-sm p-4"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2 mb-1">
                <CashflowLink
                  row={accrual}
                  label={`Accrual ${accrual.id?.slice(-8) || ""}`}
                />
                <span className="text-base font-semibold text-gray-900">
                  {formatAmount(accrual.amount)}
                </span>
              </div>
              <div className="text-xs text-gray-500 mb-2">
                Created: {accrual.createdAt || "—"}
              </div>
              <MetaLine row={accrual} />

              <div className="mt-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500 mb-2">
                  Candidate payments ({accrual.candidates?.length || 0})
                </h3>
                <CandidateList candidates={accrual.candidates} />
              </div>
            </div>
          ))}
        </div>

        {pagination && totalPages > 1 && (
          <div className="flex justify-center items-center gap-2 mt-6">
            <button
              type="button"
              onClick={() => loadMatches(currentPage - 1)}
              disabled={loading || currentPage <= 1}
              className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-sm"
            >
              Prev
            </button>
            <span className="text-sm text-gray-600">
              Page {currentPage} of {totalPages}
              {pagination.totalCount != null && (
                <span className="text-gray-400">
                  {" "}
                  ({pagination.totalCount} accruals)
                </span>
              )}
            </span>
            <button
              type="button"
              onClick={() => loadMatches(currentPage + 1)}
              disabled={loading || currentPage >= totalPages}
              className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50 text-sm"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
