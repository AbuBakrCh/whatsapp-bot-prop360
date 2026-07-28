import React, { useState } from "react";
import { ExternalLink } from "lucide-react";
import PageShell, { PageIntro, ToolPanel } from "../components/PageShell";
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
      className="inline-flex items-center gap-1.5 text-whatsapp-500 hover:brightness-110 font-medium"
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
    <div className="border border-emerald-900/10 rounded-xl p-4 bg-emerald-50/40">
      <h3 className="text-sm font-semibold text-emerald-950">{title}</h3>
      {description && (
        <p className="text-sm text-slate-600 mt-1 mb-3">{description}</p>
      )}
      <div className="flex flex-col sm:flex-row gap-4">
        <label className="flex flex-col gap-1 text-slate-700 flex-1">
          Start
          <input
            type="date"
            value={startValue}
            onChange={(e) => onStartChange(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 bg-white"
          />
        </label>
        <label className="flex flex-col gap-1 text-slate-700 flex-1">
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
    <PageShell title="Accrual–Payment Matching" maxWidthClass="max-w-4xl">
      <PageIntro
        eyebrow="Matching"
        title="Accrual–Payment"
        description="Match cashflow accruals to equal-amount payments by period. Nothing is saved."
      />

      <div className="animate-welcome-rise space-y-6">
        <ToolPanel>
          <form onSubmit={handleLoad}>
            <p className="text-sm text-slate-600 mb-4">
              Dates are in Greece time. Matching is by equal amount only.
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
              className="px-4 py-2 rounded-lg bg-whatsapp-500 text-white font-medium transition hover:brightness-110 disabled:opacity-50"
            >
              {loading ? "Loading…" : "Load matches"}
            </button>

            {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
          </form>
        </ToolPanel>

        {summary && (
          <div className="flex flex-wrap gap-3 text-sm">
            <span className="rounded-xl border border-emerald-900/8 bg-white/80 px-3 py-2 backdrop-blur-sm">
              Accruals: <strong>{summary.accrualCount}</strong>
            </span>
            <span className="rounded-xl border border-emerald-900/8 bg-white/80 px-3 py-2 backdrop-blur-sm">
              Payments on this page: <strong>{summary.paymentCount}</strong>
            </span>
            <span className="rounded-xl border border-emerald-900/8 bg-white/80 px-3 py-2 backdrop-blur-sm">
              With candidates: <strong>{summary.withCandidates}</strong>
            </span>
            <span className="rounded-xl border border-emerald-900/8 bg-white/80 px-3 py-2 backdrop-blur-sm">
              No match: <strong>{summary.withoutCandidates}</strong>
            </span>
          </div>
        )}

        {result && accruals.length === 0 && !loading && (
          <p className="text-center text-slate-500 py-10">
            No active accruals in this period.
          </p>
        )}

        <div className="space-y-4">
          {accruals.map((accrual) => (
            <div
              key={accrual.id}
              className="rounded-2xl border border-emerald-900/8 bg-white/80 p-4 shadow-[0_1px_2px_rgba(6,78,59,0.04)] backdrop-blur-sm"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2 mb-1">
                <CashflowLink
                  row={accrual}
                  label={`Accrual ${accrual.id?.slice(-8) || ""}`}
                />
                <span className="text-base font-semibold text-emerald-950">
                  {formatAmount(accrual.amount)}
                </span>
              </div>
              <div className="text-xs text-slate-500 mb-2">
                Created: {accrual.createdAt || "—"}
              </div>
              <MetaLine row={accrual} />

              <div className="mt-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-emerald-800/55 mb-2">
                  Candidate payments ({accrual.candidates?.length || 0})
                </h3>
                <CandidateList candidates={accrual.candidates} />
              </div>
            </div>
          ))}
        </div>

        {pagination && totalPages > 1 && (
          <div className="flex justify-center items-center gap-2">
            <button
              type="button"
              onClick={() => loadMatches(currentPage - 1)}
              disabled={loading || currentPage <= 1}
              className="rounded-lg border border-emerald-900/10 bg-white/80 px-3 py-1.5 text-sm font-medium text-emerald-900 transition hover:border-whatsapp-500/40 disabled:opacity-50"
            >
              Prev
            </button>
            <span className="text-sm text-slate-600">
              Page {currentPage} of {totalPages}
              {pagination.totalCount != null && (
                <span className="text-slate-400">
                  {" "}
                  ({pagination.totalCount} accruals)
                </span>
              )}
            </span>
            <button
              type="button"
              onClick={() => loadMatches(currentPage + 1)}
              disabled={loading || currentPage >= totalPages}
              className="rounded-lg border border-emerald-900/10 bg-white/80 px-3 py-1.5 text-sm font-medium text-emerald-900 transition hover:border-whatsapp-500/40 disabled:opacity-50"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </PageShell>
  );
}
