import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  getIncompleteCashflows,
  getIncompleteCashflowsForUser,
} from "../api";

const LIMIT = 20;
const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function buildPeriodOptions() {
  const now = new Date();
  const year = 2026;
  const currentMonth =
    now.getFullYear() === year ? now.getMonth() + 1 : now.getFullYear() > year ? 12 : 0;

  const months = [];
  for (let m = 1; m <= currentMonth; m += 1) {
    const value = `${year}-${String(m).padStart(2, "0")}`;
    months.push({
      value,
      label: `${MONTH_LABELS[m - 1]} ${String(year).slice(2)}`,
    });
  }

  return [
    { value: "yesterday", label: "Yesterday" },
    ...months,
    { value: "all", label: "All time" },
  ];
}

function matchesSearch(row, query) {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const name = (row.userName || "").toLowerCase();
  const email = (row.email || "").toLowerCase();
  return name.includes(q) || email.includes(q);
}

export default function IncompleteCashflows() {
  const periodOptions = useMemo(() => buildPeriodOptions(), []);
  const [period, setPeriod] = useState("yesterday");
  const [rows, setRows] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [totals, setTotals] = useState({ contactCount: 0, propertyCount: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [serverQuery, setServerQuery] = useState("");
  const [clientFiltered, setClientFiltered] = useState(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState("");
  const [modalUser, setModalUser] = useState(null);
  const [modalCashflows, setModalCashflows] = useState([]);
  const [modalTruncated, setModalTruncated] = useState(false);

  const displayRows = clientFiltered !== null ? clientFiltered : rows;
  const totalPages = Math.max(1, Math.ceil(total / LIMIT));

  const fetchPage = useCallback(
    async (pageNumber = 1, query = "") => {
      setIsLoading(true);
      setError("");
      setClientFiltered(null);

      const res = await getIncompleteCashflows({
        period,
        page: pageNumber,
        pageSize: LIMIT,
        q: query,
      });

      setIsLoading(false);

      if (res.error) {
        setError(typeof res.error === "string" ? res.error : "Failed to load");
        setRows([]);
        setTotal(0);
        setTotals({ contactCount: 0, propertyCount: 0 });
        return;
      }

      setRows(res.data || []);
      setTotal(res.total || 0);
      setPage(res.page || pageNumber);
      setTotals(res.totals || { contactCount: 0, propertyCount: 0 });
      setServerQuery(query || "");
    },
    [period]
  );

  useEffect(() => {
    fetchPage(1, "");
    setSearchInput("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [period]);

  const handleSearch = async (e) => {
    e?.preventDefault?.();
    const q = searchInput.trim();

    if (!q) {
      await fetchPage(1, "");
      return;
    }

    const localMatches = rows.filter((row) => matchesSearch(row, q));
    if (localMatches.length > 0) {
      setClientFiltered(localMatches);
      return;
    }

    await fetchPage(1, q);
  };

  const handleClearSearch = async () => {
    setSearchInput("");
    setClientFiltered(null);
    if (serverQuery) {
      await fetchPage(1, "");
    }
  };

  const openUserModal = async (row) => {
    setModalOpen(true);
    setModalUser(row);
    setModalLoading(true);
    setModalError("");
    setModalCashflows([]);
    setModalTruncated(false);

    const res = await getIncompleteCashflowsForUser(row.userId, period);
    setModalLoading(false);

    if (res.error) {
      setModalError(
        typeof res.error === "string" ? res.error : "Failed to load cashflows"
      );
      return;
    }

    setModalUser({
      userId: res.userId,
      userName: res.userName || row.userName,
      email: res.email || row.email,
    });
    setModalCashflows(res.cashflows || []);
    setModalTruncated(Boolean(res.truncated));
  };

  const closeModal = () => {
    setModalOpen(false);
    setModalUser(null);
    setModalCashflows([]);
    setModalError("");
  };

  return (
    <div className="max-w-4xl mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">
            Incomplete Cashflows
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Agents with cashflows missing contact/owner and/or property
          </p>
        </div>

        <select
          value={period}
          onChange={(e) => setPeriod(e.target.value)}
          className="border border-slate-300 rounded-md px-3 py-2 text-sm bg-white"
        >
          {periodOptions.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      <form
        onSubmit={handleSearch}
        className="flex flex-col sm:flex-row gap-2 mb-4"
      >
        <input
          type="search"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Search user by name or email"
          className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
        <div className="flex gap-2">
          <button
            type="submit"
            disabled={isLoading}
            className="px-3 py-2 rounded-md text-sm text-white bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300"
          >
            Search
          </button>
          <button
            type="button"
            onClick={handleClearSearch}
            disabled={isLoading}
            className="px-3 py-2 rounded-md text-sm bg-gray-200 hover:bg-gray-300 disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </form>

      {clientFiltered !== null && (
        <p className="text-xs text-amber-700 mb-2">
          Showing client-side matches on the current page. Clear search to
          reset.
        </p>
      )}
      {serverQuery && clientFiltered === null && (
        <p className="text-xs text-slate-500 mb-2">
          Server search: &ldquo;{serverQuery}&rdquo;
        </p>
      )}

      {error && (
        <p className="text-sm text-red-600 mb-3">{error}</p>
      )}

      {isLoading ? (
        <p className="text-gray-500">Loading...</p>
      ) : displayRows.length === 0 ? (
        <p className="text-gray-500">No incomplete cashflows found.</p>
      ) : (
        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-gray-600">
              <tr>
                <th className="px-4 py-3 font-semibold">User</th>
                <th className="px-4 py-3 font-semibold text-right">Contact</th>
                <th className="px-4 py-3 font-semibold text-right">Property</th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map((row) => (
                <tr
                  key={row.userId}
                  onClick={() => openUserModal(row)}
                  className="border-t border-slate-100 hover:bg-slate-50 cursor-pointer"
                >
                  <td className="px-4 py-3">
                    <div className="font-medium text-gray-800">
                      {row.userName || "Unknown"}
                    </div>
                    {row.email ? (
                      <div className="text-xs text-gray-500">{row.email}</div>
                    ) : null}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {row.contactCount}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {row.propertyCount}
                  </td>
                </tr>
              ))}
            </tbody>
            {clientFiltered === null && (
              <tfoot>
                <tr className="border-t-2 border-slate-200 bg-slate-50 font-semibold">
                  <td className="px-4 py-3">Total</td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {totals.contactCount}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {totals.propertyCount}
                  </td>
                </tr>
              </tfoot>
            )}
          </table>
        </div>
      )}

      {clientFiltered === null && (
        <div className="mt-6 flex flex-col items-center gap-2">
          <div className="text-sm text-gray-600">
            {total === 0
              ? "No records"
              : `Showing ${(page - 1) * LIMIT + 1} - ${Math.min(
                  page * LIMIT,
                  total
                )} of ${total} users`}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => fetchPage(1, serverQuery)}
              disabled={page === 1 || isLoading}
              className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
            >
              First
            </button>
            <button
              onClick={() => fetchPage(page - 1, serverQuery)}
              disabled={page === 1 || isLoading}
              className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
            >
              Prev
            </button>
            <span className="px-3 text-sm">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => fetchPage(page + 1, serverQuery)}
              disabled={page >= totalPages || isLoading}
              className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
            >
              Next
            </button>
            <button
              onClick={() => fetchPage(totalPages, serverQuery)}
              disabled={page >= totalPages || isLoading}
              className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
            >
              Last
            </button>
          </div>
        </div>
      )}

      {modalOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-lg w-full max-w-2xl max-h-[85vh] flex flex-col relative">
            <div className="flex items-start justify-between gap-4 p-6 border-b border-slate-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  {modalUser?.userName || "Unknown"}
                </h3>
                {modalUser?.email ? (
                  <p className="text-sm text-gray-500">{modalUser.email}</p>
                ) : null}
              </div>
              <button
                type="button"
                onClick={closeModal}
                className="text-gray-500 hover:text-gray-800 text-sm px-2 py-1"
              >
                Close
              </button>
            </div>

            <div className="p-6 overflow-y-auto">
              {modalLoading ? (
                <p className="text-gray-500 text-sm">Loading cashflows...</p>
              ) : modalError ? (
                <p className="text-red-600 text-sm">{modalError}</p>
              ) : modalCashflows.length === 0 ? (
                <p className="text-gray-500 text-sm">No cashflows found.</p>
              ) : (
                <div className="space-y-3">
                  {modalCashflows.map((cashflow) => (
                    <div
                      key={cashflow.id}
                      className="border border-slate-200 rounded-lg p-3 bg-slate-50"
                    >
                      <div className="flex flex-wrap items-center gap-2 mb-2">
                        <span className="text-xs text-gray-500">
                          {cashflow.date || "No date"}
                        </span>
                        {cashflow.missingContact && (
                          <span className="text-xs px-2 py-0.5 rounded bg-amber-100 text-amber-800">
                            No contact
                          </span>
                        )}
                        {cashflow.missingProperty && (
                          <span className="text-xs px-2 py-0.5 rounded bg-rose-100 text-rose-800">
                            No property
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-800 mb-2 line-clamp-3">
                        {cashflow.description || "(no description)"}
                      </p>
                      <a
                        href={cashflow.url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-sm text-blue-600 hover:underline"
                      >
                        Open cashflow
                      </a>
                    </div>
                  ))}
                  {modalTruncated && (
                    <p className="text-xs text-amber-700">
                      Showing first 500 cashflows only.
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
