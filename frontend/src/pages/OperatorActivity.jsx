import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  getOperatorActivity,
  getOperatorActivityDetail,
} from "../api";

const VIEWS = [
  { value: "day", label: "Day" },
  { value: "week", label: "Week" },
  { value: "month", label: "Month" },
];

function todayAthensISO() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/Athens",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());
}

export default function OperatorActivity() {
  const [view, setView] = useState("day");
  const [date, setDate] = useState(() => todayAthensISO());
  const [rankings, setRankings] = useState([]);
  const [summary, setSummary] = useState(null);
  const [period, setPeriod] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [searchInput, setSearchInput] = useState("");

  const [modalOpen, setModalOpen] = useState(false);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState("");
  const [modalOperator, setModalOperator] = useState(null);
  const [modalEvents, setModalEvents] = useState([]);
  const [modalTruncated, setModalTruncated] = useState(false);

  const fetchReport = useCallback(async () => {
    setIsLoading(true);
    setError("");
    const res = await getOperatorActivity({ view, date });
    setIsLoading(false);

    if (res.error) {
      setError(typeof res.error === "string" ? res.error : "Failed to load");
      setRankings([]);
      setSummary(null);
      setPeriod(null);
      return;
    }

    setRankings(res.rankings || []);
    setSummary(res.summary || null);
    setPeriod(res.period || null);
  }, [view, date]);

  useEffect(() => {
    fetchReport();
  }, [fetchReport]);

  const filteredRows = useMemo(() => {
    const q = searchInput.trim().toLowerCase();
    if (!q) return rankings;
    return rankings.filter((row) => {
      const name = (row.displayName || "").toLowerCase();
      const email = (row.email || "").toLowerCase();
      return name.includes(q) || email.includes(q);
    });
  }, [rankings, searchInput]);

  const openDetail = async (row) => {
    setModalOpen(true);
    setModalOperator(row);
    setModalLoading(true);
    setModalError("");
    setModalEvents([]);
    setModalTruncated(false);

    const res = await getOperatorActivityDetail(row.firebaseId, { view, date });
    setModalLoading(false);

    if (res.error) {
      setModalError(
        typeof res.error === "string" ? res.error : "Failed to load detail"
      );
      return;
    }

    setModalOperator(res.operator || row);
    setModalEvents(res.events || []);
    setModalTruncated(Boolean(res.truncated));
  };

  const closeModal = () => {
    setModalOpen(false);
    setModalOperator(null);
    setModalEvents([]);
    setModalError("");
  };

  const top = summary?.topOperator;

  return (
    <div className="max-w-5xl mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between mb-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-800">
            Operator Activity
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Timezone shown in Greece timezone.
          </p>
          {period?.label ? (
            <p className="text-sm text-slate-600 mt-1 font-medium">
              {period.label}
            </p>
          ) : null}
        </div>

        <div className="flex flex-col sm:flex-row gap-2 items-stretch sm:items-center">
          <div className="inline-flex rounded-md border border-slate-300 overflow-hidden">
            {VIEWS.map((opt) => (
              <button
                key={opt.value}
                type="button"
                onClick={() => setView(opt.value)}
                className={`px-3 py-2 text-sm ${
                  view === opt.value
                    ? "bg-blue-600 text-white"
                    : "bg-white text-gray-700 hover:bg-slate-50"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 text-sm bg-white"
          />
        </div>
      </div>

      {summary ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
          <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500">
              Total active
            </div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {summary.totalDurationLabel || "0m"}
            </div>
          </div>
          <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500">
              With activity
            </div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {summary.operatorsWithActivity ?? 0}
              <span className="text-sm font-medium text-slate-500">
                {" "}
                / {summary.totalOperators ?? 0}
              </span>
            </div>
          </div>
          <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500">
              Zero activity
            </div>
            <div className="text-xl font-bold text-slate-800 mt-1">
              {summary.operatorsWithZero ?? 0}
            </div>
          </div>
          <div className="rounded-xl bg-slate-50 border border-slate-200 px-4 py-3">
            <div className="text-xs uppercase tracking-wide text-slate-500">
              Top operator
            </div>
            <div className="text-base font-bold text-slate-800 mt-1 truncate">
              {top?.displayName || "—"}
            </div>
            <div className="text-xs text-slate-500">
              {top?.durationLabel || ""}
            </div>
          </div>
        </div>
      ) : null}

      <div className="flex flex-col sm:flex-row gap-2 mb-4">
        <input
          type="search"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Search operator by name or email"
          className="flex-1 border border-slate-300 rounded-md px-3 py-2 text-sm"
        />
        <button
          type="button"
          onClick={() => setSearchInput("")}
          disabled={!searchInput}
          className="px-3 py-2 rounded-md text-sm bg-gray-200 hover:bg-gray-300 disabled:opacity-50"
        >
          Clear
        </button>
      </div>

      {error && <p className="text-sm text-red-600 mb-3">{error}</p>}

      {isLoading ? (
        <p className="text-gray-500">Loading...</p>
      ) : filteredRows.length === 0 ? (
        <p className="text-gray-500">No operators found for this period.</p>
      ) : (
        <div className="overflow-x-auto border border-slate-200 rounded-lg">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-left text-gray-600">
              <tr>
                <th className="px-3 py-3 font-semibold w-12 text-center">#</th>
                <th className="px-4 py-3 font-semibold">Operator</th>
                <th className="px-4 py-3 font-semibold text-right whitespace-nowrap">
                  Active
                </th>
                <th className="px-4 py-3 font-semibold min-w-[140px]"> </th>
                <th className="px-4 py-3 font-semibold text-right">Logins</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => (
                <tr
                  key={row.firebaseId}
                  onClick={() => openDetail(row)}
                  className="border-t border-slate-100 hover:bg-slate-50 cursor-pointer"
                >
                  <td
                    className={`px-3 py-3 text-center font-semibold ${
                      row.rank <= 3 ? "text-blue-700" : "text-slate-700"
                    }`}
                  >
                    {row.rank}
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      {row.photoURL ? (
                        <img
                          src={row.photoURL}
                          alt=""
                          className="w-8 h-8 rounded-full object-cover bg-slate-100"
                        />
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-xs font-semibold text-slate-600">
                          {(row.displayName || "?").slice(0, 1).toUpperCase()}
                        </div>
                      )}
                      <div>
                        <div className="font-medium text-gray-800">
                          {row.displayName || "Unknown"}
                        </div>
                        {row.email ? (
                          <div className="text-xs text-gray-500">
                            {row.email}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums font-medium whitespace-nowrap">
                    {row.durationLabel}
                  </td>
                  <td className="px-4 py-3">
                    <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
                      <div
                        className="h-2 rounded-full bg-blue-700"
                        style={{ width: `${row.barPct || 0}%` }}
                      />
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {row.loginCount ?? 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!isLoading && rankings.length > 0 ? (
        <p className="text-xs text-slate-500 mt-3">
          Showing {filteredRows.length} of {rankings.length} active users.
          Click a row for session timeline.
        </p>
      ) : null}

      {modalOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-lg w-full max-w-2xl max-h-[85vh] flex flex-col relative">
            <button
              type="button"
              onClick={closeModal}
              className="absolute top-3 right-3 text-gray-500 hover:text-gray-800 text-xl leading-none"
              aria-label="Close"
            >
              ×
            </button>
            <div className="p-5 border-b border-slate-200 pr-10">
              <h3 className="text-lg font-semibold text-gray-800">
                {modalOperator?.displayName || "Operator"}
              </h3>
              {modalOperator?.email ? (
                <p className="text-sm text-gray-500">{modalOperator.email}</p>
              ) : null}
              <p className="text-sm text-slate-600 mt-1">
                {modalOperator?.durationLabel || "0m"} active
                {typeof modalOperator?.loginCount === "number"
                  ? ` · ${modalOperator.loginCount} logins`
                  : ""}
                {period?.label ? ` · ${period.label}` : ""}
              </p>
            </div>
            <div className="p-5 overflow-y-auto">
              {modalLoading ? (
                <p className="text-gray-500">Loading...</p>
              ) : modalError ? (
                <p className="text-sm text-red-600">{modalError}</p>
              ) : modalEvents.length === 0 ? (
                <p className="text-gray-500">
                  No session events in this period.
                </p>
              ) : (
                <ul className="space-y-3">
                  {modalEvents.map((ev) => (
                    <li
                      key={ev.id}
                      className="border border-slate-200 rounded-lg px-3 py-2"
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <span
                            className={`inline-block text-xs font-semibold px-2 py-0.5 rounded ${
                              ev.action === "login"
                                ? "bg-emerald-100 text-emerald-800"
                                : "bg-blue-100 text-blue-800"
                            }`}
                          >
                            {ev.action}
                          </span>
                          <div className="text-sm text-gray-800 mt-1">
                            {ev.description || "—"}
                          </div>
                          {ev.activeMinutes != null ? (
                            <div className="text-xs text-slate-500 mt-0.5">
                              {ev.activeMinutes} minutes
                            </div>
                          ) : null}
                          {ev.ip ? (
                            <div className="text-xs text-slate-400 mt-0.5">
                              IP {ev.ip}
                            </div>
                          ) : null}
                        </div>
                        <div className="text-xs text-slate-500 whitespace-nowrap text-right">
                          {ev.createdAtLabel || ev.createdAt || "—"}
                        </div>
                      </div>
                    </li>
                  ))}
                </ul>
              )}
              {modalTruncated ? (
                <p className="text-xs text-amber-700 mt-3">
                  Showing the most recent 500 events only.
                </p>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
