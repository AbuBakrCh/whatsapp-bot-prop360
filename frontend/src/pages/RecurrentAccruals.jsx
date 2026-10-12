import React, { useCallback, useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import ContactSearchCombobox from "../components/ContactSearchCombobox";
import PropertySearchCombobox from "../components/PropertySearchCombobox";
import {
  createRecurrentAccrualSchedule,
  getRecurrentAccrualDocumentTypes,
  listRecurrentAccrualSchedules,
  updateRecurrentAccrualSchedule,
} from "../api";

const FREQUENCY_OPTIONS = [
  { value: "weekly", label: "Weekly" },
  { value: "monthly", label: "Monthly" },
  { value: "every_2_months", label: "Every 2 months" },
  { value: "quarterly", label: "Quarterly" },
  { value: "yearly", label: "Yearly" },
];

const CASHFLOW_URL =
  "https://prop360.pro/en/dashboard/forms/custom-a462rgbzo";

function formatDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export default function RecurrentAccruals() {
  const [tab, setTab] = useState("create");
  const [documentTypes, setDocumentTypes] = useState([]);
  const [trxDescriptions, setTrxDescriptions] = useState([]);

  const [ownerContact, setOwnerContact] = useState(null);
  const [properties, setProperties] = useState([]);
  const [documentType, setDocumentType] = useState("");
  const [trxDescription, setTrxDescription] = useState("");
  const [notes, setNotes] = useState("");
  const [amount, setAmount] = useState("");
  const [frequency, setFrequency] = useState("monthly");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [createLoading, setCreateLoading] = useState(false);
  const [createMessage, setCreateMessage] = useState("");

  const [filterOwner, setFilterOwner] = useState(null);
  const [schedules, setSchedules] = useState([]);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState("");
  const [expandedId, setExpandedId] = useState(null);
  const [actionLoadingId, setActionLoadingId] = useState(null);

  useEffect(() => {
    getRecurrentAccrualDocumentTypes().then((res) => {
      if (res.error) return;
      if (res.documentTypes?.length) {
        setDocumentTypes(res.documentTypes);
        setDocumentType((prev) => prev || res.documentTypes[0]);
      }
      if (res.trxDescriptions?.length) {
        setTrxDescriptions(res.trxDescriptions);
        setTrxDescription((prev) => prev || res.trxDescriptions[0]);
      }
    });
  }, []);

  const loadSchedules = useCallback(async () => {
    setReportLoading(true);
    setReportError("");
    const res = await listRecurrentAccrualSchedules(filterOwner?.pid || "");
    if (res.error) {
      setReportError(res.error);
      setSchedules([]);
    } else {
      setSchedules(res.schedules || []);
    }
    setReportLoading(false);
  }, [filterOwner]);

  useEffect(() => {
    if (tab === "report") {
      loadSchedules();
    }
  }, [tab, loadSchedules]);

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreateMessage("");

    if (!ownerContact) {
      setCreateMessage("Select a property owner contact.");
      return;
    }
    if (!properties.length) {
      setCreateMessage("Select at least one property.");
      return;
    }
    if (!startDate) {
      setCreateMessage("First accrual date is required.");
      return;
    }
    if (!trxDescription) {
      setCreateMessage("Select a Trx Description.");
      return;
    }
    const amountNum = parseFloat(amount);
    if (!Number.isFinite(amountNum) || amountNum <= 0) {
      setCreateMessage("Enter a valid amount greater than zero.");
      return;
    }

    setCreateLoading(true);
    const res = await createRecurrentAccrualSchedule({
      contact: ownerContact,
      properties,
      documentType,
      trxDescription,
      notes,
      amount: amountNum,
      frequency,
      startDate,
      endDate: endDate || null,
    });
    setCreateLoading(false);

    if (res.error) {
      setCreateMessage(res.error);
      return;
    }

    const immediate = res.immediate;
    if (immediate?.error) {
      setCreateMessage(
        `Schedule saved, but cashflow create failed: ${immediate.error}`
      );
    } else if (immediate?.cashflowsCreated > 0) {
      setCreateMessage(
        `Schedule created and ${immediate.cashflowsCreated} cashflow(s) created for today’s period.`
      );
    } else {
      setCreateMessage(
        "Schedule created. Accruals will be inserted on due dates via the daily job."
      );
    }
    setOwnerContact(null);
    setProperties([]);
    setNotes("");
    setAmount("");
    setEndDate("");
  };

  const handleStatusChange = async (scheduleId, status) => {
    setActionLoadingId(scheduleId);
    const res = await updateRecurrentAccrualSchedule(scheduleId, { status });
    setActionLoadingId(null);
    if (res.error) {
      setReportError(res.error);
      return;
    }
    await loadSchedules();
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex gap-2 mb-6 border-b border-slate-200">
        <button
          type="button"
          onClick={() => setTab("create")}
          className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
            tab === "create"
              ? "border-green-600 text-green-700"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
        >
          Create schedule
        </button>
        <button
          type="button"
          onClick={() => setTab("report")}
          className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
            tab === "report"
              ? "border-green-600 text-green-700"
              : "border-transparent text-gray-500 hover:text-gray-700"
          }`}
        >
          Report
        </button>
      </div>

      {tab === "create" && (
        <form
          onSubmit={handleCreate}
          className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm space-y-5"
        >
          <p className="text-sm text-gray-600">
            Creates one Debit Accrual cashflow per property on each due date. Cashflows
            are owned by Kostas and tagged as console recurrent.
          </p>

          <ContactSearchCombobox
            selectedContact={ownerContact}
            onChange={setOwnerContact}
          />

          <PropertySearchCombobox
            selectedProperties={properties}
            onChange={setProperties}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Document type
              </label>
              <select
                value={documentType}
                onChange={(e) => setDocumentType(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              >
                {documentTypes.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Amount (EUR)
              </label>
              <input
                type="number"
                min="0"
                step="0.01"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Trx Description
            </label>
            <select
              value={trxDescription}
              onChange={(e) => setTrxDescription(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              required
            >
              {trxDescriptions.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Invest Greece notes
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              rows={3}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              placeholder="Optional notes"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Frequency
              </label>
              <select
                value={frequency}
                onChange={(e) => setFrequency(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              >
                {FREQUENCY_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                First accrual date
              </label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                End date (optional)
              </label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={createLoading}
            className="px-5 py-2 rounded-lg bg-green-600 text-white text-sm font-medium hover:bg-green-700 disabled:opacity-50"
          >
            {createLoading ? "Saving…" : "Create schedule"}
          </button>

          {createMessage && (
            <p
              className={`text-sm ${
                createMessage.toLowerCase().includes("fail") ||
                createMessage.includes("Select") ||
                createMessage.includes("Enter")
                  ? "text-red-600"
                  : "text-green-700"
              }`}
            >
              {createMessage}
            </p>
          )}
        </form>
      )}

      {tab === "report" && (
        <div className="space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
            <ContactSearchCombobox
              selectedContact={filterOwner}
              onChange={setFilterOwner}
              label="Filter by property owner"
            />
            <div className="flex gap-2 mt-3">
              <button
                type="button"
                onClick={loadSchedules}
                disabled={reportLoading}
                className="px-4 py-2 text-sm bg-slate-800 text-white rounded-lg hover:bg-slate-900 disabled:opacity-50"
              >
                {reportLoading ? "Loading…" : "Refresh"}
              </button>
              {filterOwner && (
                <button
                  type="button"
                  onClick={() => setFilterOwner(null)}
                  className="px-4 py-2 text-sm border border-slate-300 rounded-lg"
                >
                  Clear filter
                </button>
              )}
            </div>
          </div>

          {reportError && <p className="text-sm text-red-600">{reportError}</p>}

          <div className="overflow-x-auto bg-white rounded-xl border border-slate-200 shadow-sm">
            <table className="w-full text-sm">
              <thead className="bg-slate-100 text-left">
                <tr>
                  <th className="px-3 py-2">Owner</th>
                  <th className="px-3 py-2">Properties</th>
                  <th className="px-3 py-2">Type</th>
                  <th className="px-3 py-2">Amount</th>
                  <th className="px-3 py-2">Frequency</th>
                  <th className="px-3 py-2">Next due</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {schedules.length === 0 && !reportLoading && (
                  <tr>
                    <td colSpan={8} className="px-3 py-8 text-center text-gray-400">
                      No schedules found
                    </td>
                  </tr>
                )}
                {schedules.map((row) => {
                  const propLabels = (row.properties || [])
                    .map((p) => p.label)
                    .join(", ");
                  const isExpanded = expandedId === row.id;
                  const busy = actionLoadingId === row.id;
                  return (
                    <React.Fragment key={row.id}>
                      <tr className="border-t border-slate-100">
                        <td className="px-3 py-2 align-top">
                          <div className="font-medium">{row.contact?.label}</div>
                          <div className="text-xs text-gray-500">
                            PID {row.contact?.pid}
                          </div>
                        </td>
                        <td className="px-3 py-2 align-top max-w-xs">
                          <button
                            type="button"
                            onClick={() =>
                              setExpandedId(isExpanded ? null : row.id)
                            }
                            className="text-left text-blue-600 hover:underline"
                          >
                            {(row.properties || []).length} selected
                          </button>
                          <p className="text-xs text-gray-500 truncate">{propLabels}</p>
                        </td>
                        <td className="px-3 py-2 align-top">
                          <div>{row.documentType}</div>
                          <div className="text-xs text-gray-500">
                            {row.trxDescription || "—"}
                          </div>
                        </td>
                        <td className="px-3 py-2 align-top">{row.amount}</td>
                        <td className="px-3 py-2 align-top">{row.frequency}</td>
                        <td className="px-3 py-2 align-top">{row.nextDueDate}</td>
                        <td className="px-3 py-2 align-top capitalize">{row.status}</td>
                        <td className="px-3 py-2 align-top">
                          <div className="flex flex-wrap gap-1">
                            {row.status === "active" && (
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleStatusChange(row.id, "paused")}
                                className="px-2 py-1 text-xs border rounded hover:bg-slate-50"
                              >
                                Pause
                              </button>
                            )}
                            {row.status === "paused" && (
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleStatusChange(row.id, "active")}
                                className="px-2 py-1 text-xs border rounded hover:bg-green-50 text-green-800"
                              >
                                Resume
                              </button>
                            )}
                            {row.status !== "stopped" && (
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleStatusChange(row.id, "stopped")}
                                className="px-2 py-1 text-xs border border-red-200 text-red-700 rounded hover:bg-red-50"
                              >
                                Stop
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr className="bg-slate-50">
                          <td colSpan={8} className="px-4 py-3">
                            <p className="text-xs text-gray-500 mb-2">
                              Start {row.startDate}
                              {row.endDate ? ` · End ${row.endDate}` : ""}
                              {row.lastRunAt
                                ? ` · Last run ${formatDate(row.lastRunAt)}`
                                : ""}
                            </p>
                            {row.notes ? (
                              <p className="text-sm text-gray-700 mb-2">
                                Notes: {row.notes}
                              </p>
                            ) : null}
                            <ul className="space-y-1 text-sm">
                              {(row.generated || []).length === 0 && (
                                <li className="text-gray-400">No cashflows generated yet</li>
                              )}
                              {(row.generated || []).map((g, idx) => (
                                <li key={`${g.cashflowId}-${idx}`} className="flex items-center gap-2">
                                  <span>
                                    {g.period} · {g.propertyLabel || g.propertyPid} ·{" "}
                                    {g.cashflowId}
                                  </span>
                                  {g.cashflowId && (
                                    <a
                                      href={`${CASHFLOW_URL}/${g.cashflowId}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="inline-flex text-blue-600"
                                    >
                                      <ExternalLink className="w-4 h-4" />
                                    </a>
                                  )}
                                </li>
                              ))}
                            </ul>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
