import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ExternalLink } from "lucide-react";
import ContactSearchCombobox from "../components/ContactSearchCombobox";
import {
  createBankImportAccount,
  createBankImportContact,
  createBankImportPayee,
  deleteBankImport,
  deleteBankImportPayee,
  getBankImport,
  getBankImportTrxDescriptions,
  listBankImportAccounts,
  listBankImportPayees,
  listBankImports,
  patchBankImport,
  saveBankImport,
  uploadBankImport,
  searchProperties,
} from "../api";

const STATUS_FILTERS = [
  { value: "all", label: "All" },
  { value: "to_choose", label: "To choose" },
  { value: "possible_double", label: "Possible double" },
  { value: "new", label: "New" },
  { value: "fee", label: "Fee" },
  { value: "skip", label: "Skip" },
  { value: "ignored", label: "Ignored" },
  { value: "saved", label: "Saved" },
];

const AUX_PROPERTY =
  "AUX - İT İS NOT RELATED TO ANY PROPERTY - HER HANGİ BİR MÜLK İLE BAĞLANTISI YOK - DEN EİNAİ SİNDEDEMENO ME KAPİO AKİNİTO|4626743390357804";

function parsePiped(value) {
  if (!value) return null;
  const text = String(value);
  if (!text.includes("|")) return { label: text, pid: "" };
  const idx = text.lastIndexOf("|");
  return {
    label: text.slice(0, idx).trim(),
    pid: text.slice(idx + 1).trim(),
  };
}

function toPiped(contactOrProp) {
  if (!contactOrProp) return "";
  return `${contactOrProp.label}|${contactOrProp.pid}`;
}

function statusBadgeClass(status) {
  switch (status) {
    case "fee":
      return "bg-slate-100 text-slate-700";
    case "skip":
    case "ignored":
      return "bg-gray-200 text-gray-600";
    case "possible_double":
      return "bg-amber-100 text-amber-800";
    case "to_choose":
      return "bg-rose-100 text-rose-800";
    case "saved":
      return "bg-emerald-100 text-emerald-800";
    default:
      return "bg-sky-100 text-sky-800";
  }
}

function PropertySingleCombobox({ value, onChange, disabled }) {
  const selected = parsePiped(value);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      const res = await searchProperties(trimmed, 12);
      if (!res.error) setResults(res.data || []);
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  if (disabled) {
    return (
      <div className="text-xs text-gray-700 whitespace-normal break-words min-w-[12rem] max-w-[22rem]">
        {selected?.label || "—"}
      </div>
    );
  }

  if (selected?.pid) {
    return (
      <div className="flex items-start gap-1 min-w-[12rem] max-w-[22rem]">
        <span className="text-xs whitespace-normal break-words flex-1">
          {selected.label}
        </span>
        <button
          type="button"
          className="text-xs text-red-600 shrink-0"
          onClick={() => onChange("")}
        >
          ×
        </button>
      </div>
    );
  }

  return (
    <div className="relative min-w-[12rem] max-w-[22rem]">
      <input
        className="w-full border rounded px-1 py-0.5 text-xs"
        placeholder="Search property…"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
      />
      <button
        type="button"
        className="mt-0.5 text-[10px] text-emerald-700 underline"
        onClick={() => onChange(AUX_PROPERTY)}
      >
        Use AUX
      </button>
      {open && results.length > 0 && (
        <ul className="absolute z-20 mt-1 max-h-48 w-72 overflow-auto rounded border bg-white shadow">
          {results.map((p) => (
            <li key={p.pid}>
              <button
                type="button"
                className="block w-full px-2 py-1 text-left text-xs whitespace-normal break-words hover:bg-emerald-50"
                onClick={() => {
                  onChange(toPiped(p));
                  setQuery("");
                  setOpen(false);
                }}
              >
                {p.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ContactCell({ value, onChange, disabled, label }) {
  const selected = parsePiped(value);
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("");

  if (disabled) {
    return (
      <div className="text-xs text-gray-700 whitespace-normal break-words min-w-[10rem] max-w-[18rem]">
        {selected?.label || "—"}
      </div>
    );
  }

  return (
    <div className="min-w-[10rem] max-w-[18rem] space-y-1">
      <ContactSearchCombobox
        label={label}
        selectedContact={selected?.pid ? selected : null}
        onChange={(c) => onChange(c ? toPiped(c) : "")}
      />
      {!selected?.pid && (
        <div className="flex gap-1">
          {creating ? (
            <>
              <input
                className="flex-1 border rounded px-1 text-xs"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="New contact name"
              />
              <button
                type="button"
                className="text-xs text-emerald-700"
                onClick={async () => {
                  const res = await createBankImportContact(newName.trim());
                  if (res.error) return;
                  onChange(toPiped(res.contact));
                  setCreating(false);
                  setNewName("");
                }}
              >
                Add
              </button>
            </>
          ) : (
            <button
              type="button"
              className="text-[10px] text-emerald-700 underline"
              onClick={() => setCreating(true)}
            >
              Create contact
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default function BankStatementImport() {
  const [tab, setTab] = useState("import");
  const [accounts, setAccounts] = useState([]);
  const [accountId, setAccountId] = useState("");
  const [file, setFile] = useState(null);
  const [trxDescriptions, setTrxDescriptions] = useState([]);
  const [session, setSession] = useState(null);
  const [imports, setImports] = useState([]);
  const [payees, setPayees] = useState([]);
  const [statusFilter, setStatusFilter] = useState("all");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [saveResult, setSaveResult] = useState(null);

  const [newAccount, setNewAccount] = useState({
    key: "",
    label: "",
    trxBank: "",
    accountNo: "",
    holderLabel: "",
    holderPid: "",
  });
  const [newPayee, setNewPayee] = useState({
    match: "",
    category: "",
    receiverLabel: "",
    receiverPid: "",
  });

  const loadAccounts = useCallback(async () => {
    const res = await listBankImportAccounts();
    if (!res.error) {
      setAccounts(res.accounts || []);
      if (!accountId && res.accounts?.length) {
        setAccountId(res.accounts[0].id);
      }
    }
  }, [accountId]);

  const loadMeta = useCallback(async () => {
    const [desc, payeeRes, listRes] = await Promise.all([
      getBankImportTrxDescriptions(),
      listBankImportPayees(),
      listBankImports(),
    ]);
    if (!desc.error) setTrxDescriptions(desc.trxDescriptions || []);
    if (!payeeRes.error) setPayees(payeeRes.payees || []);
    if (!listRes.error) setImports(listRes.imports || []);
  }, []);

  useEffect(() => {
    loadAccounts();
    loadMeta();
  }, [loadAccounts, loadMeta]);

  const filteredRows = useMemo(() => {
    const rows = session?.rows || [];
    if (statusFilter === "all") return rows;
    return rows.filter((r) => r.status === statusFilter);
  }, [session, statusFilter]);

  const handleUpload = async (e) => {
    e.preventDefault();
    setMessage("");
    setSaveResult(null);
    if (!accountId) {
      setMessage("Select an account.");
      return;
    }
    if (!file) {
      setMessage("Choose an Optima Excel file.");
      return;
    }
    setLoading(true);
    const res = await uploadBankImport(accountId, file);
    setLoading(false);
    if (res.error) {
      setMessage(res.error);
      return;
    }
    setSession(res.import);
    setTab("import");
    setMessage(
      `Loaded ${res.import.rowCount} lines for ${res.import.accountLabel} (${res.import.period}).`
    );
    loadMeta();
  };

  const applyRowPatch = async (patch) => {
    if (!session?.id) return;
    const res = await patchBankImport(session.id, { rows: [patch] });
    if (res.error) {
      setMessage(res.error);
      return;
    }
    setSession(res.import);
  };

  const confirmAllSuggested = async () => {
    if (!session?.id) return;
    setLoading(true);
    const res = await patchBankImport(session.id, { confirmAllSuggested: true });
    setLoading(false);
    if (res.error) {
      setMessage(res.error);
      return;
    }
    setSession(res.import);
    setMessage("Confirmed all suggested lines that have judgement fields filled.");
  };

  const handleSave = async () => {
    if (!session?.id) return;
    setLoading(true);
    setMessage("");
    const res = await saveBankImport(session.id);
    setLoading(false);
    if (res.error) {
      setMessage(res.error);
      return;
    }
    setSession(res.import);
    setSaveResult(res.saveResult || res.import?.saveResult);
    const stats = res.saveResult || res.import?.saveResult || {};
    setMessage(
      `Saved: created ${stats.created ?? 0}, skipped ${stats.skipped ?? 0}, left open ${stats.leftOpen ?? 0}.`
    );
    loadMeta();
  };

  const reopenImport = async (id) => {
    setLoading(true);
    const res = await getBankImport(id);
    setLoading(false);
    if (res.error) {
      setMessage(res.error);
      return;
    }
    setSession(res.import);
    setSaveResult(res.import?.saveStats || null);
    setTab("import");
  };

  const handleDeleteImport = async (id) => {
    if (!window.confirm("Delete this import session? Saved Prop360 cashflows are kept.")) {
      return;
    }
    setLoading(true);
    const res = await deleteBankImport(id);
    setLoading(false);
    if (res.error) {
      setMessage(res.error);
      return;
    }
    if (session?.id === id) {
      setSession(null);
      setSaveResult(null);
    }
    setMessage("Import session deleted.");
    loadMeta();
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 border-b border-gray-200 pb-2">
        {[
          ["import", "Import"],
          ["sessions", "Sessions"],
          ["accounts", "Accounts"],
          ["payees", "Known payees"],
        ].map(([id, label]) => (
          <button
            key={id}
            type="button"
            onClick={() => setTab(id)}
            className={`px-3 py-1.5 rounded-md text-sm font-medium ${
              tab === id
                ? "bg-emerald-600 text-white"
                : "bg-white text-gray-700 border border-gray-200"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {message && (
        <div className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-900">
          {message}
        </div>
      )}

      {tab === "import" && (
        <div className="space-y-4">
          <form
            onSubmit={handleUpload}
            className="grid gap-3 rounded-xl border border-slate-200 bg-white p-4 sm:grid-cols-3"
          >
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Account
              </label>
              <select
                className="w-full border rounded-md px-2 py-2 text-sm"
                value={accountId}
                onChange={(e) => setAccountId(e.target.value)}
              >
                {accounts.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.label} (…{a.accountNo.slice(-4)})
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Optima Excel (.xlsx)
              </label>
              <input
                type="file"
                accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                className="w-full text-sm"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
            </div>
            <div className="flex items-end">
              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
              >
                {loading ? "Working…" : "Upload & preview"}
              </button>
            </div>
          </form>

          {session && (
            <>
              <div className="flex flex-wrap items-center gap-3 text-sm text-gray-700">
                <span className="font-medium">{session.filename}</span>
                <span>{session.period}</span>
                <span className="text-gray-500">
                  {session.counts
                    ? Object.entries(session.counts)
                        .filter(([k]) => k !== "total")
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(" · ")
                    : null}
                </span>
                {session.integrity?.runningBalanceOk === false && (
                  <span className="text-amber-700">Running balance check failed</span>
                )}
              </div>

              <div className="flex flex-wrap gap-2 items-center">
                <select
                  className="border rounded-md px-2 py-1 text-sm"
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                >
                  {STATUS_FILTERS.map((f) => (
                    <option key={f.value} value={f.value}>
                      {f.label}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={confirmAllSuggested}
                  className="rounded-md border border-emerald-300 px-3 py-1 text-sm text-emerald-800 hover:bg-emerald-50"
                >
                  Confirm all suggested
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={loading}
                  className="rounded-md bg-emerald-600 px-3 py-1 text-sm font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
                >
                  Save ready lines
                </button>
                {saveResult && (
                  <span className="text-sm text-gray-600">
                    Last save — created {saveResult.created}, skipped{" "}
                    {saveResult.skipped}, left open {saveResult.leftOpen}
                  </span>
                )}
              </div>

              <div className="overflow-x-auto rounded-xl border border-slate-200 bg-white">
                <table className="min-w-full text-left text-sm">
                  <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                    <tr>
                      <th className="px-2 py-2">Status</th>
                      <th className="px-2 py-2">Date</th>
                      <th className="px-2 py-2">Amount</th>
                      <th className="px-2 py-2">Dir</th>
                      <th className="px-2 py-2">Statement</th>
                      <th className="px-2 py-2">Category</th>
                      <th className="px-2 py-2">Beneficiary</th>
                      <th className="px-2 py-2">Receiver</th>
                      <th className="px-2 py-2">Property</th>
                      <th className="px-2 py-2">Rule</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map((row) => {
                      const muted =
                        row.status === "skip" || row.status === "ignored";
                      const editable =
                        row.status === "new" ||
                        row.status === "to_choose" ||
                        row.status === "possible_double" ||
                        row.status === "fee";
                      const canIgnore =
                        row.status === "new" ||
                        row.status === "to_choose" ||
                        row.status === "possible_double" ||
                        row.status === "fee";
                      const rowBg =
                        row.status === "saved"
                          ? "bg-emerald-200"
                          : muted
                            ? "bg-slate-50 opacity-70"
                            : row.confirmed
                              ? "bg-emerald-50"
                              : "";
                      return (
                        <tr
                          key={row.index}
                          className={`border-t border-slate-100 align-top ${rowBg}`}
                        >
                          <td className="px-2 py-2">
                            <span
                              className={`inline-flex rounded px-1.5 py-0.5 text-[11px] font-medium ${statusBadgeClass(
                                row.status
                              )}`}
                            >
                              {row.status}
                            </span>
                            {row.existingCashflowUrl && (
                              <a
                                href={row.existingCashflowUrl}
                                target="_blank"
                                rel="noreferrer"
                                className="mt-1 flex items-center gap-1 text-[11px] text-emerald-700"
                              >
                                Open <ExternalLink className="h-3 w-3" />
                              </a>
                            )}
                            {row.status === "possible_double" && (
                              <button
                                type="button"
                                className="mt-1 block text-[11px] text-amber-800 underline"
                                onClick={() =>
                                  applyRowPatch({
                                    index: row.index,
                                    confirmAsNew: true,
                                    status: "new",
                                    confirmed: true,
                                  })
                                }
                              >
                                Keep as new
                              </button>
                            )}
                            {canIgnore && (
                              <button
                                type="button"
                                className="mt-1 block text-[11px] text-red-600 underline"
                                onClick={() =>
                                  applyRowPatch({
                                    index: row.index,
                                    ignore: true,
                                    previousStatus: row.status,
                                  })
                                }
                              >
                                Ignore
                              </button>
                            )}
                            {row.status === "ignored" && (
                              <button
                                type="button"
                                className="mt-1 block text-[11px] text-emerald-700 underline"
                                onClick={() =>
                                  applyRowPatch({
                                    index: row.index,
                                    restoreIgnored: true,
                                  })
                                }
                              >
                                Restore
                              </button>
                            )}
                            {row.internalTransfer && (
                              <div className="mt-1 text-[10px] text-violet-700">
                                Internal transfer
                              </div>
                            )}
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs">
                            {row.trxDate}
                          </td>
                          <td className="px-2 py-2 whitespace-nowrap text-xs tabular-nums">
                            {Number(row.amount).toLocaleString("el-GR", {
                              minimumFractionDigits: 2,
                            })}
                          </td>
                          <td className="px-2 py-2 text-xs">
                            {row.paymentDirection === "Incoming Payment" ? "In" : "Out"}
                          </td>
                          <td className="px-2 py-2 text-xs min-w-[14rem] max-w-[28rem] whitespace-normal break-words">
                            {row.statementText}
                          </td>
                          <td className="px-2 py-2">
                            {editable && row.status !== "fee" ? (
                              <select
                                className="min-w-[10rem] max-w-[16rem] border rounded px-1 py-0.5 text-xs"
                                value={row.trxDescription || ""}
                                onChange={(e) =>
                                  applyRowPatch({
                                    index: row.index,
                                    trxDescription: e.target.value,
                                  })
                                }
                              >
                                <option value="">— choose —</option>
                                {trxDescriptions.map((d) => (
                                  <option key={d} value={d}>
                                    {d}
                                  </option>
                                ))}
                              </select>
                            ) : (
                              <span className="text-xs whitespace-normal break-words">
                                {row.trxDescription || "—"}
                              </span>
                            )}
                          </td>
                          <td className="px-2 py-2">
                            {editable && row.status !== "fee" ? (
                              <ContactCell
                                label=""
                                value={row.trxBeneficiary}
                                onChange={(v) =>
                                  applyRowPatch({
                                    index: row.index,
                                    trxBeneficiary: v,
                                  })
                                }
                              />
                            ) : (
                              <span className="text-xs whitespace-normal break-words block min-w-[10rem] max-w-[18rem]">
                                {parsePiped(row.trxBeneficiary)?.label || "—"}
                              </span>
                            )}
                          </td>
                          <td className="px-2 py-2">
                            {editable && row.status !== "fee" ? (
                              <ContactCell
                                label=""
                                value={row.trxReceiver}
                                onChange={(v) =>
                                  applyRowPatch({
                                    index: row.index,
                                    trxReceiver: v,
                                  })
                                }
                              />
                            ) : (
                              <span className="text-xs whitespace-normal break-words block min-w-[10rem] max-w-[18rem]">
                                {parsePiped(row.trxReceiver)?.label || "—"}
                              </span>
                            )}
                          </td>
                          <td className="px-2 py-2">
                            {editable && row.status !== "fee" ? (
                              <PropertySingleCombobox
                                value={row.propertyId}
                                onChange={(v) =>
                                  applyRowPatch({
                                    index: row.index,
                                    propertyId: v,
                                    propertyOwner:
                                      v === AUX_PROPERTY
                                        ? `${session.holderLabel}|${session.holderPid}`
                                        : row.propertyOwner,
                                  })
                                }
                              />
                            ) : (
                              <span className="text-xs whitespace-normal break-words block min-w-[12rem] max-w-[22rem]">
                                {parsePiped(row.propertyId)?.label || "—"}
                              </span>
                            )}
                          </td>
                          <td className="px-2 py-2 text-[11px] text-gray-500">
                            {row.rule}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </div>
      )}

      {tab === "sessions" && (
        <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
          <table className="min-w-full text-sm">
            <thead className="bg-slate-50 text-xs uppercase text-slate-500">
              <tr>
                <th className="px-3 py-2 text-left">File</th>
                <th className="px-3 py-2 text-left">Account</th>
                <th className="px-3 py-2 text-left">Period</th>
                <th className="px-3 py-2 text-left">Status</th>
                <th className="px-3 py-2 text-left">Created</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {imports.map((item) => (
                <tr key={item.id} className="border-t">
                  <td className="px-3 py-2">{item.filename}</td>
                  <td className="px-3 py-2">{item.accountLabel}</td>
                  <td className="px-3 py-2">{item.period}</td>
                  <td className="px-3 py-2">{item.status}</td>
                  <td className="px-3 py-2 text-xs text-gray-500">
                    {item.createdAt}
                  </td>
                  <td className="px-3 py-2 text-right space-x-3 whitespace-nowrap">
                    <button
                      type="button"
                      className="text-emerald-700 text-sm underline"
                      onClick={() => reopenImport(item.id)}
                    >
                      Open
                    </button>
                    <button
                      type="button"
                      className="text-red-600 text-sm underline"
                      onClick={() => handleDeleteImport(item.id)}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {!imports.length && (
                <tr>
                  <td colSpan={6} className="px-3 py-6 text-center text-gray-500">
                    No import sessions yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {tab === "accounts" && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2 text-left">Label</th>
                  <th className="px-3 py-2 text-left">Trx Bank</th>
                  <th className="px-3 py-2 text-left">Account no.</th>
                  <th className="px-3 py-2 text-left">Holder</th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((a) => (
                  <tr key={a.id} className="border-t">
                    <td className="px-3 py-2">{a.label}</td>
                    <td className="px-3 py-2">{a.trxBank}</td>
                    <td className="px-3 py-2 font-mono text-xs">{a.accountNo}</td>
                    <td className="px-3 py-2">{a.holderLabel}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <form
            className="grid gap-2 rounded-xl border border-slate-200 bg-white p-4 sm:grid-cols-3"
            onSubmit={async (e) => {
              e.preventDefault();
              const res = await createBankImportAccount(newAccount);
              if (res.error) {
                setMessage(res.error);
                return;
              }
              setMessage("Account added.");
              setNewAccount({
                key: "",
                label: "",
                trxBank: "",
                accountNo: "",
                holderLabel: "",
                holderPid: "",
              });
              loadAccounts();
            }}
          >
            <h3 className="sm:col-span-3 font-medium text-gray-800">Add account</h3>
            {["key", "label", "trxBank", "accountNo", "holderLabel", "holderPid"].map(
              (field) => (
                <input
                  key={field}
                  className="border rounded-md px-2 py-1.5 text-sm"
                  placeholder={field}
                  value={newAccount[field]}
                  onChange={(e) =>
                    setNewAccount((prev) => ({ ...prev, [field]: e.target.value }))
                  }
                  required={["trxBank", "accountNo", "holderPid", "holderLabel"].includes(
                    field
                  )}
                />
              )
            )}
            <button
              type="submit"
              className="rounded-md bg-emerald-600 px-3 py-2 text-sm text-white sm:col-span-3"
            >
              Add account
            </button>
          </form>
        </div>
      )}

      {tab === "payees" && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white overflow-hidden">
            <table className="min-w-full text-sm">
              <thead className="bg-slate-50 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2 text-left">Match</th>
                  <th className="px-3 py-2 text-left">Category</th>
                  <th className="px-3 py-2 text-left">Receiver</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {payees.map((p) => (
                  <tr key={p.id} className="border-t">
                    <td className="px-3 py-2">{p.match}</td>
                    <td className="px-3 py-2">{p.category}</td>
                    <td className="px-3 py-2">{p.receiverLabel}</td>
                    <td className="px-3 py-2 text-right">
                      <button
                        type="button"
                        className="text-red-600 text-xs"
                        onClick={async () => {
                          await deleteBankImportPayee(p.id);
                          loadMeta();
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <form
            className="grid gap-2 rounded-xl border border-slate-200 bg-white p-4 sm:grid-cols-2"
            onSubmit={async (e) => {
              e.preventDefault();
              const res = await createBankImportPayee(newPayee);
              if (res.error) {
                setMessage(res.error);
                return;
              }
              setNewPayee({
                match: "",
                category: "",
                receiverLabel: "",
                receiverPid: "",
              });
              loadMeta();
            }}
          >
            <h3 className="sm:col-span-2 font-medium text-gray-800">Add known payee</h3>
            <input
              className="border rounded-md px-2 py-1.5 text-sm"
              placeholder="Match text (e.g. ΔΕΗ)"
              value={newPayee.match}
              onChange={(e) =>
                setNewPayee((prev) => ({ ...prev, match: e.target.value }))
              }
              required
            />
            <select
              className="border rounded-md px-2 py-1.5 text-sm"
              value={newPayee.category}
              onChange={(e) =>
                setNewPayee((prev) => ({ ...prev, category: e.target.value }))
              }
              required
            >
              <option value="">Category</option>
              {trxDescriptions.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <input
              className="border rounded-md px-2 py-1.5 text-sm"
              placeholder="Receiver label"
              value={newPayee.receiverLabel}
              onChange={(e) =>
                setNewPayee((prev) => ({ ...prev, receiverLabel: e.target.value }))
              }
            />
            <input
              className="border rounded-md px-2 py-1.5 text-sm"
              placeholder="Receiver PID (optional)"
              value={newPayee.receiverPid}
              onChange={(e) =>
                setNewPayee((prev) => ({ ...prev, receiverPid: e.target.value }))
              }
            />
            <button
              type="submit"
              className="rounded-md bg-emerald-600 px-3 py-2 text-sm text-white sm:col-span-2"
            >
              Add payee
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
