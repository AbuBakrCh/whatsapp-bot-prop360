import React, { useEffect, useRef, useState } from "react";
import { searchEntities, unshareEntity } from "../api";

const MODULE_OPTIONS = [
  { value: "property", label: "Property" },
  { value: "contact", label: "Contact" },
  { value: "timetable", label: "Timetable" },
  { value: "cashflow", label: "Cashflow" },
];

const SEARCH_PLACEHOLDERS = {
  property: "Search by title, PID, or ID from the Prop360 link",
  contact: "Search by name, PID, or ID from the Prop360 link",
  timetable: "Search by description, property/contact, PID, or ID from the link",
  cashflow: "Search by description, property/contact, PID, or ID from the link",
};

export default function UnshareEntity() {
  const [entityType, setEntityType] = useState("property");
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    setQuery("");
    setResults([]);
    setSelected(null);
    setIsOpen(false);
    setResponseMsg(null);
  }, [entityType]);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    const timer = setTimeout(async () => {
      const res = await searchEntities(entityType, trimmed, 15);
      if (!res.error) {
        setResults(res.data || []);
        setIsOpen(true);
      } else {
        setResults([]);
      }
      setIsSearching(false);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, entityType]);

  const handleSelect = (entity) => {
    setSelected(entity);
    setQuery("");
    setResults([]);
    setIsOpen(false);
    setResponseMsg(null);
  };

  const handleClearSelection = () => {
    setSelected(null);
    setResponseMsg(null);
  };

  const handleUnshare = async (e) => {
    e.preventDefault();
    setResponseMsg(null);

    if (!selected?.mongoId) {
      setResponseMsg({
        type: "error",
        text: "Select an entity to unshare.",
      });
      return;
    }

    const moduleLabel =
      MODULE_OPTIONS.find((m) => m.value === entityType)?.label || "entity";
    const confirmed = window.confirm(
      `Unshare this ${moduleLabel} from all merchants?\n\n${selected.label}\nPID: ${selected.pid || "—"}\nID: ${selected.mongoId}\nCurrently shared with ${selected.sharedWithMerchantsCount ?? 0} merchant(s).`
    );
    if (!confirmed) return;

    setLoading(true);
    const res = await unshareEntity({
      type: entityType,
      mongoId: selected.mongoId,
    });
    setLoading(false);

    if (res.error) {
      setResponseMsg({ type: "error", text: res.error });
      return;
    }

    setSelected({
      ...selected,
      sharedWithMerchantsCount: 0,
    });
    setResponseMsg({
      type: "success",
      text: `${res.message || "Unshared successfully."} Cleared ${
        res.previousSharedWithMerchantsCount ?? 0
      } merchant share(s).`,
    });
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Unshare Entity</h2>
      <p className="text-sm text-gray-600 mb-4">
        Select a module, find one record by name/title, PID, or the ID from its
        Prop360 link, then remove it from all shared merchants.
      </p>

      <form onSubmit={handleUnshare} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Module
          </label>
          <select
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            {MODULE_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        <div ref={containerRef} className="relative">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Find entity
          </label>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              if (results.length > 0) setIsOpen(true);
            }}
            placeholder={SEARCH_PLACEHOLDERS[entityType]}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          />

          {isSearching && (
            <p className="text-xs text-gray-500 mt-1">Searching...</p>
          )}

          {isOpen && results.length > 0 && (
            <ul className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {results.map((entity) => (
                <li key={entity.mongoId}>
                  <button
                    type="button"
                    onClick={() => handleSelect(entity)}
                    className="w-full text-left px-3 py-2 hover:bg-blue-50 border-b border-gray-100 last:border-b-0"
                  >
                    <p className="text-sm font-medium text-gray-800">
                      {entity.label}
                    </p>
                    <p className="text-xs text-gray-500">
                      PID {entity.pid || "—"} · ID {entity.mongoId} · Shared with{" "}
                      {entity.sharedWithMerchantsCount ?? 0}
                    </p>
                  </button>
                </li>
              ))}
            </ul>
          )}

          {isOpen &&
            !isSearching &&
            query.trim().length >= 2 &&
            results.length === 0 && (
              <p className="text-xs text-gray-500 mt-1">No matches found.</p>
            )}
        </div>

        {selected && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-1">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-gray-800">
                  {selected.label}
                </p>
                <p className="text-xs text-gray-600 mt-1">
                  PID: {selected.pid || "—"}
                </p>
                <p className="text-xs text-gray-600">ID: {selected.mongoId}</p>
                <p className="text-xs text-gray-600">
                  Shared with {selected.sharedWithMerchantsCount ?? 0}{" "}
                  merchant(s)
                </p>
              </div>
              <button
                type="button"
                onClick={handleClearSelection}
                className="text-xs text-slate-600 hover:text-slate-900 underline"
              >
                Clear
              </button>
            </div>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !selected}
          className={`px-4 py-2 rounded-lg text-white text-sm font-medium ${
            loading || !selected
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading ? "Unsharing..." : "Unshare from all"}
        </button>
      </form>

      {responseMsg && (
        <p
          className={`mt-4 text-sm ${
            responseMsg.type === "error" ? "text-red-600" : "text-green-600"
          }`}
        >
          {responseMsg.text}
        </p>
      )}
    </div>
  );
}
