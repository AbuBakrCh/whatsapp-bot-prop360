import React, { useState } from "react";
import { getDuplicates } from "../api";

const GROUP_PAGE_SIZE = 10;
const RECORD_PAGE_SIZE = 5;

export default function ShowDuplicates() {
  const [data, setData] = useState([]);
  const [activeTab, setActiveTab] = useState("email");
  const [groupPage, setGroupPage] = useState(1);
  const [expandedKey, setExpandedKey] = useState(null);
  const [recordPages, setRecordPages] = useState({});
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false); // only show panel after click

  const fetchDuplicates = async () => {
    setLoading(true);
    setLoaded(false);
    setData([]);
    setExpandedKey(null);
    setGroupPage(1);

    try {
      const res = await getDuplicates();
      setData(res.duplicates || []);
      setLoaded(true);
    } finally {
      setLoading(false);
    }
  };

  const grouped = data.reduce((acc, d) => {
    acc[d.field] = acc[d.field] || [];
    acc[d.field].push(d);
    return acc;
  }, {});

  const activeList = grouped[activeTab] || [];
  const totalGroupPages = Math.ceil(activeList.length / GROUP_PAGE_SIZE);
  const visibleGroups = activeList.slice(
    (groupPage - 1) * GROUP_PAGE_SIZE,
    groupPage * GROUP_PAGE_SIZE
  );

  const toggleExpand = (key) => {
    setExpandedKey(expandedKey === key ? null : key);
  };

  const renderRecords = (dup, key) => {
    const page = recordPages[key] || 1;
    const sorted = [...dup.contacts].sort(
      (a, b) => new Date(a.createdAt) - new Date(b.createdAt)
    );
    const totalPages = Math.ceil(sorted.length / RECORD_PAGE_SIZE);
    const visible = sorted.slice(
      (page - 1) * RECORD_PAGE_SIZE,
      page * RECORD_PAGE_SIZE
    );

    return (
      <div className="mt-3 bg-slate-50 rounded-lg p-3">
        {visible.map((c, i) => (
          <div
            key={i}
            className="flex justify-between items-center bg-white border rounded px-3 py-2 mb-2 text-sm"
          >
            <div>
              <div>{new Date(c.createdAt).toLocaleString()}</div>
              <div className="text-gray-500">
                {c.ownerName || "Unknown"} · {c.ownerEmail || "N/A"}
              </div>
            </div>
            <a
              href={c.contactUrl}
              target="_blank"
              rel="noreferrer"
              className="text-blue-600"
            >
              Open
            </a>
          </div>
        ))}

        {totalPages > 1 && (
          <div className="flex justify-end gap-2 text-xs">
            <button
              disabled={page === 1}
              onClick={() =>
                setRecordPages({ ...recordPages, [key]: page - 1 })
              }
              className="px-2 py-1 border rounded"
            >
              Prev
            </button>
            <span>{page}/{totalPages}</span>
            <button
              disabled={page === totalPages}
              onClick={() =>
                setRecordPages({ ...recordPages, [key]: page + 1 })
              }
              className="px-2 py-1 border rounded"
            >
              Next
            </button>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto mt-8 bg-white rounded-xl shadow p-6 flex flex-col">
      <h2 className="text-xl font-bold mb-4">Duplicate Contacts</h2>

      <button
        onClick={fetchDuplicates}
        disabled={loading}
        className={`mb-4 px-4 py-2 rounded text-white ${
          loading ? "bg-green-300" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Fetching..." : "Show Duplicates"}
      </button>

      {/* ✅ Only render duplicates section if data exists */}
      {loaded && activeList.length > 0 && (
        <div className="flex flex-col">
          {/* Tabs */}
          <div className="flex gap-3 mb-4">
            {["email", "phone", "name"].map((f) => (
              <button
                key={f}
                onClick={() => {
                  setActiveTab(f);
                  setGroupPage(1);
                  setExpandedKey(null);
                }}
                className={`px-4 py-2 rounded ${
                  activeTab === f
                    ? "bg-green-600 text-white"
                    : "bg-slate-100"
                }`}
              >
                {f.toUpperCase()} ({grouped[f]?.length || 0})
              </button>
            ))}
          </div>

          {/* Duplicate Groups */}
          <div className="overflow-y-auto border rounded p-3">
            {visibleGroups.map((dup) => {
              const key = `${activeTab}-${dup.value}`;
              return (
                <div key={key} className="border-b py-2">
                  <div
                    className="cursor-pointer flex justify-between font-medium"
                    onClick={() => toggleExpand(key)}
                  >
                    <span>{dup.value}</span>
                    <span className="text-gray-500">{dup.count} records</span>
                  </div>

                  {expandedKey === key && renderRecords(dup, key)}
                </div>
              );
            })}

            {/* Group-level pagination */}
            {totalGroupPages > 1 && (
              <div className="flex justify-center items-center gap-2 mt-4">
                <button
                  onClick={() => setGroupPage((p) => Math.max(p - 1, 1))}
                  disabled={groupPage === 1}
                  className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
                >
                  Prev
                </button>
                <span>
                  Page {groupPage} of {totalGroupPages}
                </span>
                <button
                  onClick={() =>
                    setGroupPage((p) => Math.min(p + 1, totalGroupPages))
                  }
                  disabled={groupPage === totalGroupPages}
                  className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* No duplicates message */}
      {loaded && activeList.length === 0 && (
        <p className="text-gray-500 mt-4">No duplicate records found.</p>
      )}
    </div>
  );
}
