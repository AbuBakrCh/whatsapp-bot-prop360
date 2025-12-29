import React, { useEffect, useState } from "react";
import { getPropertyActivitySummaries, updateActivitySummaryStatus } from "../api";

export default function PropertyActivitySummaries() {
  const [summaries, setSummaries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [page, setPage] = useState(1);
  const pageSize = 10; // items per page

  const fetchSummaries = async () => {
    setLoading(true);
    setErrorMsg("");
    try {
      const data = await getPropertyActivitySummaries();
      if (data.error) {
        setErrorMsg(data.error);
      } else {
        setSummaries(data.data);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg("Failed to fetch activity summaries.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSummaries();
  }, []);

  const handleMarkReady = async (id) => {
    try {
      await updateActivitySummaryStatus(id, "ready to send");
      // Update locally so UI shows the change immediately
      setSummaries((prev) =>
        prev.map((s) => (s._id === id ? { ...s, status: "ready to send" } : s))
      );
    } catch (err) {
      console.error(err);
      alert("Failed to update status.");
    }
  };

  const totalPages = Math.ceil(summaries.length / pageSize);
  const displayedSummaries = summaries.slice((page - 1) * pageSize, page * pageSize);

  return (
    <div className="max-w-5xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">
          Property Activity Summaries
        </h2>
        <button
          onClick={fetchSummaries}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>

      {loading && <p className="text-gray-500">Loading summaries...</p>}
      {errorMsg && <p className="text-red-600">{errorMsg}</p>}
      {!loading && summaries.length === 0 && (
        <p className="text-gray-500">No activity summaries found.</p>
      )}

      {!loading && displayedSummaries.length > 0 && (
        <div className="space-y-4 max-h-[600px] overflow-y-auto">
          {displayedSummaries.map((summary) => (
            <div
              key={summary._id}
              className="p-4 border border-slate-200 rounded-md shadow-sm"
            >
              <p><strong>Client:</strong> {summary.clientId || "N/A"}</p>
              <p><strong>Property:</strong> {summary.propertyId || "N/A"}</p>
              <p><strong>Indicator:</strong> {summary.indicator || "N/A"}</p>
              <p><strong>Client Email:</strong> {summary.clientEmail || "Not Available"}</p>
              <p>
                <strong>Status:</strong> {summary.status}
                {summary.status === "pending" && (
                  <button
                    onClick={() => handleMarkReady(summary._id)}
                    className="ml-2 px-2 py-1 bg-yellow-500 text-white text-xs rounded hover:bg-yellow-600"
                  >
                    Mark Ready
                  </button>
                )}
              </p>

              <p className="mt-2"><strong>Summary:</strong> {summary.summary || "No summary"}</p>
              {summary.activities && summary.activities.length > 0 && (
                <div className="mt-2">
                  <strong>Activities:</strong>
                  <ul className="list-disc list-inside">
                    {summary.activities.map((act, idx) => (
                      <li key={idx}>{act}</li>
                    ))}
                  </ul>
                </div>
              )}
              <p className="mt-2 text-gray-400 text-sm">
                Created: {summary.createdAt ? new Date(summary.createdAt + "Z").toLocaleString() : "N/A"}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-4">
          <button
            onClick={() => setPage((p) => Math.max(p - 1, 1))}
            disabled={page === 1}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
          >
            Prev
          </button>
          <span>
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
            disabled={page === totalPages}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
