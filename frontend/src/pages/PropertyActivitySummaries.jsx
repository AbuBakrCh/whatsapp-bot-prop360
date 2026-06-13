import React, { useEffect, useState } from "react";
import { getPropertyActivitySummaries, updateActivitySummaryStatus } from "../api";

function getAthensParts(date) {
  const parts = new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Athens",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).formatToParts(date);
  const get = (type) => parseInt(parts.find((p) => p.type === type)?.value ?? "0", 10);
  return {
    year: get("year"),
    month: get("month"),
    day: get("day"),
    hour: get("hour"),
    minute: get("minute"),
    second: get("second"),
  };
}

function findAthensHourFloorUtc(now) {
  const nowA = getAthensParts(now);
  for (let t = now.getTime() - 24 * 3600000; t < now.getTime() + 24 * 3600000; t += 1000) {
    const a = getAthensParts(new Date(t));
    if (
      a.year === nowA.year &&
      a.month === nowA.month &&
      a.day === nowA.day &&
      a.hour === nowA.hour &&
      a.minute === 0 &&
      a.second === 0
    ) {
      return t;
    }
  }
  return now.getTime();
}

function computeNextHourlyRunGreeceDate(now = new Date()) {
  const floor = findAthensHourFloorUtc(now);
  if (now.getTime() <= floor) {
    return new Date(floor);
  }
  for (let t = floor + 60000; t < floor + 25 * 3600000; t += 60000) {
    const a = getAthensParts(new Date(t));
    if (a.minute === 0 && a.second === 0) {
      return new Date(t);
    }
  }
  return new Date(floor + 3600000);
}

function formatGreeceDateTime(date) {
  return date.toLocaleString("en-GB", {
    timeZone: "Europe/Athens",
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function parseScheduledSendAt(scheduledSendAt) {
  if (!scheduledSendAt) return null;
  if (scheduledSendAt instanceof Date) return scheduledSendAt;
  if (typeof scheduledSendAt !== "string") return null;
  const iso = scheduledSendAt.endsWith("Z") ? scheduledSendAt : `${scheduledSendAt}Z`;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

function getExpectedSendTime(summary) {
  const fromApi = parseScheduledSendAt(summary.scheduledSendAt);
  if (fromApi) return formatGreeceDateTime(fromApi);
  if (summary.status === "ready to send") {
    return formatGreeceDateTime(computeNextHourlyRunGreeceDate());
  }
  return null;
}

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

  const handleStatusChange = async (id, status) => {
    try {
      const result = await updateActivitySummaryStatus(id, status);
      if (result?.error) {
        alert(result.error);
        return;
      }
      setSummaries((prev) =>
        prev.map((s) =>
          s._id === id
            ? {
                ...s,
                status,
                scheduledSendAt:
                  status === "ready to send"
                    ? result.scheduledSendAt ?? s.scheduledSendAt ?? computeNextHourlyRunGreeceDate().toISOString()
                    : null,
              }
            : s
        )
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
              {summary.periodStart && summary.periodEnd && (
              <p>
                <strong>Period:</strong>{" "}
                {new Date(summary.periodStart).toLocaleDateString()} –{" "}
                {new Date(summary.periodEnd).toLocaleDateString()}
              </p>
              )}
              <div className="space-y-1">
                <p className="flex flex-wrap items-center gap-2">
                  <span>
                    <strong>Status:</strong> {summary.status}
                  </span>
                  {summary.status === "pending" && (
                    <button
                      onClick={() => handleStatusChange(summary._id, "ready to send")}
                      className="px-2 py-1 bg-yellow-500 text-white text-xs rounded hover:bg-yellow-600"
                    >
                      Mark Ready
                    </button>
                  )}
                  {summary.status === "ready to send" && (
                    <button
                      onClick={() => handleStatusChange(summary._id, "pending")}
                      className="px-2 py-1 bg-gray-500 text-white text-xs rounded hover:bg-gray-600"
                    >
                      Revert to Pending
                    </button>
                  )}
                </p>
                {summary.status === "ready to send" && (
                  <p className="text-sm text-blue-700">
                    <strong>Expected email send time:</strong>{" "}
                    {getExpectedSendTime(summary)} (Greece time)
                  </p>
                )}
              </div>
              {summary.status === "sent" && summary.emailSentAt && (
                <p className="text-sm text-green-700">
                  <strong>Email sent:</strong>{" "}
                  {new Date(summary.emailSentAt + "Z").toLocaleString()}
                </p>
              )}

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
