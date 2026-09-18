import React, { useCallback, useEffect, useRef, useState } from "react";
import axios from "axios";

function formatGreeceTime(isoString) {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return isoString;
  return new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Athens",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function formatResult(data) {
  if (!data) return "";
  if (data.lastError) {
    return `Failed: ${data.lastError}`;
  }
  const result = data.lastResult;
  if (!result) {
    return data.running ? "Running…" : "Idle";
  }
  const when = data.finishedAt
    ? ` at ${formatGreeceTime(data.finishedAt)} (Greece)`
    : "";
  return `Finished${when}: scanned ${result.scanned ?? 0}`;
}

export default function JobControl({ jobId, jobName, pollStatus = false }) {
  const [status, setStatus] = useState(""); // start / stop
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");
  const [msgIsError, setMsgIsError] = useState(false);
  const pollRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const fetchStatus = useCallback(async () => {
    if (!pollStatus) return null;
    const res = await axios.get(
      `${import.meta.env.VITE_API_BASE}/jobs/${jobId}`
    );
    return res.data;
  }, [jobId, pollStatus]);

  const applyStatus = useCallback(
    (data) => {
      if (!data) return;
      setRunning(Boolean(data.running));
      if (data.running) {
        setResponseMsg("Running…");
        setMsgIsError(false);
        return;
      }
      if (data.lastError || data.lastResult || data.finishedAt) {
        setResponseMsg(formatResult(data));
        setMsgIsError(Boolean(data.lastError));
        setStatus("");
      }
    },
    []
  );

  const startPolling = useCallback(() => {
    if (!pollStatus) return;
    stopPolling();
    pollRef.current = setInterval(async () => {
      try {
        const data = await fetchStatus();
        applyStatus(data);
        if (data && !data.running) {
          stopPolling();
        }
      } catch {
        // keep polling; transient network errors
      }
    }, 1500);
  }, [pollStatus, stopPolling, fetchStatus, applyStatus]);

  useEffect(() => {
    if (!pollStatus) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchStatus();
        if (!cancelled) {
          applyStatus(data);
          if (data?.running) startPolling();
        }
      } catch {
        // ignore initial status errors
      }
    })();
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, [pollStatus, fetchStatus, applyStatus, startPolling, stopPolling]);

  const handleJobAction = async (action) => {
    setLoading(true);
    setResponseMsg("");
    setMsgIsError(false);

    try {
      const res = await axios.post(
        `${import.meta.env.VITE_API_BASE}/jobs/${jobId}`,
        null,
        {
          params: { action },
        }
      );

      if (res.data?.error) {
        setResponseMsg(res.data.error);
        setMsgIsError(true);
      } else if (res.data?.message) {
        setResponseMsg(res.data.message);
        setStatus(action);
        setMsgIsError(false);
        if (action === "start" && pollStatus) {
          setRunning(true);
          startPolling();
        }
        if (action === "stop" && pollStatus) {
          startPolling();
        }
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(
        err.response?.data?.error || "Failed to process job action."
      );
      setMsgIsError(true);
    } finally {
      setLoading(false);
    }
  };

  const startDisabled = loading || running || status === "start";
  const stopDisabled = loading || (!running && status === "stop");

  return (
    <div className="max-w-md mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">{jobName}</h2>

      <div className="flex gap-4 mb-4">
        <button
          onClick={() => handleJobAction("start")}
          disabled={startDisabled}
          className={`flex-1 px-4 py-2 rounded-md text-white font-medium transition ${
            startDisabled
              ? "bg-green-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {running ? "Running…" : loading && status !== "stop" ? "Processing..." : "Start Job"}
        </button>

        <button
          onClick={() => handleJobAction("stop")}
          disabled={stopDisabled}
          className={`flex-1 px-4 py-2 rounded-md text-white font-medium transition ${
            stopDisabled
              ? "bg-red-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading && status !== "start" ? "Processing..." : "Stop Job"}
        </button>
      </div>

      {responseMsg && (
        <p
          className={`mt-4 text-sm ${
            msgIsError
              ? "text-red-600"
              : responseMsg.toLowerCase().includes("finished") ||
                  responseMsg.toLowerCase().includes("started")
                ? "text-green-600"
                : running
                  ? "text-amber-600"
                  : "text-slate-700"
          }`}
        >
          {responseMsg}
        </p>
      )}
    </div>
  );
}
