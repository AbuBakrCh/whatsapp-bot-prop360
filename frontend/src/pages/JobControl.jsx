import React, { useState } from "react";
import axios from "axios";

export default function JobControl({ jobId, jobName }) {
  const [status, setStatus] = useState(""); // start / stop
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const handleJobAction = async (action) => {
    setLoading(true);
    setResponseMsg("");

    try {
      const res = await axios.post(`${import.meta.env.VITE_API_BASE}/jobs/${jobId}`, null, {
        params: { action }, // send ?action=start or ?action=stop
      });

      if (res.data?.message) {
        setResponseMsg(res.data.message);
        setStatus(action);
      } else if (res.data?.error) {
        setResponseMsg(res.data.error);
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(
        err.response?.data?.error || "Failed to process job action."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">{jobName}</h2>

      <div className="flex gap-4 mb-4">
        <button
          onClick={() => handleJobAction("start")}
          disabled={loading || status === "start"}
          className={`flex-1 px-4 py-2 rounded-md text-white font-medium transition ${
            loading || status === "start"
              ? "bg-green-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading && status !== "stop" ? "Processing..." : "Start Job"}
        </button>

        <button
          onClick={() => handleJobAction("stop")}
          disabled={loading || status === "stop"}
          className={`flex-1 px-4 py-2 rounded-md text-white font-medium transition ${
            loading || status === "stop"
              ? "bg-red-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading && status !== "start" ? "Processing..." : "Stop Job"}
        </button>
      </div>

      {responseMsg && (
        <p
          className={`mt-4 ${
            responseMsg.toLowerCase().includes("started")
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {responseMsg}
        </p>
      )}
    </div>
  );
}