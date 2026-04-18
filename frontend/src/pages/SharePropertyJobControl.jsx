import React, { useEffect, useState } from "react";
import {
  getSharePropertyJobStatus,
  updateSharePropertyJobStatus,
} from "../api";

export default function SharePropertyJobControl() {
  const [status, setStatus] = useState("disable");
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  // -------------------------
  // Load current status
  // -------------------------
  const loadStatus = async () => {
    const res = await getSharePropertyJobStatus();

    if (!res.error && res.status) {
      setStatus(res.status);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  // -------------------------
  // Toggle
  // -------------------------
  const handleToggle = async () => {
    const newStatus = status === "enable" ? "disable" : "enable";

    setLoading(true);
    setResponseMsg("");

    const res = await updateSharePropertyJobStatus(newStatus);

    if (res.error) {
      setResponseMsg(res.error);
    } else {
      setStatus(newStatus);
      setResponseMsg(res.message);
    }

    setLoading(false);
  };

  return (
    <div className="max-w-xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Share Property Job
      </h2>

      {/* Status */}
      <div className="mb-4">
        <span className="text-sm text-gray-600">Current Status: </span>
        <span
          className={`font-semibold ${
            status === "enable" ? "text-green-600" : "text-red-600"
          }`}
        >
          {status === "enable" ? "Enabled" : "Disabled"}
        </span>
      </div>

      {/* Toggle Button */}
      <button
        onClick={handleToggle}
        disabled={loading}
        className={`w-full px-4 py-2 rounded-md text-white font-medium ${
          status === "enable"
            ? "bg-red-600 hover:bg-red-700"
            : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading
          ? "Updating..."
          : status === "enable"
          ? "Disable Job"
          : "Enable Job"}
      </button>

      {/* Response */}
      {responseMsg && (
        <p className="mt-4 text-gray-700">{responseMsg}</p>
      )}
    </div>
  );
}