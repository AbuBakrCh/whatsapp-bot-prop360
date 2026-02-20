import React, { useState } from "react";
import { mergeContacts } from "../api";

export default function MergeContacts() {
  const [sourcePid, setSourcePid] = useState("");
  const [targetPid, setTargetPid] = useState("");

  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const handleMerge = async () => {
    if (!sourcePid || !targetPid) {
      setResponseMsg("Both Source PID and Target PID are required.");
      return;
    }

    // Convert to numbers
    const sourcePidNum = Number(sourcePid.trim());
    const targetPidNum = Number(targetPid.trim());

    if (isNaN(sourcePidNum) || isNaN(targetPidNum)) {
      setResponseMsg("PID must be a valid numeric value.");
      return;
    }

    if (sourcePidNum === targetPidNum) {
      setResponseMsg("Source and Target PID cannot be the same.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const payload = {
        sourcePid: sourcePidNum,
        targetPid: targetPidNum,
      };

      const res = await mergeContacts(payload);

      if (res.error) {
        setResponseMsg(res.error);
      } else if (res.message) {
        setResponseMsg(res.message);
      } else {
        setResponseMsg("Merge completed.");
      }

    } catch (err) {
      setResponseMsg(
        err.response?.data?.error || "Failed to merge contacts."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">

      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Merge Contacts
      </h2>

      {/* Inputs */}
      <div className="flex flex-col gap-4 mb-6">

        <input
          type="text"
          value={sourcePid}
          onChange={(e) => setSourcePid(e.target.value)}
          placeholder="Source Contact PID"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <input
          type="text"
          value={targetPid}
          onChange={(e) => setTargetPid(e.target.value)}
          placeholder="Target Contact PID"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

      </div>

      {/* Merge Button */}
      <button
        onClick={handleMerge}
        disabled={loading}
        className={`w-full px-4 py-2 rounded-md text-white font-medium transition ${
          loading
            ? "bg-blue-300 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {loading ? "Merging..." : "Merge Contacts"}
      </button>

      {/* Response */}
      {responseMsg && (
        <p
          className={`mt-4 ${
            responseMsg.toLowerCase().includes("success")
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
