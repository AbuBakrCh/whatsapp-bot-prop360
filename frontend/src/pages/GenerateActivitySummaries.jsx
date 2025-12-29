import React, { useState, useRef } from "react";
import { generateActivitySummaries, getActivitySummaryProgress } from "../api";

export default function GenerateActivitySummaries() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const [progress, setProgress] = useState(0);
  const [total, setTotal] = useState(0); // store total count from progress

  const jobIdRef = useRef(null);
  const intervalRef = useRef(null);

  const handleGenerate = async () => {
    if (!startDate || !endDate) {
      setStatusMsg("Start Date and End Date are required.");
      return;
    }

    setLoading(true);
    setProgress(0);
    setTotal(0);
    setStatusMsg("Starting job...");

    try {
      const start = `${startDate}T00:00:00`;
      const end = `${endDate}T23:59:59`;
      const payload = { startDate: start, endDate: end };

      // Start job
      const data = await generateActivitySummaries(payload);
      if (data.error) {
        setStatusMsg(`Error: ${data.error}`);
        setLoading(false);
        return;
      }

      const jobId = data.jobId;
      jobIdRef.current = jobId;

      setStatusMsg("Job started. Waiting for progress...");

      // Polling
      intervalRef.current = setInterval(async () => {
        const progressData = await getActivitySummaryProgress(jobId);
        if (progressData.error) {
          setStatusMsg(`Error: ${progressData.error}`);
          clearInterval(intervalRef.current);
          setLoading(false);
          return;
        }

        const { processed = 0, total: jobTotal = 0, status } = progressData;
        setTotal(jobTotal);
        setProgress(jobTotal > 0 ? (processed / jobTotal) * 100 : 0);
        setStatusMsg(`Processed ${processed}/${jobTotal}...`);

        if (status === "completed") {
          setStatusMsg(`Job completed: ${processed}/${jobTotal}`);
          setProgress(100);
          clearInterval(intervalRef.current);
          setLoading(false);
        }
      }, 1000); // poll every 1 second
    } catch (err) {
      console.error(err);
      setStatusMsg("Failed to start activity summary job.");
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Generate Activity Summaries
      </h2>

      {/* Date Inputs */}
      <div className="flex flex-col gap-4 mb-6">
        <label className="flex flex-col gap-1 text-gray-700">
          Start Date:
          <input
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2"
          />
        </label>

        <label className="flex flex-col gap-1 text-gray-700">
          End Date:
          <input
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2"
          />
        </label>
      </div>

      {/* Action Button */}
      <button
        onClick={handleGenerate}
        disabled={loading}
        className={`w-full px-4 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        Generate
      </button>

      {/* Status */}
      {statusMsg && <p className="mt-4 text-gray-700">{statusMsg}</p>}

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 h-4 rounded mt-2">
        <div
          className="bg-green-600 h-4 rounded"
          style={{ width: `${progress}%`, transition: "width 0.3s" }}
        ></div>
      </div>
    </div>
  );
}
