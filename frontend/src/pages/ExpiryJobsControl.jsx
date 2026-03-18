import React, { useState, useEffect } from "react";
import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

// Predefined jobs
const JOBS = [
  { id: "ide_expiry_job", name: "IDE Expiry Job" },
  { id: "lease_expiry_job", name: "Lease Expiry Job" },
  { id: "passport_expiry_job", name: "Passport Expiry Job" },
];

export default function JobControl() {
  const [selectedJob, setSelectedJob] = useState("");
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");
  const [recipients, setRecipients] = useState(""); // comma-separated
  const [allJobs, setAllJobs] = useState([]);

  // --- Fetch all job recipients ---
  const fetchAllJobs = async () => {
    try {
      const res = await axios.get(`${BASE}/job-control/expiry`);
      if (res.data?.success) setAllJobs(res.data.data || []);
    } catch (err) {
      console.error("Failed to fetch job controls:", err);
    }
  };

  useEffect(() => {
    fetchAllJobs();
  }, []);

  // --- Add/Update Recipients ---
  const handleSaveRecipients = async (e) => {
    e.preventDefault();
    if (!selectedJob || !recipients) {
      setResponseMsg("Please select a job and enter recipients.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const res = await axios.post(`${BASE}/job-control/expiry/email-recipients`, {
        job_id: selectedJob,
        emails: recipients,
      });

      if (res.data?.success) {
        setResponseMsg("Recipients updated successfully.");
        setRecipients("");
        fetchAllJobs();
      } else {
        setResponseMsg(res.data?.message || "Failed to update recipients.");
      }
    } catch (err) {
      setResponseMsg("Error updating recipients.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Expiry Jobs Control</h2>

      {/* --- Job Selector --- */}
      <div>
        <label className="block mb-1 font-medium text-gray-700">Select Job:</label>
        <select
          className="w-full border px-3 py-2 rounded-md"
          value={selectedJob}
          onChange={(e) => setSelectedJob(e.target.value)}
        >
          <option value="">-- Select a job --</option>
          {JOBS.map((job) => (
            <option key={job.id} value={job.id}>
              {job.name}
            </option>
          ))}
        </select>
      </div>

      {/* --- Add / Update Recipients Form --- */}
      <form onSubmit={handleSaveRecipients} className="space-y-3">
        <label className="block mb-1 font-medium text-gray-700">
          Add / Update Recipients (comma-separated emails)
        </label>
        <input
          type="text"
          placeholder="e.g. email1@test.com, email2@test.com"
          className="w-full border px-3 py-2 rounded-md"
          value={recipients}
          onChange={(e) => setRecipients(e.target.value)}
        />
        <button
          type="submit"
          disabled={loading || !selectedJob}
          className={`px-4 py-2 rounded-md text-white font-medium ${
            loading || !selectedJob
              ? "bg-blue-300 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Saving..." : "Save Recipients"}
        </button>
      </form>

      {/* --- Show All Job Recipients --- */}
      <div className="mt-4">
        <h3 className="font-semibold text-gray-700 mb-2">All Jobs & Recipients</h3>
        {allJobs.length === 0 ? (
          <p className="text-gray-500">No jobs found.</p>
        ) : (
          <ul className="space-y-2">
            {allJobs.map((job) => (
              <li key={job.job_id} className="border p-2 rounded-md bg-slate-50">
                <strong>{job.job_id}:</strong>{" "}
                {job.recipients.length > 0 ? job.recipients.join(", ") : "No recipients"}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* --- Status / Message --- */}
      {responseMsg && (
        <p className={`mt-2 text-green-600`}>{responseMsg}</p>
      )}
    </div>
  );
}