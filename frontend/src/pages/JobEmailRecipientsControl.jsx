import React, { useState, useEffect, useCallback } from "react";
import {
  getJobEmailRecipients,
  upsertJobEmailRecipients,
} from "../api";

const JOBS = [
  {
    id: "common_expenses_owner_email_job",
    name: "Common Expenses Owner Email",
  },
  {
    id: "electricity_bill_owner_email_job",
    name: "Electricity Bill Owner Email",
  },
  {
    id: "water_bill_owner_email_job",
    name: "Water Bill Owner Email",
  },
];

function formatList(emails) {
  return Array.isArray(emails) && emails.length > 0 ? emails.join(", ") : "";
}

export default function JobEmailRecipientsControl() {
  const [selectedJob, setSelectedJob] = useState("");
  const [toEmails, setToEmails] = useState("");
  const [ccEmails, setCcEmails] = useState("");
  const [allJobs, setAllJobs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const fetchAllJobs = useCallback(async () => {
    try {
      const res = await getJobEmailRecipients();
      if (res?.success) setAllJobs(res.data || []);
    } catch (err) {
      console.error("Failed to fetch job email recipients:", err);
    }
  }, []);

  useEffect(() => {
    fetchAllJobs();
  }, [fetchAllJobs]);

  const loadJobIntoForm = (jobId, jobs = allJobs) => {
    setSelectedJob(jobId);
    const job = jobs.find((j) => j.job_id === jobId);
    setToEmails(formatList(job?.to));
    setCcEmails(formatList(job?.cc));
    setResponseMsg("");
  };

  const handleJobChange = (e) => {
    const jobId = e.target.value;
    if (!jobId) {
      setSelectedJob("");
      setToEmails("");
      setCcEmails("");
      return;
    }
    loadJobIntoForm(jobId);
  };

  const handleSave = async (e) => {
    e.preventDefault();
    if (!selectedJob) {
      setResponseMsg("Please select a job.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const res = await upsertJobEmailRecipients(selectedJob, toEmails, ccEmails);
      if (res?.success) {
        setResponseMsg("Recipients updated successfully.");
        await fetchAllJobs();
      } else {
        setResponseMsg(res?.message || "Failed to update recipients.");
      }
    } catch (err) {
      setResponseMsg("Error updating recipients.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = async () => {
    if (!selectedJob) {
      setResponseMsg("Please select a job.");
      return;
    }

    setLoading(true);
    setResponseMsg("");
    setToEmails("");
    setCcEmails("");

    try {
      const res = await upsertJobEmailRecipients(selectedJob, "", "");
      if (res?.success) {
        setResponseMsg("Recipients cleared.");
        await fetchAllJobs();
      } else {
        setResponseMsg(res?.message || "Failed to clear recipients.");
      }
    } catch (err) {
      setResponseMsg("Error clearing recipients.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto mt-6 bg-white rounded-2xl shadow-lg p-6 border border-slate-200 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Job Email Recipients</h2>

      <div>
        <label className="block mb-1 font-medium text-gray-700">Select Job:</label>
        <select
          className="w-full border px-3 py-2 rounded-md"
          value={selectedJob}
          onChange={handleJobChange}
        >
          <option value="">-- Select a job --</option>
          {JOBS.map((job) => (
            <option key={job.id} value={job.id}>
              {job.name}
            </option>
          ))}
        </select>
      </div>

      <form onSubmit={handleSave} className="space-y-3">
        <div>
          <label className="block mb-1 font-medium text-gray-700">
            To (comma-separated emails)
          </label>
          <input
            type="text"
            placeholder="e.g. email1@test.com, email2@test.com"
            className="w-full border px-3 py-2 rounded-md"
            value={toEmails}
            onChange={(e) => setToEmails(e.target.value)}
            disabled={!selectedJob}
          />
        </div>
        <div>
          <label className="block mb-1 font-medium text-gray-700">
            Cc (comma-separated emails)
          </label>
          <input
            type="text"
            placeholder="e.g. email1@test.com, email2@test.com"
            className="w-full border px-3 py-2 rounded-md"
            value={ccEmails}
            onChange={(e) => setCcEmails(e.target.value)}
            disabled={!selectedJob}
          />
        </div>
        <div className="flex gap-3">
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
          <button
            type="button"
            onClick={handleClear}
            disabled={loading || !selectedJob}
            className={`px-4 py-2 rounded-md font-medium border ${
              loading || !selectedJob
                ? "bg-slate-100 text-slate-400 cursor-not-allowed border-slate-200"
                : "bg-white text-red-600 border-red-200 hover:bg-red-50"
            }`}
          >
            Clear Recipients
          </button>
        </div>
      </form>

      <div className="mt-4">
        <h3 className="font-semibold text-gray-700 mb-2">All Jobs & Recipients</h3>
        {allJobs.length === 0 ? (
          <p className="text-gray-500">No jobs found.</p>
        ) : (
          <ul className="space-y-2">
            {allJobs.map((job) => (
              <li key={job.job_id}>
                <button
                  type="button"
                  onClick={() => loadJobIntoForm(job.job_id)}
                  className="w-full text-left border p-2 rounded-md bg-slate-50 hover:bg-slate-100"
                >
                  <strong>{job.job_id}</strong>
                  <div className="text-sm text-gray-600 mt-1">
                    To:{" "}
                    {job.to?.length > 0 ? job.to.join(", ") : "None"}
                  </div>
                  <div className="text-sm text-gray-600">
                    Cc:{" "}
                    {job.cc?.length > 0 ? job.cc.join(", ") : "None"}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      {responseMsg && <p className="mt-2 text-green-600">{responseMsg}</p>}
    </div>
  );
}
