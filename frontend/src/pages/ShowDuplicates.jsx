import React, { useState } from "react";
import { getDuplicates } from "../api"; // you'll create this API function

export default function ShowDuplicates() {
  const [duplicates, setDuplicates] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const handleFetchDuplicates = async () => {
    setLoading(true);
    setErrorMessage("");
    setDuplicates([]);

    try {
      const result = await getDuplicates(); // call API
      setDuplicates(result.duplicates || []);
    } catch (err) {
      console.error(err);
      setErrorMessage(err.response?.data?.detail || err.message || "Failed to fetch duplicates");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Duplicate Fields</h2>

      <button
        onClick={handleFetchDuplicates}
        disabled={loading}
        className={`mb-4 px-4 py-2 rounded-md text-white font-medium transition ${
          loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
        }`}
      >
        {loading ? "Fetching..." : "Show Duplicates"}
      </button>

      {errorMessage && <div className="mb-4 text-red-600">{errorMessage}</div>}

      {duplicates.length > 0 && (
        <div className="overflow-x-auto max-h-80 overflow-y-auto border-t border-slate-200 pt-2">
          <table className="w-full text-sm text-left text-gray-700">
            <thead className="bg-slate-100 sticky top-0">
              <tr>
                <th className="px-3 py-2">Field</th>
                <th className="px-3 py-2">Value</th>
                <th className="px-3 py-2">Count</th>
                <th className="px-3 py-2">Form IDs</th>
              </tr>
            </thead>
            <tbody>
              {duplicates.map((dup, idx) => (
                <tr key={idx} className="border-b hover:bg-slate-50">
                  <td className="px-3 py-2">{dup.field}</td>
                  <td className="px-3 py-2">{dup.value}</td>
                  <td className="px-3 py-2">{dup.count}</td>
                  <td className="px-3 py-2">{dup.formIds.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {duplicates.length === 0 && !loading && !errorMessage && (
        <p className="text-gray-600">No duplicates found yet.</p>
      )}
    </div>
  );
}
