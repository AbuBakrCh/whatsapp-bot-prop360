import React, { useState } from "react";
import { shareWithGroups } from "../api";
import GroupSearchCombobox from "../components/GroupSearchCombobox";

function extractId(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (value.$oid) return value.$oid;
  return String(value);
}

export default function ShareWithGroups() {
  const [shareType, setShareType] = useState("property");
  const [pid, setPid] = useState("");
  const [selectedGroups, setSelectedGroups] = useState([]);
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState(null);

  const handleShare = async (e) => {
    e.preventDefault();
    setResponseMsg(null);

    const trimmedPid = pid.trim();
    if (!trimmedPid) {
      setResponseMsg({ type: "error", text: "PID is required." });
      return;
    }

    if (selectedGroups.length === 0) {
      setResponseMsg({ type: "error", text: "Select at least one group." });
      return;
    }

    setLoading(true);

    const res = await shareWithGroups({
      shareType,
      pid: trimmedPid,
      groupIds: selectedGroups.map((g) => extractId(g._id)),
    });

    setLoading(false);

    if (res.error) {
      setResponseMsg({ type: "error", text: res.error });
      return;
    }

    const groupsLabel = (res.groups || []).join(", ");
    setResponseMsg({
      type: "success",
      text: `${res.message || "Shared successfully."}${
        groupsLabel ? ` Groups: ${groupsLabel}.` : ""
      } Merchants: ${(res.merchantIdsAdded || []).length}.`,
    });
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Share with Groups
      </h2>
      <p className="text-sm text-gray-600 mb-4">
        Enter a property, contact, timetable, or cashflow PID and select groups.
        The record will be shared with all merchants in those groups.
      </p>

      <form onSubmit={handleShare} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Type
          </label>
          <select
            value={shareType}
            onChange={(e) => setShareType(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          >
            <option value="property">Property</option>
            <option value="contact">Contact</option>
            <option value="timetable">Timetable</option>
            <option value="cashflow">Cashflow</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            PID
          </label>
          <input
            type="text"
            value={pid}
            onChange={(e) => setPid(e.target.value)}
            placeholder="e.g. 3389207818011006"
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          />
        </div>

        <GroupSearchCombobox
          selectedGroups={selectedGroups}
          onChange={setSelectedGroups}
        />

        <button
          type="submit"
          disabled={loading}
          className={`px-4 py-2 rounded-lg text-white text-sm font-medium ${
            loading
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Sharing..." : "Share"}
        </button>
      </form>

      {responseMsg && (
        <p
          className={`mt-4 text-sm ${
            responseMsg.type === "error" ? "text-red-600" : "text-green-600"
          }`}
        >
          {responseMsg.text}
        </p>
      )}
    </div>
  );
}
