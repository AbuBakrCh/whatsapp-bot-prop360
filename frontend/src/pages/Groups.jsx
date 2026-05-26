import React, { useState } from "react";
import { Link } from "react-router-dom";
import GroupForm from "./GroupForm";
import GroupsList from "./GroupsList";
import ShareWithGroups from "./ShareWithGroups";

export default function Groups() {
  const [editingGroupId, setEditingGroupId] = useState(null);
  const [listRefreshKey, setListRefreshKey] = useState(0);

  const handleSaved = () => {
    setEditingGroupId(null);
    setListRefreshKey((k) => k + 1);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <div className="p-4 bg-white border-b border-green-200 flex justify-between items-center shadow-sm">
        <h1 className="text-lg font-semibold text-green-600">Groups</h1>
        <Link
          to="/"
          className="px-3 py-1.5 bg-gray-200 hover:bg-gray-300 text-gray-800 rounded-lg text-sm transition"
        >
          ← Back to Dashboard
        </Link>
      </div>

      <div className="flex-1 p-6 space-y-8">
        <GroupForm
          editingGroupId={editingGroupId}
          onCancel={() => setEditingGroupId(null)}
          onSaved={handleSaved}
        />
        <GroupsList
          refreshKey={listRefreshKey}
          onEdit={(groupId) => {
            setEditingGroupId(groupId);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
        <ShareWithGroups />
      </div>
    </div>
  );
}
