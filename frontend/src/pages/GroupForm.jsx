import React, { useEffect, useState } from "react";
import { createGroup, getGroup, updateGroup } from "../api";
import UserSearchCombobox from "../components/UserSearchCombobox";

function extractId(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (value.$oid) return value.$oid;
  return String(value);
}

export default function GroupForm({ editingGroupId, onCancel, onSaved }) {
  const [name, setName] = useState("");
  const [selectedUsers, setSelectedUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingGroup, setLoadingGroup] = useState(false);
  const [responseMsg, setResponseMsg] = useState(null);

  const isEditMode = Boolean(editingGroupId);

  useEffect(() => {
    if (!editingGroupId) {
      setName("");
      setSelectedUsers([]);
      setResponseMsg(null);
      return;
    }

    const loadGroup = async () => {
      setLoadingGroup(true);
      setResponseMsg(null);

      const res = await getGroup(editingGroupId);

      if (res.error) {
        setResponseMsg({ type: "error", text: res.error });
        setLoadingGroup(false);
        return;
      }

      setName(res.name || "");
      setSelectedUsers(
        (res.members || []).map((m) => ({
          _id: extractId(m._id),
          displayName: m.displayName || "",
          email: m.email || "",
          merchantId: m.merchantId || "",
        }))
      );
      setLoadingGroup(false);
    };

    loadGroup();
  }, [editingGroupId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponseMsg(null);

    const trimmedName = name.trim();
    if (!trimmedName) {
      setResponseMsg({ type: "error", text: "Group name is required." });
      return;
    }

    if (selectedUsers.length === 0) {
      setResponseMsg({ type: "error", text: "Select at least one member." });
      return;
    }

    setLoading(true);

    const payload = {
      name: trimmedName,
      userIds: selectedUsers.map((u) => u._id),
    };

    const res = isEditMode
      ? await updateGroup(editingGroupId, payload)
      : await createGroup(payload);

    setLoading(false);

    if (res.error) {
      setResponseMsg({ type: "error", text: res.error });
      return;
    }

    setResponseMsg({
      type: "success",
      text: res.message || (isEditMode ? "Group updated." : "Group created."),
    });

    if (!isEditMode) {
      setName("");
      setSelectedUsers([]);
    }

    onSaved?.();
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        {isEditMode ? "Edit Group" : "Create Group"}
      </h2>

      {loadingGroup ? (
        <p className="text-gray-500">Loading group...</p>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Group name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter group name"
              className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          <UserSearchCombobox
            selectedUsers={selectedUsers}
            onChange={setSelectedUsers}
          />

          <div className="flex flex-wrap gap-2 pt-2">
            <button
              type="submit"
              disabled={loading}
              className={`px-4 py-2 rounded-lg text-white text-sm font-medium ${
                loading
                  ? "bg-gray-300 cursor-not-allowed"
                  : "bg-green-600 hover:bg-green-700"
              }`}
            >
              {loading
                ? "Saving..."
                : isEditMode
                ? "Update group"
                : "Save group"}
            </button>

            {isEditMode && (
              <button
                type="button"
                onClick={onCancel}
                disabled={loading}
                className="px-4 py-2 rounded-lg text-sm font-medium bg-gray-200 hover:bg-gray-300 text-gray-800"
              >
                Cancel
              </button>
            )}
          </div>
        </form>
      )}

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
