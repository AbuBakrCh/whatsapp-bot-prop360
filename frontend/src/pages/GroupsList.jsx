import React, { useEffect, useState } from "react";
import { deleteGroup, getGroups } from "../api";

function extractId(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (value.$oid) return value.$oid;
  return String(value);
}

function formatMemberEmails(members, maxShown = 3) {
  if (!members?.length) return "No members";
  const emails = members.map((m) => m.email).filter(Boolean);
  if (emails.length <= maxShown) return emails.join(", ");
  return `${emails.slice(0, maxShown).join(", ")} +${emails.length - maxShown} more`;
}

export default function GroupsList({ refreshKey = 0, onEdit }) {
  const [groups, setGroups] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);

  const LIMIT = 5;

  const fetchGroups = async (pageNumber = 1) => {
    setIsLoading(true);

    const res = await getGroups(pageNumber, LIMIT);

    if (!res.error) {
      setGroups(res.data || []);
      setTotal(res.total || 0);
      setPage(pageNumber);
    }

    setIsLoading(false);
  };

  useEffect(() => {
    fetchGroups(1);
  }, [refreshKey]);

  const handleDelete = async (groupId, groupName) => {
    const confirmDelete = window.confirm(`Delete group "${groupName}"?`);
    if (!confirmDelete) return;

    setDeletingId(groupId);

    const res = await deleteGroup(groupId);

    setDeletingId(null);

    if (res.error) {
      alert(res.error);
      return;
    }

    fetchGroups(page);
  };

  const totalPages = Math.max(1, Math.ceil(total / LIMIT));

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Groups</h2>

        <button
          onClick={() => fetchGroups(page)}
          disabled={isLoading}
          className={`px-3 py-1 rounded-md text-white text-sm ${
            isLoading
              ? "bg-gray-300 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {isLoading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {isLoading ? (
        <p>Loading...</p>
      ) : groups.length === 0 ? (
        <p className="text-gray-500">No groups found.</p>
      ) : (
        <div className="space-y-3">
          {groups.map((group) => {
            const groupId = extractId(group._id);
            return (
              <div
                key={groupId}
                className="p-4 border rounded-lg bg-slate-50 flex justify-between items-start gap-4"
              >
                <div className="min-w-0 flex-1">
                  <p className="font-semibold text-gray-800">{group.name}</p>
                  <p className="text-sm text-gray-600 mt-1">
                    <b>Members:</b> {group.members?.length ?? 0}
                  </p>
                  <p className="text-sm text-gray-500 mt-1 truncate">
                    {formatMemberEmails(group.members)}
                  </p>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => onEdit?.(groupId)}
                    className="px-3 py-1 text-sm bg-gray-200 hover:bg-gray-300 rounded-md text-gray-800"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDelete(groupId, group.name)}
                    disabled={deletingId === groupId}
                    className={`px-3 py-1 text-sm rounded-md text-white ${
                      deletingId === groupId
                        ? "bg-gray-300 cursor-not-allowed"
                        : "bg-red-600 hover:bg-red-700"
                    }`}
                  >
                    {deletingId === groupId ? "..." : "Delete"}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-6 flex flex-col items-center gap-2">
        <div className="text-sm text-gray-600">
          {total === 0
            ? "No records"
            : `Showing ${(page - 1) * LIMIT + 1} - ${Math.min(
                page * LIMIT,
                total
              )} of ${total} records`}
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchGroups(1)}
            disabled={page === 1 || isLoading}
            className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            First
          </button>

          <button
            onClick={() => fetchGroups(page - 1)}
            disabled={page === 1 || isLoading}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Prev
          </button>

          <span className="px-3 text-sm">
            Page {page} of {totalPages}
          </span>

          <button
            onClick={() => fetchGroups(page + 1)}
            disabled={page >= totalPages || isLoading}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Next
          </button>

          <button
            onClick={() => fetchGroups(totalPages)}
            disabled={page >= totalPages || isLoading}
            className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Last
          </button>
        </div>
      </div>
    </div>
  );
}
