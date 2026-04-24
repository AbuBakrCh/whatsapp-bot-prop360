import React, { useEffect, useState } from "react";
import { getAllPropertyFilters, deletePropertyFilter } from "../api";

export default function PropertyFiltersList() {
  const [filters, setFilters] = useState([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [deletingEmail, setDeletingEmail] = useState(null);

  const LIMIT = 5;

  const fetchFilters = async (pageNumber = 1) => {
    setIsLoading(true);

    const res = await getAllPropertyFilters(pageNumber, LIMIT);

    if (!res.error) {
      setFilters(res.data || []);
      setTotal(res.total || 0);
      setPage(pageNumber);
    }

    setIsLoading(false);
  };

  useEffect(() => {
    fetchFilters(1);
  }, []);

  const handleDelete = async (email) => {
    const confirmDelete = window.confirm(
      `Delete filter for ${email}?`
    );

    if (!confirmDelete) return;

    setDeletingEmail(email);

    const res = await deletePropertyFilter(email);

    setDeletingEmail(null);

    if (res.error) {
      alert(res.error);
      return;
    }

    fetchFilters(page);
  };

  const totalPages = Math.max(1, Math.ceil(total / LIMIT));

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">

      {/* Header + Refresh */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">
          Saved Filters
        </h2>

        <button
          onClick={() => fetchFilters(page)}
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

      {/* List */}
      {isLoading ? (
        <p>Loading...</p>
      ) : filters.length === 0 ? (
        <p className="text-gray-500">No filters found.</p>
      ) : (
        <div className="space-y-3">
          {filters.map((f) => (
            <div
              key={f._id}
              className="p-4 border rounded-lg bg-slate-50 flex justify-between items-start"
            >
              {/* LEFT CONTENT */}
              <div>
                <p><b>Email:</b> {f.clientEmail}</p>
                <p><b>Source:</b> {f.source}</p>

                {f.purpose && <p><b>Purpose:</b> {f.purpose}</p>}
                {f.category && <p><b>Category:</b> {f.category}</p>}
                {f.area && <p><b>Area:</b> {f.area}</p>}

                {f.price && (
                  <p>
                    <b>Price:</b> {f.price.min ?? "-"} → {f.price.max ?? "-"}
                  </p>
                )}

                {f.surface && (
                  <p>
                    <b>Surface:</b> {f.surface.min ?? "-"} → {f.surface.max ?? "-"}
                  </p>
                )}
              </div>

              {/* DELETE BUTTON */}
              <button
                onClick={() => handleDelete(f.clientEmail)}
                disabled={deletingEmail === f.clientEmail}
                className={`text-lg font-bold ${
                  deletingEmail === f.clientEmail
                    ? "text-gray-400 cursor-not-allowed"
                    : "text-red-600 hover:text-red-800"
                }`}
                title="Delete"
              >
                {deletingEmail === f.clientEmail ? "..." : "✕"}
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Pagination */}
      <div className="mt-6 flex flex-col items-center gap-2">

        {/* Info */}
        <div className="text-sm text-gray-600">
          {total === 0
            ? "No records"
            : `Showing ${(page - 1) * LIMIT + 1} - ${Math.min(
                page * LIMIT,
                total
              )} of ${total} records`}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchFilters(1)}
            disabled={page === 1 || isLoading}
            className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            First
          </button>

          <button
            onClick={() => fetchFilters(page - 1)}
            disabled={page === 1 || isLoading}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Prev
          </button>

          <span className="px-3 text-sm">
            Page {page} of {totalPages}
          </span>

          <button
            onClick={() => fetchFilters(page + 1)}
            disabled={page >= totalPages || isLoading}
            className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
          >
            Next
          </button>

          <button
            onClick={() => fetchFilters(totalPages)}
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