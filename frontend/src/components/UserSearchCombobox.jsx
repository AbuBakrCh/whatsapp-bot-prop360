import React, { useEffect, useRef, useState } from "react";
import { searchGroupUsers } from "../api";

export default function UserSearchCombobox({ selectedUsers, onChange }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (containerRef.current && !containerRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 2) {
      setResults([]);
      setIsSearching(false);
      return;
    }

    setIsSearching(true);
    const timer = setTimeout(async () => {
      const excludeIds = selectedUsers.map((u) => u._id);
      const res = await searchGroupUsers(trimmed, 10, excludeIds);

      if (!res.error) {
        setResults(res.data || []);
        setIsOpen(true);
      } else {
        setResults([]);
      }
      setIsSearching(false);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, selectedUsers]);

  const handleSelect = (user) => {
    if (selectedUsers.some((u) => u._id === user._id)) return;
    onChange([...selectedUsers, user]);
    setQuery("");
    setResults([]);
    setIsOpen(false);
  };

  const handleRemove = (userId) => {
    onChange(selectedUsers.filter((u) => u._id !== userId));
  };

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Add members
      </label>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => {
          if (results.length > 0) setIsOpen(true);
        }}
        placeholder="Search by name or email (min 2 characters)"
        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
      />

      {isSearching && (
        <p className="text-xs text-gray-500 mt-1">Searching...</p>
      )}

      {isOpen && results.length > 0 && (
        <ul className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {results.map((user) => (
            <li key={user._id}>
              <button
                type="button"
                onClick={() => handleSelect(user)}
                className="w-full text-left px-3 py-2 hover:bg-green-50 border-b border-gray-100 last:border-b-0"
              >
                <p className="text-sm font-medium text-gray-800">
                  {user.displayName || "Unnamed"}
                </p>
                <p className="text-xs text-gray-500">{user.email}</p>
              </button>
            </li>
          ))}
        </ul>
      )}

      {selectedUsers.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {selectedUsers.map((user) => (
            <span
              key={user._id}
              className="inline-flex items-center gap-1 px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs"
            >
              <span>
                {user.displayName || user.email}
                {user.email && user.displayName ? ` (${user.email})` : ""}
              </span>
              <button
                type="button"
                onClick={() => handleRemove(user._id)}
                className="text-green-700 hover:text-green-900 font-bold leading-none"
                aria-label="Remove member"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
