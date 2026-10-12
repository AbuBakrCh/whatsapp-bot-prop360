import React, { useEffect, useRef, useState } from "react";
import { searchContacts } from "../api";

export default function ContactSearchCombobox({
  selectedContact,
  onChange,
  label = "Property owner (contact)",
}) {
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
      const res = await searchContacts(trimmed, 15);
      if (!res.error) {
        setResults(res.data || []);
        setIsOpen(true);
      } else {
        setResults([]);
      }
      setIsSearching(false);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  const handleSelect = (contact) => {
    onChange(contact);
    setQuery("");
    setResults([]);
    setIsOpen(false);
  };

  const handleClear = () => {
    onChange(null);
    setQuery("");
  };

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      {selectedContact ? (
        <div className="flex items-center gap-2 border border-green-200 bg-green-50 rounded-lg px-3 py-2">
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-800 truncate">
              {selectedContact.label}
            </p>
            <p className="text-xs text-gray-500">PID {selectedContact.pid}</p>
          </div>
          <button
            type="button"
            onClick={handleClear}
            className="text-sm text-red-600 hover:text-red-800"
          >
            Clear
          </button>
        </div>
      ) : (
        <>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => {
              if (results.length > 0) setIsOpen(true);
            }}
            placeholder="Search contact by name, email, phone, or PID (min 2 characters)"
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
          />
          {isSearching && (
            <p className="text-xs text-gray-500 mt-1">Searching...</p>
          )}
          {isOpen && results.length > 0 && (
            <ul className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {results.map((contact) => (
                <li key={contact.pid}>
                  <button
                    type="button"
                    onClick={() => handleSelect(contact)}
                    className="w-full text-left px-3 py-2 hover:bg-green-50 border-b border-gray-100 last:border-b-0"
                  >
                    <p className="text-sm font-medium text-gray-800">{contact.label}</p>
                    <p className="text-xs text-gray-500">PID {contact.pid}</p>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
