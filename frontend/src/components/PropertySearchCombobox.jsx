import React, { useEffect, useRef, useState } from "react";
import { searchProperties } from "../api";

export default function PropertySearchCombobox({
  selectedProperties,
  onChange,
  label = "Properties",
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
      const res = await searchProperties(trimmed, 15);
      if (!res.error) {
        const filtered = (res.data || []).filter(
          (p) => !selectedProperties.some((s) => s.pid === p.pid)
        );
        setResults(filtered);
        setIsOpen(true);
      } else {
        setResults([]);
      }
      setIsSearching(false);
    }, 300);

    return () => clearTimeout(timer);
  }, [query, selectedProperties]);

  const handleSelect = (property) => {
    if (selectedProperties.some((p) => p.pid === property.pid)) return;
    onChange([...selectedProperties, property]);
    setQuery("");
    setResults([]);
    setIsOpen(false);
  };

  const handleRemove = (pid) => {
    onChange(selectedProperties.filter((p) => p.pid !== pid));
  };

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => {
          if (results.length > 0) setIsOpen(true);
        }}
        placeholder="Search property by title or PID (min 2 characters)"
        className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
      />

      {isSearching && <p className="text-xs text-gray-500 mt-1">Searching...</p>}

      {isOpen && results.length > 0 && (
        <ul className="absolute z-20 mt-1 w-full bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {results.map((property) => (
            <li key={property.pid}>
              <button
                type="button"
                onClick={() => handleSelect(property)}
                className="w-full text-left px-3 py-2 hover:bg-blue-50 border-b border-gray-100 last:border-b-0"
              >
                <p className="text-sm font-medium text-gray-800">{property.label}</p>
                <p className="text-xs text-gray-500">PID {property.pid}</p>
              </button>
            </li>
          ))}
        </ul>
      )}

      {selectedProperties.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {selectedProperties.map((property) => (
            <span
              key={property.pid}
              className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-900 rounded-full text-xs max-w-full"
            >
              <span className="truncate max-w-[240px]">{property.label}</span>
              <button
                type="button"
                onClick={() => handleRemove(property.pid)}
                className="text-blue-800 hover:text-blue-950 font-bold leading-none shrink-0"
                aria-label="Remove property"
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
