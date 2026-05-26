import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { searchGroups } from "../api";

const DROPDOWN_MAX_HEIGHT = 240;
const VIEWPORT_PADDING = 8;

function extractId(value) {
  if (!value) return null;
  if (typeof value === "string") return value;
  if (value.$oid) return value.$oid;
  return String(value);
}

export default function GroupSearchCombobox({ selectedGroups, onChange }) {
  const [query, setQuery] = useState("");
  const [allGroups, setAllGroups] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownStyle, setDropdownStyle] = useState(null);
  const containerRef = useRef(null);
  const inputRef = useRef(null);

  const selectedIds = useMemo(
    () => new Set(selectedGroups.map((g) => extractId(g._id))),
    [selectedGroups]
  );

  const loadGroups = useCallback(async () => {
    setIsLoading(true);
    const res = await searchGroups("", 500);
    if (!res.error) {
      setAllGroups(
        (res.data || []).map((g) => ({
          ...g,
          _id: extractId(g._id),
        }))
      );
    }
    setIsLoading(false);
  }, []);

  const updateDropdownPosition = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom - VIEWPORT_PADDING;
    const spaceAbove = rect.top - VIEWPORT_PADDING;
    const openUpward =
      spaceBelow < DROPDOWN_MAX_HEIGHT && spaceAbove > spaceBelow;
    const available = openUpward ? spaceAbove : spaceBelow;
    const maxHeight = Math.max(120, Math.min(DROPDOWN_MAX_HEIGHT, available));

    setDropdownStyle({
      position: "fixed",
      left: rect.left,
      width: rect.width,
      maxHeight,
      zIndex: 9999,
      ...(openUpward
        ? { bottom: window.innerHeight - rect.top + 4 }
        : { top: rect.bottom + 4 }),
    });
  }, []);

  const openDropdown = useCallback(async () => {
    setIsOpen(true);
    await loadGroups();
    requestAnimationFrame(updateDropdownPosition);
  }, [loadGroups, updateDropdownPosition]);

  const closeDropdown = useCallback(() => {
    setIsOpen(false);
    setDropdownStyle(null);
  }, []);

  useEffect(() => {
    const handleClickOutside = (event) => {
      const target = event.target;
      if (containerRef.current?.contains(target)) return;
      if (target.closest?.("[data-group-dropdown]")) return;
      closeDropdown();
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [closeDropdown]);

  useEffect(() => {
    if (!isOpen) return;

    updateDropdownPosition();
    const onReposition = () => updateDropdownPosition();
    window.addEventListener("resize", onReposition);
    window.addEventListener("scroll", onReposition, true);

    return () => {
      window.removeEventListener("resize", onReposition);
      window.removeEventListener("scroll", onReposition, true);
    };
  }, [isOpen, updateDropdownPosition, allGroups.length, isLoading]);

  const filteredGroups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return allGroups.filter((group) => {
      if (selectedIds.has(extractId(group._id))) return false;
      if (!q) return true;
      return (group.name || "").toLowerCase().includes(q);
    });
  }, [allGroups, query, selectedIds]);

  const handleSelect = (group) => {
    const groupId = extractId(group._id);
    if (selectedIds.has(groupId)) return;
    onChange([...selectedGroups, { ...group, _id: groupId }]);
    setQuery("");
    closeDropdown();
  };

  const handleRemove = (groupId) => {
    onChange(selectedGroups.filter((g) => extractId(g._id) !== groupId));
  };

  const dropdown =
    isOpen &&
    dropdownStyle &&
    createPortal(
      <div
        data-group-dropdown
        style={dropdownStyle}
        className="bg-white border border-gray-200 rounded-lg shadow-lg overflow-y-auto"
      >
        {isLoading ? (
          <p className="px-3 py-2 text-sm text-gray-500">Loading groups...</p>
        ) : filteredGroups.length === 0 ? (
          <p className="px-3 py-2 text-sm text-gray-500">
            {allGroups.length === 0
              ? "No groups available. Create a group first."
              : "No matching groups."}
          </p>
        ) : (
          <ul>
            {filteredGroups.map((group) => {
              const groupId = extractId(group._id);
              return (
                <li key={groupId}>
                  <button
                    type="button"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => handleSelect(group)}
                    className="w-full text-left px-3 py-2 hover:bg-green-50 border-b border-gray-100 last:border-b-0"
                  >
                    <p className="text-sm font-medium text-gray-800">
                      {group.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {group.memberCount ?? 0} member(s)
                    </p>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>,
      document.body
    );

  return (
    <div ref={containerRef} className="relative">
      <label className="block text-sm font-medium text-gray-700 mb-1">
        Select groups
      </label>
      <div className="flex gap-2">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={openDropdown}
          placeholder="Search or pick from the list"
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
        />
        <button
          type="button"
          onClick={() => (isOpen ? closeDropdown() : openDropdown())}
          className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-gray-50 hover:bg-gray-100 text-gray-700 shrink-0"
          aria-label={isOpen ? "Close groups list" : "Open groups list"}
          aria-expanded={isOpen}
        >
          {isOpen ? "▲" : "▼"}
        </button>
      </div>

      {dropdown}

      {selectedGroups.length > 0 && (
        <div className="flex flex-wrap gap-2 mt-3">
          {selectedGroups.map((group) => {
            const groupId = extractId(group._id);
            return (
              <span
                key={groupId}
                className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs"
              >
                <span>{group.name}</span>
                <button
                  type="button"
                  onClick={() => handleRemove(groupId)}
                  className="text-blue-700 hover:text-blue-900 font-bold leading-none"
                  aria-label="Remove group"
                >
                  ×
                </button>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
