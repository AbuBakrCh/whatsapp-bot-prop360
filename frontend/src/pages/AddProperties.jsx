import React, { useState } from "react";
import { addProperties, deleteProperties } from "../api";
import UserSearchCombobox from "../components/UserSearchCombobox";

const YES_NO_OPTIONS = ["Yes", "No"];

const DIVISION_OPTIONS = [
  "MANAGEMENT",
  "RENTAL",
  "LAWYERS",
  "BUYING",
  "SELLING",
  "INSURANCE",
  "RENOVATION",
  "MARKETING",
  "REAL ESTATE",
  "NO DIVISION",
];

const INSURANCE_STATUS_OPTIONS = ["Insured", "Not Insured"];

const CITY_OPTIONS = [
  "Athens",
  "Patras",
  "Piraeus",
  "Euboea",
  "Aigio",
  "Corinth",
  "Antiparos",
  "Malesina",
  "Porto Cheli",
  "Thessaloniki",
  "Halkidiki",
  "Diakopto",
  "Kalavryta",
  "Chania (Crete)",
  "Zakynthos",
  "Mintilogli",
  "Karia",
  "Paralia",
  "Derveni",
  "Mpouka",
  "Skioessa",
  "Ovria",
  "Loutraki",
  "Galatas",
  "Kavala",
  "Larissa",
  "Serres",
  "Drama",
  "Veria",
  "Edessa",
  "Naoussa",
  "Giannitsa",
  "Kilkis",
  "Katerini",
  "Kozani",
  "Ptolemaida",
  "Florina",
  "Kastoria",
  "Alexandroupoli (Dedeağaç)",
  "Komotini (Gümülcine)",
  "Xanthi (İskeçe)",
  "Orestiada",
  "Didymoteicho",
  "Volos",
  "Trikala",
  "Karditsa",
  "Kalambaka",
  "Ioannina (Yanya)",
  "Arta",
  "Preveze",
  "Igoumenitsa",
  "Lamia",
  "Livadia",
  "Chalkida",
  "Karpenisi",
  "Amfissa",
  "Tripoli",
  "Kalamata",
  "Sparta",
  "Argos",
  "Megalopoli",
  "Agrinio",
  "Messolonghi",
  "Amaliada",
  "(Heraklion) Kandiye",
  "Chania (Hanya)",
  "Rethymno (Resmo)",
  "Agios Nikolaos",
  "Ierapetra",
  "Rhodes (Rodos)",
  "Kos (İstanköy)",
  "Mytilene (Midilli)",
  "Chios (Sakiz)",
  "Samos",
  "Syros (Ermoupoli)",
  "Naxos",
  "Paros",
  "Santorini (Thira)",
  "Mikonos",
  "Korfu",
  "Lefkada",
  "Kefalonya (Argostoli)",
  "Sitia (Crete)",
];

const emptyFilters = () => ({
  isPublic: "",
  createdByMode: "",
  cities: [],
  division: "",
  propertyManagement: "",
  taxManagement: "",
  purchasingDateMode: "",
  purchasingDateAfter: "",
  purchasingDateBefore: "",
  leaseStatus: "",
  insuranceStatus: "",
});

function YesNoSelect({ label, value, onChange }) {
  return (
    <label className="flex flex-col gap-1 text-gray-700">
      <span className="text-sm font-medium">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border border-slate-300 rounded-md px-3 py-2 bg-white"
      >
        <option value="">Any</option>
        {YES_NO_OPTIONS.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function SelectFilter({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1 text-gray-700">
      <span className="text-sm font-medium">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="border border-slate-300 rounded-md px-3 py-2 bg-white"
      >
        <option value="">Any</option>
        {options.map((opt) => (
          <option key={opt.value ?? opt} value={opt.value ?? opt}>
            {opt.label ?? opt}
          </option>
        ))}
      </select>
    </label>
  );
}

function MultiSelectCheckboxGroup({ label, options, values, onChange }) {
  const toggle = (option) => {
    if (values.includes(option)) {
      onChange(values.filter((v) => v !== option));
    } else {
      onChange([...values, option]);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-gray-700">{label}</span>
        {values.length > 0 && (
          <button
            type="button"
            onClick={() => onChange([])}
            className="text-xs text-slate-500 hover:text-slate-700 underline"
          >
            Clear
          </button>
        )}
      </div>
      <div className="border border-slate-300 rounded-md px-3 py-2 max-h-40 overflow-y-auto bg-white">
        <div className="flex flex-col gap-2">
          {options.map((option) => (
            <label
              key={option}
              className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer"
            >
              <input
                type="checkbox"
                checked={values.includes(option)}
                onChange={() => toggle(option)}
                className="w-4 h-4"
              />
              {option}
            </label>
          ))}
        </div>
      </div>
      <span className="text-xs text-slate-500">
        {values.length === 0
          ? "No selection — filter ignored"
          : `${values.length} selected`}
      </span>
    </div>
  );
}

export default function AddingProperties() {
  const [sourceMerchantEmail, setSourceMerchantEmail] = useState("");
  const [targetMerchantEmails, setTargetMerchantEmails] = useState("");
  const [filters, setFilters] = useState(emptyFilters);
  const [createdByOwners, setCreatedByOwners] = useState([]);
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleCreatedByModeChange = (mode) => {
    updateFilter("createdByMode", mode);
    if (mode !== "owners") {
      setCreatedByOwners([]);
    }
  };

  const handlePurchasingDateModeChange = (mode) => {
    setFilters((prev) => ({
      ...prev,
      purchasingDateMode: mode,
      purchasingDateAfter: mode === "range" ? prev.purchasingDateAfter : "",
      purchasingDateBefore: mode === "range" ? prev.purchasingDateBefore : "",
    }));
  };

  const handleProcess = async (actionType) => {
    if (!targetMerchantEmails) {
      setResponseMsg("Target Merchant Emails are required.");
      return;
    }

    if (actionType === "add" && !sourceMerchantEmail) {
      setResponseMsg("Source Merchant Email is required for sharing properties.");
      return;
    }

    if (
      filters.createdByMode === "owners" &&
      createdByOwners.filter((u) => u.firebaseId).length === 0
    ) {
      setResponseMsg("Select at least one owner when filtering by specific owners.");
      return;
    }

    if (
      filters.purchasingDateMode === "range" &&
      !filters.purchasingDateAfter &&
      !filters.purchasingDateBefore
    ) {
      setResponseMsg("Set at least one of Purchasing Date after/before for a date range.");
      return;
    }

    setLoading(true);
    setResponseMsg("");

    try {
      const payload = {
        sourceMerchantEmail: sourceMerchantEmail.trim().toLowerCase(),
        targetMerchantEmails: targetMerchantEmails
          .split(",")
          .map((email) => email.trim().toLowerCase())
          .filter(Boolean),
        filters: {
          isPublic: filters.isPublic || undefined,
          createdByMode: filters.createdByMode || undefined,
          createdByOwnerIds:
            filters.createdByMode === "owners"
              ? createdByOwners.map((u) => u.firebaseId).filter(Boolean)
              : undefined,
          cities: filters.cities,
          division: filters.division || undefined,
          propertyManagement: filters.propertyManagement || undefined,
          taxManagement: filters.taxManagement || undefined,
          purchasingDateMode: filters.purchasingDateMode || undefined,
          purchasingDateAfter: filters.purchasingDateAfter || undefined,
          purchasingDateBefore: filters.purchasingDateBefore || undefined,
          leaseStatus: filters.leaseStatus || undefined,
          insuranceStatus: filters.insuranceStatus || undefined,
        },
      };

      let res;

      if (actionType === "add") {
        res = await addProperties(payload);
      } else if (actionType === "delete") {
        res = await deleteProperties(payload);
      }

      if (res.error) {
        setResponseMsg(res.error);
      } else if (res.message) {
        setResponseMsg(
          res.updated !== undefined
            ? `${res.message} (updated: ${res.updated})`
            : res.message
        );
      } else {
        setResponseMsg("Operation completed.");
      }
    } catch (err) {
      setResponseMsg(err.response?.data?.error || "Failed to process properties.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Manage Properties</h2>

      <div className="flex flex-col gap-4 mb-6">
        <input
          type="text"
          value={sourceMerchantEmail}
          onChange={(e) => setSourceMerchantEmail(e.target.value)}
          placeholder="Source Merchant Email"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <textarea
          value={targetMerchantEmails}
          onChange={(e) => setTargetMerchantEmails(e.target.value)}
          placeholder="Target Merchant Emails (comma separated)"
          className="border border-slate-300 rounded-md px-3 py-2 h-24 resize-none"
        />
      </div>

      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xl font-semibold text-gray-700">Filters</h3>
        <button
          type="button"
          onClick={() => {
            setFilters(emptyFilters());
            setCreatedByOwners([]);
          }}
          className="text-sm text-slate-500 hover:text-slate-700 underline"
        >
          Reset filters
        </button>
      </div>
      <p className="text-sm text-slate-500 mb-4">
        Leave filters empty / on Any to ignore them. Selected filters are combined
        with AND.
      </p>

      <div className="flex flex-col gap-4 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <SelectFilter
            label="Is Public"
            value={filters.isPublic}
            onChange={(v) => updateFilter("isPublic", v)}
            options={[
              { value: "true", label: "true" },
              { value: "false", label: "false" },
            ]}
          />
          <SelectFilter
            label="Division"
            value={filters.division}
            onChange={(v) => updateFilter("division", v)}
            options={DIVISION_OPTIONS}
          />
          <YesNoSelect
            label="Property Management"
            value={filters.propertyManagement}
            onChange={(v) => updateFilter("propertyManagement", v)}
          />
          <YesNoSelect
            label="Tax Management"
            value={filters.taxManagement}
            onChange={(v) => updateFilter("taxManagement", v)}
          />
          <SelectFilter
            label="Insurance Status"
            value={filters.insuranceStatus}
            onChange={(v) => updateFilter("insuranceStatus", v)}
            options={INSURANCE_STATUS_OPTIONS}
          />
          <label className="flex flex-col gap-1 text-gray-700">
            <span className="text-sm font-medium">Lease Status</span>
            <input
              type="text"
              value={filters.leaseStatus}
              onChange={(e) => updateFilter("leaseStatus", e.target.value)}
              placeholder="Exact match (leave empty to ignore)"
              className="border border-slate-300 rounded-md px-3 py-2"
            />
          </label>
        </div>

        <label className="flex flex-col gap-1 text-gray-700">
          <span className="text-sm font-medium">Created By</span>
          <select
            value={filters.createdByMode}
            onChange={(e) => handleCreatedByModeChange(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 bg-white"
          >
            <option value="">Any</option>
            <option value="api">Created by API (starts with api-)</option>
            <option value="owners">Specific owners</option>
          </select>
        </label>

        {filters.createdByMode === "owners" && (
          <UserSearchCombobox
            label="Owners"
            selectedUsers={createdByOwners}
            onChange={setCreatedByOwners}
            requireFirebaseId
          />
        )}

        <label className="flex flex-col gap-1 text-gray-700">
          <span className="text-sm font-medium">Purchasing Date</span>
          <select
            value={filters.purchasingDateMode}
            onChange={(e) => handlePurchasingDateModeChange(e.target.value)}
            className="border border-slate-300 rounded-md px-3 py-2 bg-white"
          >
            <option value="">Any</option>
            <option value="empty">Empty</option>
            <option value="notEmpty">Not empty</option>
            <option value="range">Date range (after / before)</option>
          </select>
        </label>

        {filters.purchasingDateMode === "range" && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <label className="flex flex-col gap-1 text-gray-700">
              <span className="text-sm font-medium">Date after</span>
              <input
                type="date"
                value={filters.purchasingDateAfter}
                onChange={(e) => updateFilter("purchasingDateAfter", e.target.value)}
                className="border border-slate-300 rounded-md px-3 py-2"
              />
            </label>
            <label className="flex flex-col gap-1 text-gray-700">
              <span className="text-sm font-medium">Date before</span>
              <input
                type="date"
                value={filters.purchasingDateBefore}
                onChange={(e) =>
                  updateFilter("purchasingDateBefore", e.target.value)
                }
                className="border border-slate-300 rounded-md px-3 py-2"
              />
            </label>
          </div>
        )}

        <MultiSelectCheckboxGroup
          label="City"
          options={CITY_OPTIONS}
          values={filters.cities}
          onChange={(v) => updateFilter("cities", v)}
        />
      </div>

      <div className="flex gap-4">
        <button
          onClick={() => handleProcess("add")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading ? "bg-green-300 cursor-not-allowed" : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Processing..." : "Share Property"}
        </button>

        <button
          onClick={() => handleProcess("delete")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading ? "bg-red-300 cursor-not-allowed" : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading ? "Processing..." : "Unshare Property"}
        </button>
      </div>

      {responseMsg && (
        <p
          className={`mt-4 ${
            responseMsg.toLowerCase().includes("success")
              ? "text-green-600"
              : "text-red-600"
          }`}
        >
          {responseMsg}
        </p>
      )}
    </div>
  );
}
