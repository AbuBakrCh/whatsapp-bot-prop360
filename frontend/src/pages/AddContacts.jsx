import React, { useState } from "react";
import { addContacts, deleteContacts } from "../api";

const YES_NO_OPTIONS = ["Yes", "No"];

const CONTACT_TYPE_OPTIONS = [
  "Client",
  "Supplier",
  "Lead",
  "Tenant",
  "Real Estate  Company",
  "Management Company",
  "Cleaning Company",
  "Pest Control Company",
  "Government Institution",
];

const LEAD_ROLE_OPTIONS = [
  "Interviewed Remotely",
  "Property Viewed",
  "Negotiation stage",
];

const ENTRY_TYPE_OPTIONS = ["Individual", "Corporate"];

const emptyFilters = () => ({
  searchForProperty: false,
  doesHeHaveProperty: false,
  taxClient: "",
  insuranceCustomer: "",
  rentalCustomer: "",
  managementClient: "",
  contactTypes: [],
  leadRoles: [],
  entryTypes: [],
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

export default function AddingContacts() {
  const [sourceMerchantEmail, setSourceMerchantEmail] = useState("");
  const [targetMerchantEmails, setTargetMerchantEmails] = useState("");
  const [filters, setFilters] = useState(emptyFilters);
  const [loading, setLoading] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");

  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
  };

  const handleProcess = async (actionType) => {
    if (!targetMerchantEmails) {
      setResponseMsg("Target Merchant Emails are required.");
      return;
    }

    if (actionType === "add" && !sourceMerchantEmail) {
      setResponseMsg("Source Merchant Email is required for adding contacts.");
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
          searchForProperty: filters.searchForProperty,
          doesHeHaveProperty: filters.doesHeHaveProperty,
          taxClient: filters.taxClient || undefined,
          insuranceCustomer: filters.insuranceCustomer || undefined,
          rentalCustomer: filters.rentalCustomer || undefined,
          managementClient: filters.managementClient || undefined,
          contactTypes: filters.contactTypes,
          leadRoles: filters.leadRoles,
          entryTypes: filters.entryTypes,
        },
      };

      let res;

      if (actionType === "add") {
        res = await addContacts(payload);
      } else if (actionType === "delete") {
        res = await deleteContacts(payload);
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
      setResponseMsg(err.response?.data?.error || "Failed to process contacts.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Manage Contacts</h2>

      {/* Main Inputs */}
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

      {/* Filters */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xl font-semibold text-gray-700">Filters</h3>
        <button
          type="button"
          onClick={() => setFilters(emptyFilters())}
          className="text-sm text-slate-500 hover:text-slate-700 underline"
        >
          Reset filters
        </button>
      </div>
      <p className="text-sm text-slate-500 mb-4">
        Leave filters empty / unchecked to ignore them. Selected filters are
        combined with AND.
      </p>

      <div className="flex flex-col gap-4 mb-6">
        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={filters.searchForProperty}
            onChange={(e) => updateFilter("searchForProperty", e.target.checked)}
            className="w-4 h-4"
          />
          Search For Property?
        </label>

        <label className="flex items-center gap-2 text-gray-700">
          <input
            type="checkbox"
            checked={filters.doesHeHaveProperty}
            onChange={(e) => updateFilter("doesHeHaveProperty", e.target.checked)}
            className="w-4 h-4"
          />
          Does He Have Property?
        </label>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          <YesNoSelect
            label="Tax Client"
            value={filters.taxClient}
            onChange={(v) => updateFilter("taxClient", v)}
          />
          <YesNoSelect
            label="Insurance Customer"
            value={filters.insuranceCustomer}
            onChange={(v) => updateFilter("insuranceCustomer", v)}
          />
          <YesNoSelect
            label="Rental Customer"
            value={filters.rentalCustomer}
            onChange={(v) => updateFilter("rentalCustomer", v)}
          />
          <YesNoSelect
            label="Management Client"
            value={filters.managementClient}
            onChange={(v) => updateFilter("managementClient", v)}
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
          <MultiSelectCheckboxGroup
            label="Contact Type"
            options={CONTACT_TYPE_OPTIONS}
            values={filters.contactTypes}
            onChange={(v) => updateFilter("contactTypes", v)}
          />
          <MultiSelectCheckboxGroup
            label="Lead Role"
            options={LEAD_ROLE_OPTIONS}
            values={filters.leadRoles}
            onChange={(v) => updateFilter("leadRoles", v)}
          />
          <MultiSelectCheckboxGroup
            label="Entry Type"
            options={ENTRY_TYPE_OPTIONS}
            values={filters.entryTypes}
            onChange={(v) => updateFilter("entryTypes", v)}
          />
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={() => handleProcess("add")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading
              ? "bg-green-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {loading ? "Processing..." : "Share Contact"}
        </button>

        <button
          onClick={() => handleProcess("delete")}
          disabled={loading}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium transition ${
            loading
              ? "bg-red-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {loading ? "Processing..." : "Unshare Contact"}
        </button>
      </div>

      {/* Response Message */}
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
