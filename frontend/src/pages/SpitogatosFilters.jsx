import React, { useState } from "react";
import { addPropertyFilter } from "../api";
import { getPropertyFilter } from "../api";

export default function SpitogatosFilters() {
  const [email, setEmail] = useState("");

  const [purpose, setPurpose] = useState("");
  const [category, setCategory] = useState("");
  const [type, setType] = useState("");

  const [priceMin, setPriceMin] = useState("");
  const [priceMax, setPriceMax] = useState("");

  const [surfaceMin, setSurfaceMin] = useState("");
  const [surfaceMax, setSurfaceMax] = useState("");

  const [isSaving, setIsSaving] = useState(false);
  const [responseMsg, setResponseMsg] = useState("");
  const [isLoading, setIsLoading] = useState(false);


  const handleLoad = async () => {
  if (!email) {
    setResponseMsg("Enter email to load filter.");
    return;
  }

  setIsLoading(true);
  setResponseMsg("Loading filter...");

  try {
    const res = await getPropertyFilter(email);

    if (res.error) {
      setResponseMsg(res.error);
      return;
    }

    if (res.message) {
      setResponseMsg(res.message);
      return;
    }

    setPurpose(res.purpose || "");
    setCategory(res.category || "");
    setType(res.type || "");

    setPriceMin(res.price?.min ?? "");
    setPriceMax(res.price?.max ?? "");

    setSurfaceMin(res.surface?.min ?? "");
    setSurfaceMax(res.surface?.max ?? "");

    setResponseMsg("Filter loaded successfully ✅");
  } catch (err) {
    setResponseMsg("Failed to load filter.");
  } finally {
    setIsLoading(false);
  }
};


  const handleSave = async () => {
  if (!email) {
    setResponseMsg("Client email is required.");
    return;
  }

  // Convert safely
  const pMin = priceMin !== "" ? Number(priceMin) : null;
  const pMax = priceMax !== "" ? Number(priceMax) : null;

  const sMin = surfaceMin !== "" ? Number(surfaceMin) : null;
  const sMax = surfaceMax !== "" ? Number(surfaceMax) : null;

  // ---------------------------
  // Validation
  // ---------------------------

  // Price validation
  if (pMin !== null && pMin < 0) {
    setResponseMsg("Price min cannot be negative.");
    return;
  }

  if (pMax !== null && pMax < 0) {
    setResponseMsg("Price max cannot be negative.");
    return;
  }

  if (pMin !== null && pMax !== null && pMin > pMax) {
    setResponseMsg("Price min cannot be greater than max.");
    return;
  }

  // Surface validation
  if (sMin !== null && sMin < 0) {
    setResponseMsg("Surface min cannot be negative.");
    return;
  }

  if (sMax !== null && sMax < 0) {
    setResponseMsg("Surface max cannot be negative.");
    return;
  }

  if (sMin !== null && sMax !== null && sMin > sMax) {
    setResponseMsg("Surface min cannot be greater than max.");
    return;
  }

  // ---------------------------
  // Build payload
  // ---------------------------
  setIsSaving(true);
  setResponseMsg("Saving filters...");

  const payload = { email };

  if (purpose) payload.purpose = purpose;
  if (category) payload.category = category;
  if (type) payload.type = type;

  if (pMin !== null || pMax !== null) {
    payload.price = {};
    if (pMin !== null) payload.price.min = pMin;
    if (pMax !== null) payload.price.max = pMax;
  }

  if (sMin !== null || sMax !== null) {
    payload.surface = {};
    if (sMin !== null) payload.surface.min = sMin;
    if (sMax !== null) payload.surface.max = sMax;
  }

  try {
    const res = await addPropertyFilter(payload);

    if (res.error) {
      setResponseMsg(res.error);
    } else {
      setResponseMsg("Filters saved successfully ✅");

      // reset
      setEmail("");
      setPurpose("");
      setCategory("");
      setType("");
      setPriceMin("");
      setPriceMax("");
      setSurfaceMin("");
      setSurfaceMax("");
    }
  } catch (err) {
    setResponseMsg("Failed to save filters.");
  } finally {
    setIsSaving(false);
  }
};

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Store Client Filters
      </h2>

      {/* Email + Load */}
      <div className="flex gap-3 mb-4">
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Client Email"
          className="w-full border border-slate-300 rounded-md px-3 py-2"
        />

        <button
          onClick={handleLoad}
          disabled={isLoading}
          className={`px-4 py-2 rounded-md text-white font-medium ${
            isLoading
              ? "bg-gray-300"
              : "bg-gray-600 hover:bg-gray-700"
          }`}
        >
          {isLoading ? "Loading..." : "Load"}
        </button>
      </div>

      {/* Inputs */}
      <div className="flex flex-col gap-4 mb-6">
        {/* Purpose */}
        <select
          value={purpose}
          onChange={(e) => setPurpose(e.target.value)}
          className={`border border-slate-300 rounded-md px-3 py-2 ${
            !purpose ? "text-gray-400" : "text-black"
          }`}
        >
          <option value="">Select Purpose</option>
          <option value="sale">Sale</option>
          <option value="rent">Rent</option>
        </select>

        {/* Category */}
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className={`border border-slate-300 rounded-md px-3 py-2 ${
            !category ? "text-gray-400" : "text-black"
          }`}
        >
          <option value="">Select Category</option>
          <option value="home">Home</option>
          <option value="commercial">Commercial</option>
          <option value="land">Land</option>
          <option value="other">Other</option>
          <option value="new-development">New development</option>
          <option value="student-housing">Student housing</option>
        </select>

        {/* Type */}
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          className={`border border-slate-300 rounded-md px-3 py-2 ${
            !type ? "text-gray-400" : "text-black"
          }`}
        >
          <option value="">Select Property Type</option>
          <option value="apartment">Apartment</option>
          <option value="studio">Studio</option>
          <option value="villa">Villa</option>
          <option value="loft">Loft</option>
          <option value="bungalow">Bungalow</option>
          <option value="maisonette">Maisonette</option>
          <option value="detached_house">Detached house</option>
        </select>

        {/* Price */}
        <div className="flex gap-3">
          <input
            type="number"
            value={priceMin}
            onChange={(e) => setPriceMin(e.target.value)}
            placeholder="Min Price"
            className="w-1/2 border border-slate-300 rounded-md px-3 py-2"
          />
          <input
            type="number"
            value={priceMax}
            onChange={(e) => setPriceMax(e.target.value)}
            placeholder="Max Price"
            className="w-1/2 border border-slate-300 rounded-md px-3 py-2"
          />
        </div>

        {/* Surface */}
        <div className="flex gap-3">
          <input
            type="number"
            value={surfaceMin}
            onChange={(e) => setSurfaceMin(e.target.value)}
            placeholder="Min Surface (sqm)"
            className="w-1/2 border border-slate-300 rounded-md px-3 py-2"
          />
          <input
            type="number"
            value={surfaceMax}
            onChange={(e) => setSurfaceMax(e.target.value)}
            placeholder="Max Surface (sqm)"
            className="w-1/2 border border-slate-300 rounded-md px-3 py-2"
          />
        </div>
      </div>

      {/* Button */}
      <button
        onClick={handleSave}
        disabled={isSaving}
        className={`w-full px-4 py-2 rounded-md text-white font-medium ${
          isSaving
            ? "bg-blue-300 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700"
        }`}
      >
        {isSaving ? "Saving..." : "Save Filters"}
      </button>

      {/* Status */}
      {responseMsg && (
        <p className="mt-4 text-gray-700">
          {responseMsg}
        </p>
      )}
    </div>
  );
}