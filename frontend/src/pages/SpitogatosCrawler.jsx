import React, { useState } from "react";
import { startSpitogatosCrawler, stopSpitogatosCrawler } from "../api";

export default function SpitogatosCrawler() {
  const [url, setUrl] = useState("");
  const [pages, setPages] = useState(3);
  const [cookie, setCookie] = useState("");
  const [propToken, setPropToken] = useState("");

  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  const [processedCount, setProcessedCount] = useState(null);
  const [responseMsg, setResponseMsg] = useState("");

  const handleStart = async () => {
    if (!url || !cookie || !propToken) {
      setResponseMsg("URL, cookie and prop token are required.");
      return;
    }

    setIsStarting(true);
    setResponseMsg("Crawler running...");
    setProcessedCount(null);

    try {
      const res = await startSpitogatosCrawler({
        url,
        pages: Number(pages),
        cookie,
        "prop-token": propToken,
      });

      if (res.error) {
        setResponseMsg(res.error);
      } else {
        setProcessedCount(res.total);
        setResponseMsg("Crawler completed successfully.");
      }
    } catch (err) {
      setResponseMsg("Failed to start crawler.");
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    setIsStopping(true);

    try {
      const res = await stopSpitogatosCrawler();

      if (res.error) {
        setResponseMsg(res.error);
      } else {
        setResponseMsg("Stop signal sent. Crawler will stop shortly.");
      }
    } catch (err) {
      setResponseMsg("Failed to stop crawler.");
    } finally {
      setIsStopping(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-10 bg-white rounded-2xl shadow-lg p-6 border border-slate-200">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">
        Spitogatos Crawler
      </h2>

      {/* Inputs */}
      <div className="flex flex-col gap-4 mb-6">
        <input
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Base URL"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <input
          type="number"
          value={pages}
          onChange={(e) => setPages(e.target.value)}
          placeholder="Pages"
          className="border border-slate-300 rounded-md px-3 py-2"
        />

        <textarea
          value={cookie}
          onChange={(e) => setCookie(e.target.value)}
          placeholder="Spitogatos Cookie (reese84)"
          className="border border-slate-300 rounded-md px-3 py-2 h-20 resize-none"
        />

        <textarea
          value={propToken}
          onChange={(e) => setPropToken(e.target.value)}
          placeholder="Solomon Bearer Token"
          className="border border-slate-300 rounded-md px-3 py-2 h-20 resize-none"
        />
      </div>

      {/* Buttons */}
      <div className="flex gap-4">
        <button
          onClick={handleStart}
          disabled={isStarting}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium ${
            isStarting
              ? "bg-green-300 cursor-not-allowed"
              : "bg-green-600 hover:bg-green-700"
          }`}
        >
          {isStarting ? "Running..." : "Start Crawl"}
        </button>

        <button
          onClick={handleStop}
          disabled={isStopping}
          className={`w-1/2 px-4 py-2 rounded-md text-white font-medium ${
            isStopping
              ? "bg-red-300 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-700"
          }`}
        >
          {isStopping ? "Stopping..." : "Stop Crawl"}
        </button>
      </div>

      {/* Status */}
      {responseMsg && (
        <p className="mt-4 text-gray-700">
          {responseMsg}
        </p>
      )}

      {/* Result */}
      {processedCount !== null && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-green-700 font-semibold">
            ✅ Total Properties Processed: {processedCount}
          </p>
        </div>
      )}
    </div>
  );
}