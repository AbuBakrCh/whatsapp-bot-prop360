import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import Utilities from "./pages/Utilities";
import ImportantLinks from './pages/ImportantLinks'
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/utilities" element={<Utilities />} />
      <Route path="/important-links" element={<ImportantLinks />} />
    </Routes>
  </BrowserRouter>
);
