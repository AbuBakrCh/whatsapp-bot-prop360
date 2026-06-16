import React from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import App from "./App";
import WhatsAppMessenger from "./pages/WhatsAppMessenger";
import Utilities from "./pages/Utilities";
import ImportantLinks from './pages/ImportantLinks'
import Spitogatos from "./pages/Spitogatos";
import Ledger from "./pages/Ledger";
import Groups from "./pages/Groups";
import "./styles.css";

createRoot(document.getElementById("root")).render(
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/messenger" element={<WhatsAppMessenger />} />
      <Route path="/utilities" element={<Utilities />} />
      <Route path="/important-links" element={<ImportantLinks />} />
      <Route path="/spitogatos" element={<Spitogatos />} />
      <Route path="/ledger" element={<Ledger />} />
      <Route path="/groups" element={<Groups />} />
    </Routes>
  </BrowserRouter>
);
