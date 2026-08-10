import { jsPDF } from "jspdf";

const ACCENT = [91, 115, 149]; // #5B7395
const ACCENT_SOFT = [232, 238, 244]; // light slate wash
const TEXT = [33, 37, 41];
const MUTED = [120, 130, 140];
const LABEL = [70, 80, 95];
const BORDER = [210, 216, 222];
const TABLE_HEADER_BG = [246, 248, 250];
const ROW_ALT = [251, 252, 253];

export const ISSUER = {
  name: "Muhammd Abu Bakr",
  title: "Software Consultant",
  location: "Lahore, Pakistan",
  phone: "(92) 324-4181389",
  email: "m.abubakr916@gmail.com",
};

export const CLIENT = {
  name: "Kostas Arslanoglu",
  company: "InvestGreece",
  address: "Eleftheriou Venizelou 212, Kallithea, Athens, Greece",
  phone: "+30 698 382 8390",
  email: "ka@investgreece.gr",
};

export const SERVICE = {
  item: "001",
  description: "IT services and consultation",
};

export const PAYMENT = {
  method: "Wise",
  wisetag: "@muhammada21124",
  link: "https://wise.com/pay/me/muhammada21124",
  instruction: "Please make payments to m.abubakr916@gmail.com via Wise.",
};

const NOTES = [
  "Payment is appreciated promptly upon receipt of this invoice.",
  "Late payments may incur additional charges.",
  "If you have any questions or concerns regarding this invoice, please contact us at (92) 324 4181389 or m.abubakr916@gmail.com.",
];

function ordinal(n) {
  const v = n % 100;
  if (v >= 11 && v <= 13) return `${n}th`;
  switch (n % 10) {
    case 1:
      return `${n}st`;
    case 2:
      return `${n}nd`;
    case 3:
      return `${n}rd`;
    default:
      return `${n}th`;
  }
}

const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
];

/** Format a Date as "2nd December, 2025" */
export function formatOrdinalDate(date) {
  const d = date instanceof Date ? date : new Date(date);
  return `${ordinal(d.getDate())} ${MONTHS[d.getMonth()]}, ${d.getFullYear()}`;
}

/** Parse YYYY-MM-DD as local date (avoids UTC shift). */
export function parseLocalDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

export function formatAmount(amount) {
  const n = Number(amount);
  return `$${n.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

/** Invoice number from year + month, e.g. August 2026 → #INV2026-008 */
export function buildInvoiceNumber(date) {
  const d = date instanceof Date ? date : parseLocalDate(date);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(3, "0");
  return `#INV${year}-${month}`;
}

function drawLabelValue(doc, label, value, x, y, opts = {}) {
  const { maxValueWidth } = opts;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(9.5);
  doc.setTextColor(...LABEL);
  doc.text(label, x, y);
  const lw = doc.getTextWidth(label);
  doc.setFont("helvetica", "normal");
  doc.setTextColor(...TEXT);
  doc.setFontSize(10);
  const valueX = x + lw + 2.5;
  if (maxValueWidth) {
    const lines = doc.splitTextToSize(value, maxValueWidth - lw - 2.5);
    doc.text(lines, valueX, y);
    return lines.length;
  }
  doc.text(value, valueX, y);
  return 1;
}

function drawSectionBar(doc, text, x, y, width) {
  const h = 7.5;
  doc.setFillColor(...ACCENT);
  doc.roundedRect(x, y, width, h, 2.2, 2.2, "F");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(10);
  doc.setTextColor(255, 255, 255);
  doc.text(text, x + 6, y + 5.1);
  doc.setTextColor(...TEXT);
  return y + h + 7;
}

/**
 * @param {{ invoiceNumber: string, invoiceDate: Date, amount: number }} opts
 * @returns {jsPDF}
 */
export function buildInvoicePdf({ invoiceNumber, invoiceDate, amount }) {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const pageW = doc.internal.pageSize.getWidth();
  const pageH = doc.internal.pageSize.getHeight();
  const margin = 18;
  const contentW = pageW - margin * 2;
  const rightX = pageW - margin;
  let y = 20;

  const amountStr = formatAmount(amount);
  const invoiceDateStr = formatOrdinalDate(invoiceDate);

  // Subtle top accent strip
  doc.setFillColor(...ACCENT);
  doc.rect(0, 0, pageW, 3.2, "F");

  // —— Header ——
  doc.setFont("times", "bold");
  doc.setFontSize(30);
  doc.setTextColor(...TEXT);
  doc.text("Invoice", margin, y + 4);

  doc.setFont("times", "bold");
  doc.setFontSize(17);
  doc.text(ISSUER.name, rightX, y + 2, { align: "right" });
  y += 9;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(...MUTED);
  doc.text(ISSUER.title, rightX, y, { align: "right" });

  y += 7;
  doc.setDrawColor(...ACCENT);
  doc.setLineWidth(0.55);
  doc.line(margin, y, rightX, y);

  // —— Meta + contact ——
  y += 11;
  const metaY = y;

  drawLabelValue(doc, "Invoice Number:", invoiceNumber, margin, metaY);
  drawLabelValue(doc, "Invoice Date:", invoiceDateStr, margin, metaY + 6.5);

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9.5);
  doc.setTextColor(...MUTED);
  doc.text(ISSUER.location, rightX, metaY, { align: "right" });
  doc.text(ISSUER.phone, rightX, metaY + 5.5, { align: "right" });
  doc.text(ISSUER.email, rightX, metaY + 11, { align: "right" });

  y = metaY + 22;

  // —— Bill To ——
  y = drawSectionBar(doc, "Bill To", margin, y, contentW * 0.28);

  // Soft card behind bill-to
  const billCardH = 28;
  doc.setFillColor(...ACCENT_SOFT);
  doc.roundedRect(margin, y - 2, contentW, billCardH, 2, 2, "F");

  const midX = margin + contentW * 0.56;
  const billPad = 4;
  drawLabelValue(doc, "Client Name:", CLIENT.name, margin + billPad, y + 5);
  drawLabelValue(
    doc,
    "Client Company:",
    CLIENT.company,
    margin + billPad,
    y + 12
  );
  drawLabelValue(
    doc,
    "Client Address:",
    CLIENT.address,
    margin + billPad,
    y + 19,
    { maxValueWidth: midX - margin - 10 }
  );

  drawLabelValue(doc, "Client Phone:", CLIENT.phone, midX, y + 5);
  drawLabelValue(doc, "Client Email:", CLIENT.email, midX, y + 12);

  y += billCardH + 10;

  // —— Description of Service ——
  y = drawSectionBar(doc, "Description of Service", margin, y, contentW * 0.48);

  const colItemW = 22;
  const colTotalW = 38;
  const colDescX = margin + colItemW;
  const colTotalX = rightX - colTotalW;
  const rowH = 9;
  const tableRows = 3;
  const tableTop = y;
  const headerH = 8.5;
  const tableH = headerH + tableRows * rowH;

  // Table shadow-ish border
  doc.setDrawColor(...BORDER);
  doc.setLineWidth(0.35);
  doc.setFillColor(...TABLE_HEADER_BG);
  doc.roundedRect(margin, tableTop, contentW, tableH, 1.5, 1.5, "FD");

  // Clip-ish: redraw header fill inside
  doc.setFillColor(...TABLE_HEADER_BG);
  doc.rect(margin + 0.2, tableTop + 0.2, contentW - 0.4, headerH, "F");

  // Alternating body rows
  for (let i = 0; i < tableRows; i++) {
    if (i % 2 === 1) {
      doc.setFillColor(...ROW_ALT);
      doc.rect(
        margin + 0.2,
        tableTop + headerH + i * rowH,
        contentW - 0.4,
        rowH,
        "F"
      );
    }
  }

  // Column dividers
  doc.setDrawColor(...BORDER);
  doc.line(colDescX, tableTop, colDescX, tableTop + tableH);
  doc.line(colTotalX, tableTop, colTotalX, tableTop + tableH);
  doc.line(margin, tableTop + headerH, rightX, tableTop + headerH);
  for (let i = 1; i < tableRows; i++) {
    doc.line(
      margin,
      tableTop + headerH + i * rowH,
      rightX,
      tableTop + headerH + i * rowH
    );
  }

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(...MUTED);
  doc.text("Item", margin + 4, tableTop + 5.6);
  doc.text("Description", colDescX + 4, tableTop + 5.6);
  doc.text("Total", rightX - 4, tableTop + 5.6, { align: "right" });

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(...TEXT);
  const dataY = tableTop + headerH + 6;
  doc.text(SERVICE.item, margin + 4, dataY);
  doc.text(SERVICE.description, colDescX + 4, dataY);
  doc.setFont("helvetica", "bold");
  doc.text(amountStr, rightX - 4, dataY, { align: "right" });

  y = tableTop + tableH + 9;

  // TOTAL block
  const totalBoxW = 58;
  const totalBoxX = rightX - totalBoxW;
  doc.setDrawColor(...ACCENT);
  doc.setLineWidth(1);
  doc.line(totalBoxX, y, rightX, y);
  y += 7;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.setTextColor(...ACCENT);
  doc.text("TOTAL", totalBoxX, y);
  doc.setTextColor(...TEXT);
  doc.setFontSize(13);
  doc.text(amountStr, rightX, y, { align: "right" });

  y += 14;

  // —— Payment Information ——
  y = drawSectionBar(doc, "Payment Information", margin, y, contentW * 0.42);

  const payCardH = 30;
  doc.setFillColor(...ACCENT_SOFT);
  doc.roundedRect(margin, y - 2, contentW, payCardH, 2, 2, "F");

  drawLabelValue(
    doc,
    "Recipient Payment Method:",
    PAYMENT.method,
    margin + 4,
    y + 5
  );
  drawLabelValue(doc, "Wisetag:", PAYMENT.wisetag, margin + 4, y + 12);
  drawLabelValue(doc, "Link:", PAYMENT.link, margin + 4, y + 19);

  y += payCardH + 2;
  doc.setFont("helvetica", "italic");
  doc.setFontSize(8.5);
  doc.setTextColor(...MUTED);
  doc.text(PAYMENT.instruction, margin + 1, y);

  y += 12;

  // —— Notes ——
  y = drawSectionBar(doc, "Notes", margin, y, contentW * 0.2);

  doc.setFont("helvetica", "italic");
  doc.setFontSize(9);
  doc.setTextColor(...LABEL);
  NOTES.forEach((note) => {
    const lines = doc.splitTextToSize(`•  ${note}`, contentW - 2);
    doc.text(lines, margin, y);
    y += lines.length * 4.4 + 2.5;
  });

  // Footer
  const footerY = Math.max(y + 14, pageH - 22);
  doc.setDrawColor(...BORDER);
  doc.setLineWidth(0.3);
  doc.line(margin, footerY - 6, rightX, footerY - 6);
  doc.setFont("helvetica", "bolditalic");
  doc.setFontSize(11);
  doc.setTextColor(...ACCENT);
  doc.text("Thank you for choosing Us!", margin, footerY);

  return doc;
}

/**
 * Build and download the invoice PDF.
 * @param {{ invoiceDate: Date|string, amount: number }} opts
 * @returns {{ invoiceNumber: string, filename: string }}
 */
export function downloadInvoicePdf({ invoiceDate, amount }) {
  const date =
    invoiceDate instanceof Date ? invoiceDate : parseLocalDate(invoiceDate);
  const invoiceNumber = buildInvoiceNumber(date);

  const doc = buildInvoicePdf({
    invoiceNumber,
    invoiceDate: date,
    amount: Number(amount),
  });

  const filename = `${invoiceNumber.replace("#", "")}.pdf`;
  doc.save(filename);

  return { invoiceNumber, filename };
}
