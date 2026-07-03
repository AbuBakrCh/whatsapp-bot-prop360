"""
Extract Prop360 cashflow form `data` fields from utility bill documents (PDF/image).
Uses Gemini vision to read the document and maps results to Prop360 field IDs.
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import google.generativeai as genai
from PIL import Image

DOCUMENT_TYPE_ELECTRICITY_BILL = "Electricity Bill"

SUPPORTED_DOCUMENT_TYPES = {
    "electricity_bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity-bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
}

# Prop360 field IDs — electricity bill cashflow entries
FIELD_DOCUMENT_TYPE = "field-1758699529035-xp7lumgx5"
FIELD_CATEGORY = "field-1780751488281-e84mgqaeo"
FIELD_DEBIT_CREDIT = "field-1757605718340-ue95ozr9u"
FIELD_PREVIOUS_BALANCE_DUE = "field-1779680024322-4z5tp495g"
FIELD_MATCHING_NUMBER = "field-1779680035201-hbd3ywbb1"
FIELD_OVERDUE_PAYMENT = "field-1779680159334-im1zzivsm"
FIELD_ELECTRICITY_METER_NO = "field-1779820117998-lcesnc9f"
FIELD_ELECTRICITY_ACCOUNT_NO = "field-1779820121635-ml77dats8"
FIELD_WHO_GETS_MONEY = "field-1779820125069-shhgq8s7h"
FIELD_BILL_TYPE = "field-1779820128415-xtjhuf301"
FIELD_PAYMENT_DUE_DATE = "field-1779823931287-9y31g3sbo"
FIELD_RECEIPT_NO = "field-1779826634068-ud90mh591"
FIELD_SERVICE_PERIOD = "field-1780046954441-ee7gfecma"
FIELD_RF_PAYMENT_CODE = "field-1781250415587-3dmfxioiz"
FIELD_WHO_GETS_MONEY_ALT = "field-1781346694051-dsbih9jen"

# Fields the caller sets manually — never include in API response
EXCLUDED_RESPONSE_FIELDS = frozenset({
    "field-1757605637506-70de9chi5",  # property id
    "field-1761124961032-67lj21506",  # property owner
    "field-1757605870503-s1lu31him",  # transaction recorded by
    "attachedForms",
})

ELECTRICITY_BILL_EXTRACTION_SCHEMA = {
    "previous_balance_due": (
        "Previous unpaid / overdue balance (section B / Προηγούμενο Ανεξόφλητο Ποσό / Ληξιπρόθεσμο). "
        "Numeric string with dot decimal. Use 0 if none."
    ),
    "matching_number": "Matching / document number (ΑΡ. ΕΓΓΡΑΦΟΥ) if present.",
    "overdue_payment": (
        "Whether there is an overdue payment notice on the bill. "
        "Answer exactly 'Yes' or 'No'."
    ),
    "electricity_meter_no": "Electricity supply / meter number (ΑΡ. ΠΑΡΟΧΗΣ) if present.",
    "electricity_account_no": "Electricity customer / account code (ΚΩΔΙΚΟΣ ΠΕΛΑΤΗ) if present.",
    "who_gets_money": "Company that receives payment (utility provider name, e.g. ΖΕΝΙΘ, ΔΕΗ).",
    "final_interim_bill": (
        "Bill type. Use 'Final' for Greek Εκκαθαριστικός, 'Interim' for estimated bills, "
        "or the closest English equivalent on the document."
    ),
    "payment_due_date": "Payment due date as YYYY-MM-DD.",
    "receipt_no": "Receipt number if present and distinct from matching number; otherwise same as matching number.",
    "service_period_start": "Service / consumption period start as YYYY-MM-DD.",
    "service_period_end": "Service / consumption period end as YYYY-MM-DD.",
    "rf_payment_code": "RF payment code (Κωδικός Πληρωμής) if present.",
}

_gen_model: genai.GenerativeModel | None = None


def normalize_document_type(document_type: str) -> str | None:
    key = re.sub(r"[-_]+", " ", document_type.strip().lower())
    key = re.sub(r"\s+", " ", key)
    if key in SUPPORTED_DOCUMENT_TYPES:
        return SUPPORTED_DOCUMENT_TYPES[key]
    if document_type.strip() == DOCUMENT_TYPE_ELECTRICITY_BILL:
        return DOCUMENT_TYPE_ELECTRICITY_BILL
    return None


def _get_model() -> genai.GenerativeModel:
    global _gen_model
    if _gen_model is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        genai.configure(api_key=api_key)
        _gen_model = genai.GenerativeModel("gemini-2.5-flash")
    return _gen_model


def detect_file_type(file_bytes: bytes, filename: str | None = None) -> str | None:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return img.format.lower()
    except Exception:
        pass

    if file_bytes[:4] == b"%PDF" or (filename and filename.lower().endswith(".pdf")):
        return "pdf"

    return None


def _build_mime(file_type: str) -> str:
    if file_type in ("jpeg", "jpg", "png"):
        return f"image/{file_type if file_type != 'jpg' else 'jpeg'}"
    if file_type == "pdf":
        return "application/pdf"
    raise ValueError(f"Unsupported file type: {file_type}")


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_amount(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    text = text.replace("€", "").replace(" ", "").replace(",", ".")
    try:
        return f"{float(text):.2f}".rstrip("0").rstrip(".")
    except ValueError:
        return text


def _format_service_period(start: str, end: str) -> str:
    start_fmt = _format_display_date(start)
    end_fmt = _format_display_date(end)
    if start_fmt and end_fmt:
        return f"{start_fmt} - {end_fmt} "
    return ""


def _format_display_date(value: str) -> str:
    if not value:
        return ""
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return value.strip()


def _to_iso_due_date(value: str) -> str:
    if not value:
        return ""
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(value.strip(), fmt)
            break
        except ValueError:
            continue
    if not parsed:
        return value.strip()

    utc_midnight = datetime(
        parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc
    ) - timedelta(hours=2)
    return utc_midnight.strftime("%Y-%m-%dT22:00:00.000Z")


def _normalize_provider(value: str) -> str:
    if not value:
        return ""
    upper = value.strip().upper()
    aliases = {
        "ZENITH": "ΖΕΝΙΘ",
        "ZENIΘ": "ΖΕΝΙΘ",
    }
    return aliases.get(upper, upper)


def _normalize_yes_no(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in ("yes", "y", "true", "1"):
        return "Yes"
    if text in ("no", "n", "false", "0"):
        return "No"
    return str(value or "").strip()


def _extract_electricity_bill_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    file_type = detect_file_type(file_bytes, filename)
    if not file_type:
        raise ValueError("Cannot determine document type. Upload a PDF, JPEG, or PNG.")

    encoded = {
        "mime_type": _build_mime(file_type),
        "data": file_bytes,
    }

    schema_lines = "\n".join(
        f'  "{key}": "{desc}"' for key, desc in ELECTRICITY_BILL_EXTRACTION_SCHEMA.items()
    )
    prompt = f"""
You are a data extraction engine for Greek electricity bills.

Read the attached document and return ONE JSON object with exactly these keys:
{{
{schema_lines}
}}

Rules:
- Return JSON only. No markdown, no commentary.
- Use empty string for missing text fields and "0" for missing numeric amounts.
- Dates must use YYYY-MM-DD.
- Numeric amounts must use dot as decimal separator, no currency symbols.
- overdue_payment must be exactly "Yes" or "No".
- If a disconnection / overdue notice is shown, overdue_payment is "Yes".
"""

    response = _get_model().generate_content([prompt, encoded])
    raw = _strip_json_fences(response.text or "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse extraction response as JSON: {raw[:500]}") from exc


def build_electricity_bill_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    provider = _normalize_provider(extracted.get("who_gets_money", ""))
    matching_number = (extracted.get("matching_number") or "").strip()
    receipt_no = (extracted.get("receipt_no") or matching_number).strip()
    previous_balance = _parse_amount(extracted.get("previous_balance_due"))
    overdue_payment = _normalize_yes_no(extracted.get("overdue_payment"))
    if not overdue_payment and previous_balance not in ("", "0", "0.0", "0.00"):
        overdue_payment = "Yes"
    elif not overdue_payment:
        overdue_payment = "No"

    data = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_ELECTRICITY_BILL,
        FIELD_CATEGORY: "Accrual",
        FIELD_DEBIT_CREDIT: "Debit",
        FIELD_PREVIOUS_BALANCE_DUE: previous_balance,
        FIELD_MATCHING_NUMBER: matching_number,
        FIELD_OVERDUE_PAYMENT: overdue_payment,
        FIELD_ELECTRICITY_METER_NO: (extracted.get("electricity_meter_no") or "").strip(),
        FIELD_ELECTRICITY_ACCOUNT_NO: (extracted.get("electricity_account_no") or "").strip(),
        FIELD_WHO_GETS_MONEY: provider,
        FIELD_BILL_TYPE: (extracted.get("final_interim_bill") or "").strip(),
        FIELD_PAYMENT_DUE_DATE: _to_iso_due_date(extracted.get("payment_due_date", "")),
        FIELD_RECEIPT_NO: receipt_no,
        FIELD_SERVICE_PERIOD: _format_service_period(
            extracted.get("service_period_start", ""),
            extracted.get("service_period_end", ""),
        ),
        FIELD_RF_PAYMENT_CODE: (extracted.get("rf_payment_code") or "").strip(),
        FIELD_WHO_GETS_MONEY_ALT: provider,
    }
    return {k: v for k, v in data.items() if k not in EXCLUDED_RESPONSE_FIELDS}


def extract_cashflow_data_from_document(
    file_bytes: bytes,
    *,
    document_type: str,
    filename: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_document_type(document_type)
    if normalized is None:
        raise ValueError(
            f"Document type '{document_type}' is not supported. "
            f"Supported types: {', '.join(sorted(set(SUPPORTED_DOCUMENT_TYPES.values())))}"
        )

    if normalized == DOCUMENT_TYPE_ELECTRICITY_BILL:
        extracted = _extract_electricity_bill_with_gemini(file_bytes, filename)
        return build_electricity_bill_cashflow_data(extracted)

    raise ValueError(f"Document type '{document_type}' is not supported.")
