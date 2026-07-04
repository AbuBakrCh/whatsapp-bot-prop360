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
DOCUMENT_TYPE_COMMON_EXPENSES = "Common Expenses"

SUPPORTED_DOCUMENT_TYPES = {
    "electricity_bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity-bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "common_expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
    "common expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
    "common-expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
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

# Prop360 field IDs — common expenses cashflow entries
FIELD_TENANT_SHARE = "field-1779680012572-nc7n1lhm9"
FIELD_LANDLORD_SHARE = "field-1779680020866-s70nwdm41"
FIELD_MANAGEMENT_COMPANY = "field-1779680027992-o2rv2qvc4"
FIELD_EXPENSE_PERIOD = "field-1779680031528-6z5osildo"
FIELD_TOTAL_AMOUNT = "field-1757605508754-uea4iadqd"

# Fields the caller sets manually — never include in API response
EXCLUDED_RESPONSE_FIELDS = frozenset({
    "field-1757605637506-70de9chi5",  # property id
    "field-1761124961032-67lj21506",  # property owner
    "field-1757605870503-s1lu31him",  # transaction recorded by
    "attachedForms",
})

ELECTRICITY_BILL_EXTRACTION_SCHEMA = {
    "previous_balance_due": (
        "Previous unpaid / overdue balance in euros — amount owed from prior bills, "
        "not the current bill total. Look for equivalent labels such as "
        "Προηγούμενο Ανεξόφλητο Ποσό, Ληξιπρόθεσμο Ποσό, Previous Balance, "
        "Outstanding Balance, Prior Amount Due, Section B balance. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
    "matching_number": (
        "Document / invoice / bill reference number used for matching. "
        "Look for equivalent labels such as ΑΡ. ΕΓΓΡΑΦΟΥ, Document No, Invoice No, "
        "Bill Number, Reference No, Account Document. Leave empty if not found."
    ),
    "electricity_meter_no": (
        "Electricity supply point / meter / provision number. "
        "Look for equivalent labels such as ΑΡ. ΠΑΡΟΧΗΣ, Supply Number, Meter No, "
        "POD, Point of Delivery, Supply Point ID. Leave empty if not found."
    ),
    "electricity_account_no": (
        "Customer / account / contract code for the electricity supply. "
        "Look for equivalent labels such as ΚΩΔΙΚΟΣ ΠΕΛΑΤΗ, Customer Code, "
        "Account Number, Contract No, Client ID. Leave empty if not found."
    ),
    "who_gets_money": (
        "Electricity provider / retailer company that receives payment. "
        "Usually the company logo or name on the bill header (utility supplier). "
        "Return the company name as printed. Leave empty if unclear."
    ),
    "final_interim_bill": (
        "Bill settlement type. Return 'Final' for settlement/accrual/clearance bills "
        "(e.g. Εκκαθαριστικός, Final Bill, Settlement). "
        "Return 'Interim' for estimated/provisional bills (e.g. Ενδιάμεσος, Estimated). "
        "Leave empty if unclear."
    ),
    "payment_due_date": (
        "Payment due date. Return as YYYY-MM-DD regardless of how it appears on the document."
    ),
    "receipt_no": (
        "Receipt or document number if explicitly labeled as receipt. "
        "Leave empty if not distinct from the matching/document number."
    ),
    "service_period": (
        "Consumption or billing service period exactly as shown on the document "
        "(e.g. 10/11/2025 - 08/12/2025, June 2026, 01/01/2026-31/01/2026). "
        "Preserve original date format and language. Leave empty if not found."
    ),
    "rf_payment_code": (
        "Payment reference / RF / Rf code for bank payment. "
        "Look for equivalent labels such as Κωδικός Πληρωμής, Payment Code, RF Code, "
        "Payment Reference. Leave empty if not found."
    ),
}

COMMON_EXPENSES_EXTRACTION_SCHEMA = {
    "tenant_share": (
        "Tenant payable amount in euros. Look for labels such as "
        "ΠΟΣΟ ΕΝΟΙΚΙΑΣΤΗ, Tenant Amount, Tenant Share, Ενοικιαστής. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
    "landlord_share": (
        "Owner/landlord payable amount in euros. Look for labels such as "
        "ΠΟΣΟ ΙΔΙΟΚΤΗΤΗ, Owner Amount, Landlord Share, Ιδιοκτήτης. "
        "Numeric string with dot decimal. Use 0 only if explicitly shown as zero."
    ),
    "previous_balance_due": (
        "Previous outstanding/unpaid balance in euros. Look for labels such as "
        "ΠΡΟΗΓΟΥΜΕΝΟ ΥΠΟΛΟΙΠΟ, ΠΡΟΗΓ. ΟΦΕΙΛΕΣ, ΠΡ.ΑΝΕΙΣΠΡΑΚΤΕΣ, Outstanding Balance, "
        "Previous Balance, Ανείσπρακτες. Numeric string with dot decimal. Leave empty if not found."
    ),
    "management_company": (
        "Management company issuing the common expenses invoice "
        "(building management / διαχείριση κτιρίων). Return the company name as printed."
    ),
    "expense_period": (
        "Billing/expense period as shown on the invoice "
        "(e.g. ΙΟΥΝΙΟΣ 2026, June 2026, 06/2026). Preserve original language and format."
    ),
    "who_gets_money": (
        "Payee company receiving the payment — usually the management company on the invoice header/footer. "
        "Do NOT use individual bank account holder names (Δικαιούχος). "
        "Leave empty if unclear."
    ),
    "total_amount": (
        "Final payable amount in euros. Look for labels such as "
        "ΣΥΝΟΛΟ, ΠΛΗΡΩΤΕΟ, ΤΕΛΙΚΟ ΠΛΗΡΩΤΕΟ, TOTAL DUE, FINAL AMOUNT, "
        "ΣΥΝΟΛΟ ΥΠΟΛΟΙΠΟ, ΣΥΝΟΛ.ΥΠΟΛΟΙΠΟ, ΠΛΗΡΩΤΕΟ ΠΟΣΟ. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
}

_gen_model: genai.GenerativeModel | None = None


def normalize_document_type(document_type: str) -> str | None:
    key = re.sub(r"[-_]+", " ", document_type.strip().lower())
    key = re.sub(r"\s+", " ", key)
    if key in SUPPORTED_DOCUMENT_TYPES:
        return SUPPORTED_DOCUMENT_TYPES[key]
    if document_type.strip() == DOCUMENT_TYPE_ELECTRICITY_BILL:
        return DOCUMENT_TYPE_ELECTRICITY_BILL
    if document_type.strip() == DOCUMENT_TYPE_COMMON_EXPENSES:
        return DOCUMENT_TYPE_COMMON_EXPENSES
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
    return value.strip()


def _normalize_service_period(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if not text.endswith(" "):
        text = f"{text} "
    return text


def _parse_amount_optional(value: Any) -> str | None:
    parsed = _parse_amount(value)
    return parsed if parsed else None


def _amount_is_positive(value: str | None) -> bool:
    if not value:
        return False
    try:
        return float(value) > 0
    except ValueError:
        return False


def _omit_unextracted_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Drop empty values; keep numeric zero when explicitly present."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key in EXCLUDED_RESPONSE_FIELDS:
            continue
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        result[key] = value
    return result


def _extract_with_gemini(
    file_bytes: bytes,
    *,
    filename: str | None,
    schema: dict[str, str],
    document_description: str,
    extra_rules: str = "",
) -> dict[str, Any]:
    file_type = detect_file_type(file_bytes, filename)
    if not file_type:
        raise ValueError("Cannot determine document type. Upload a PDF, JPEG, or PNG.")

    encoded = {
        "mime_type": _build_mime(file_type),
        "data": file_bytes,
    }

    schema_lines = "\n".join(f'  "{key}": "{desc}"' for key, desc in schema.items())
    prompt = f"""
You are a data extraction engine for {document_description}.

Read the attached document and return ONE JSON object with exactly these keys:
{{
{schema_lines}
}}

Rules:
- Return JSON only. No markdown, no commentary.
- Do NOT guess or infer values. Use empty string for any field you cannot confidently read.
- Numeric amounts must use dot as decimal separator, no currency symbols.
- The document layout may vary by issuer — use semantic understanding, not fixed positions.
- Support Greek and English labels and equivalent terminology.
- Handle OCR variations and minor recognition errors.
{extra_rules}
"""

    response = _get_model().generate_content([prompt, encoded])
    raw = _strip_json_fences(response.text or "")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse extraction response as JSON: {raw[:500]}") from exc


def _extract_electricity_bill_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    return _extract_with_gemini(
        file_bytes,
        filename=filename,
        schema=ELECTRICITY_BILL_EXTRACTION_SCHEMA,
        document_description=(
            "electricity bills and invoices from any utility provider, "
            "in any layout, language (Greek or English), or format"
        ),
        extra_rules="""
- Templates vary widely across providers — extract by meaning, not by layout or coordinates.
- Do NOT assume a specific company, logo, or invoice template.
- Recognize equivalent terminology rather than exact labels.
- payment_due_date must be YYYY-MM-DD when confidently extracted.
- For previous_balance_due, use 0 only when the document explicitly shows zero prior balance.
""",
    )


def _extract_common_expenses_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    return _extract_with_gemini(
        file_bytes,
        filename=filename,
        schema=COMMON_EXPENSES_EXTRACTION_SCHEMA,
        document_description=(
            "Greek common expenses (κοινόχρηστα) building maintenance invoices "
            "from property management companies"
        ),
        extra_rules="""
- Templates vary widely — extract by meaning, not by layout.
- For landlord_share, return "0" when the owner/landlord amount is explicitly zero or absent while tenant share is present.
- who_gets_money should be the management company, not an individual bank beneficiary (Δικαιούχος).
""",
    )


def build_electricity_bill_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    provider = _normalize_provider(extracted.get("who_gets_money", ""))
    matching_number = (extracted.get("matching_number") or "").strip()
    receipt_no = (extracted.get("receipt_no") or "").strip()
    previous_balance = _parse_amount_optional(extracted.get("previous_balance_due"))
    service_period = _normalize_service_period(extracted.get("service_period", ""))
    due_date = _to_iso_due_date(extracted.get("payment_due_date", ""))

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_ELECTRICITY_BILL,
        FIELD_CATEGORY: "Accrual",
    }

    if previous_balance is not None:
        data[FIELD_PREVIOUS_BALANCE_DUE] = previous_balance
        data[FIELD_OVERDUE_PAYMENT] = "Yes" if _amount_is_positive(previous_balance) else "No"
    if matching_number:
        data[FIELD_MATCHING_NUMBER] = matching_number
    meter_no = (extracted.get("electricity_meter_no") or "").strip()
    if meter_no:
        data[FIELD_ELECTRICITY_METER_NO] = meter_no
    account_no = (extracted.get("electricity_account_no") or "").strip()
    if account_no:
        data[FIELD_ELECTRICITY_ACCOUNT_NO] = account_no
    if provider:
        data[FIELD_WHO_GETS_MONEY] = provider
        data[FIELD_WHO_GETS_MONEY_ALT] = provider
    bill_type = (extracted.get("final_interim_bill") or "").strip()
    if bill_type:
        data[FIELD_BILL_TYPE] = bill_type
    if due_date:
        data[FIELD_PAYMENT_DUE_DATE] = due_date
    if receipt_no:
        data[FIELD_RECEIPT_NO] = receipt_no
    elif matching_number:
        data[FIELD_RECEIPT_NO] = matching_number
    if service_period:
        data[FIELD_SERVICE_PERIOD] = service_period
    rf_code = (extracted.get("rf_payment_code") or "").strip()
    if rf_code:
        data[FIELD_RF_PAYMENT_CODE] = rf_code

    return _omit_unextracted_fields(data)


def build_common_expenses_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    management_company = (extracted.get("management_company") or "").strip()
    who_gets_money = (extracted.get("who_gets_money") or management_company).strip()
    if management_company:
        who_gets_money = management_company
    expense_period = (extracted.get("expense_period") or "").strip()
    previous_balance = _parse_amount_optional(extracted.get("previous_balance_due"))
    tenant_share = _parse_amount_optional(extracted.get("tenant_share"))
    landlord_share = _parse_amount_optional(extracted.get("landlord_share"))
    total_amount = _parse_amount_optional(extracted.get("total_amount"))

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_COMMON_EXPENSES,
        FIELD_CATEGORY: "Accrual",
    }

    if tenant_share is not None:
        data[FIELD_TENANT_SHARE] = tenant_share
    if landlord_share is not None:
        data[FIELD_LANDLORD_SHARE] = landlord_share
    if previous_balance is not None:
        data[FIELD_PREVIOUS_BALANCE_DUE] = previous_balance
        data[FIELD_OVERDUE_PAYMENT] = "Yes" if _amount_is_positive(previous_balance) else "No"
    if management_company:
        data[FIELD_MANAGEMENT_COMPANY] = management_company
    if expense_period:
        data[FIELD_EXPENSE_PERIOD] = expense_period
        data[FIELD_SERVICE_PERIOD] = expense_period
    if who_gets_money:
        data[FIELD_WHO_GETS_MONEY_ALT] = who_gets_money
    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount

    return _omit_unextracted_fields(data)


def extract_cashflow_data_from_document(
    file_bytes: bytes,
    *,
    document_type: str,
    filename: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_document_type(document_type)
    if normalized is None:
        supported = ", ".join(sorted({k.replace(" ", "_") for k in SUPPORTED_DOCUMENT_TYPES}))
        raise ValueError(
            f"Document type '{document_type}' is not supported. "
            f"Supported types: {supported}"
        )

    if normalized == DOCUMENT_TYPE_ELECTRICITY_BILL:
        extracted = _extract_electricity_bill_with_gemini(file_bytes, filename)
        return build_electricity_bill_cashflow_data(extracted)

    if normalized == DOCUMENT_TYPE_COMMON_EXPENSES:
        extracted = _extract_common_expenses_with_gemini(file_bytes, filename)
        return build_common_expenses_cashflow_data(extracted)

    raise ValueError(f"Document type '{document_type}' is not supported.")
