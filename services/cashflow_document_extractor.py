"""
Extract Prop360 cashflow form `data` fields from utility bill documents (PDF/image).
Uses Gemini vision to read the document and maps results to Prop360 field IDs.
"""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import google.generativeai as genai
from PIL import Image

DOCUMENT_TYPE_ELECTRICITY_BILL = "Electricity Bill"
DOCUMENT_TYPE_COMMON_EXPENSES = "Common Expenses"
DOCUMENT_TYPE_WATER_BILL = "Water Bill"
DOCUMENT_TYPE_BANK_RECEIPT = "Bank Transaction"
DOCUMENT_TYPE_INVOICE = "Invoice"

SUPPORTED_DOCUMENT_TYPES = {
    "electricity_bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "electricity-bill": DOCUMENT_TYPE_ELECTRICITY_BILL,
    "common_expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
    "common expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
    "common-expenses": DOCUMENT_TYPE_COMMON_EXPENSES,
    "water_bill": DOCUMENT_TYPE_WATER_BILL,
    "water bill": DOCUMENT_TYPE_WATER_BILL,
    "water-bill": DOCUMENT_TYPE_WATER_BILL,
    "bank_receipt": DOCUMENT_TYPE_BANK_RECEIPT,
    "bank receipt": DOCUMENT_TYPE_BANK_RECEIPT,
    "bank-receipt": DOCUMENT_TYPE_BANK_RECEIPT,
    "bank_transaction": DOCUMENT_TYPE_BANK_RECEIPT,
    "bank transaction": DOCUMENT_TYPE_BANK_RECEIPT,
    "invoice": DOCUMENT_TYPE_INVOICE,
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

# Prop360 field IDs — water bill cashflow entries
FIELD_WATER_METER_NO = "field-1779825332418-7lwuwsbd7"
FIELD_WATER_ACCOUNT_NO = "field-1779825343116-00klf8dff"

# Prop360 field IDs — bank receipt cashflow entries
FIELD_TRX_REF_NO = "field-1757604618017-q2mtvmiqp"
FIELD_TRX_DATE = "field-1757605069078-p5plna7qr"
FIELD_TRX_VALUE_DATE = "field-1757605079803-icw8ykc19"
FIELD_TRX_BANK = "field-1757605194423-ofbqnqfso"
FIELD_INVEST_GREECE_NOTES = "field-1757605632930-2hfg96qgr"
FIELD_MONTH = "field-1759392145178-qbx06ungl"
FIELD_YEAR = "field-1759392151218-bmx5iidn6"
FIELD_PAYMENT_DIRECTION = "field-1783088283028-kwkardhwn"

# Prop360 field IDs — invoice cashflow entries
FIELD_TRX_PAYER = "field-1758478123620-9ztb0q9sq"
FIELD_COMPANY_ADDRESS = "field-1780046462769-14ll0br14"
FIELD_TAX_OFFICE = "field-1780046437513-bwmq7gzvl"
FIELD_CUST_INFO_BILL_TO = "field-1780046466675-0bbram33w"
FIELD_SERVICE_DESCRIPTION = "field-1780046470280-nqlyhq3qo"
FIELD_BANK_IBAN_INFO = "field-1780047071203-1ww7zfwdi"
FIELD_INVOICE_ISSUER = "field-1763025141515-wp3fyie36"
FIELD_INVOICE_ISSUE_DATE = "field-1763025943928-96pgpyzij"
FIELD_INVOICE_RECIPIENT = "field-1763025146443-1t1qiizo0"
FIELD_INVOICE_SOURCE = "field-1783317314494-7wrh0uzes"
FIELD_INVOICE_ISSUER_TAX_ID = "field-1763025149106-5rohv9ksd"
FIELD_INVOICE_RECIPIENT_TAX_ID = "field-1763025938971-3vmgk1kn5"
FIELD_INVOICE_ISSUED_BY = "field-1780140384161-8yovnepai"
FIELD_INVOICE_NUMBER = "field-1763025941333-lfsl04kdf"
FIELD_AMOUNT_EXCLUDING_VAT = "field-1763025946163-pwe00nucs"
FIELD_VAT_RATIO = "field-1763026912693-45fgrq8y4"
FIELD_VAT_AMOUNT = "field-1763026916002-uhnzp35vu"
FIELD_TRX_PAYMENT_METHOD = "field-1758478644281-ct8pck6lk"

TRX_PAYER_VALUES = frozenset({"Customer", "Invest Greece"})
INVOICE_SOURCE_VALUES = frozenset({"Solomon Invoice", "Third Party Invoice"})
TRX_PAYMENT_METHOD_VALUES = frozenset({"In Person", "Bank"})

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

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
    "total_amount": (
        "Total amount due now in euros — the current bill total payable. "
        "Look for equivalent labels such as ΣΥΝΟΛΟ, ΠΛΗΡΩΤΕΟ, ΤΕΛΙΚΟ ΠΛΗΡΩΤΕΟ, "
        "Total Due, Amount Due, Total Payable, Current Bill Total. "
        "Numeric string with dot decimal. Leave empty if not found."
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

WATER_BILL_EXTRACTION_SCHEMA = {
    "total_amount": (
        "Total payable amount in euros. Look for equivalent labels such as "
        "ΠΟΣΟ ΠΛΗΡΩΜΗΣ, ΠΛΗΡΩΤΕΟ, ΣΥΝΟΛΟ, Total Due, Amount Payable, Payment Amount. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
    "receipt_number": (
        "Receipt / document / invoice number. Look for equivalent labels such as "
        "ΑΡΙΘΜΟΣ ΠΑΡΑΣΤΑΤΙΚΟΥ, ΑΡ. ΠΑΡΑΣΤΑΤΙΚΟΥ, Document No, Invoice No, Receipt No. "
        "Return digits and characters only, without spaces. Leave empty if not found."
    ),
    "payment_due_date": (
        "Payment due date. Return as YYYY-MM-DD regardless of how it appears on the document."
    ),
    "water_meter_no": (
        "Water supply registry / meter registry number — typically numeric, "
        "often with a hyphen (e.g. 2344821-60). May appear under labels such as "
        "ΑΡΙΘΜΟΣ ΜΗΤΡΩΟΥ, Registry No, Meter Registry, Supply Registry. "
        "Do NOT use the alphanumeric account/contract code for this field. "
        "Leave empty if not found."
    ),
    "water_account_no": (
        "Water supply account / contract identifier — typically alphanumeric "
        "(e.g. A22B89439). May appear under labels such as "
        "ΑΡΙΘΜΟΣ ΜΕΤΡΗΤΗ, Account No, Supply Account, Customer Account, Contract No. "
        "Do NOT use the numeric registry/meter number for this field. "
        "Leave empty if not found."
    ),
    "service_period": (
        "Consumption or billing service period exactly as shown on the document "
        "(e.g. 11/10/2025-29/05/2026, 01/01/2026 - 31/01/2026). "
        "Preserve original date format. Leave empty if not found."
    ),
    "who_gets_money": (
        "Water utility company that receives payment — usually the company logo or name "
        "on the bill header (e.g. water supplier / δημοτική επιχείρηση ύδρευσης). "
        "Return the company name as printed. Leave empty if unclear."
    ),
    "previous_balance_due": (
        "Previous outstanding/unpaid balance in euros from prior bills. "
        "Look for equivalent labels such as Προηγούμενο Υπόλοιπο, Προηγ. Οφειλές, "
        "Outstanding Balance, Previous Balance, Prior Amount Due. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
    "final_interim_bill": (
        "Bill settlement type. Return 'Final' for settlement/clearance/final bills "
        "(e.g. Εκκαθαριστικός, Final Bill). "
        "Return 'Interim' for estimated/provisional bills (e.g. Ενδιάμεσος, Estimated). "
        "Leave empty if unclear."
    ),
}

BANK_RECEIPT_EXTRACTION_SCHEMA = {
    "trx_ref_no": (
        "Unique transaction / reference / receipt number. Look for labels such as "
        "Unique Transaction Number, Transaction Reference, Reference No, Αριθμός Συναλλαγής. "
        "Preserve the value as printed (including slash separators). Leave empty if not found."
    ),
    "trx_date": (
        "Transfer / transaction date. Return as YYYY-MM-DD only (ignore time of day)."
    ),
    "trx_value_date": (
        "Value date of the transfer. Return as YYYY-MM-DD only. "
        "If only one date is present, use the same date as trx_date."
    ),
    "trx_bank": (
        "Bank name issuing the receipt (logo or header, e.g. Optima bank, Eurobank, Alpha Bank). "
        "Return the bank name in title case Latin characters when possible."
    ),
    "total_amount": (
        "Transfer amount in euros. Numeric string with dot decimal. Leave empty if not found."
    ),
    "transaction_type": (
        "Transaction type / title of the transfer "
        "(e.g. Transfer to third party within the bank, SEPA transfer, Instant payment). "
        "Leave empty if not found."
    ),
    "payment_reference": (
        "Payment details / payment reference / remittance information shown on the receipt "
        "(e.g. Athinon66 Patra, for mert KILIC). Leave empty if not found."
    ),
    "beneficiary_name": (
        "Beneficiary / payee name (who receives the money). Leave empty if not found."
    ),
    "beneficiary_account": (
        "Beneficiary account number or IBAN (To account). Leave empty if not found."
    ),
}

INVOICE_EXTRACTION_SCHEMA = {
    "trx_payer": (
        "Who is the payer of this invoice transaction. Must be exactly one of: "
        "'Customer' or 'Invest Greece'. "
        "Use 'Invest Greece' when Invest Greece / Solomon United Realtors / related "
        "company is paying or is the bill-to/customer party on the invoice. "
        "Use 'Customer' when an external client/tenant/customer is the payer. "
        "Leave empty if unclear."
    ),
    "company_address": (
        "Issuer company address as printed on the invoice. Leave empty if not found."
    ),
    "tax_office": (
        "Tax office (Δ.Ο.Υ. / DOY / Tax Office) of the issuer. Leave empty if not found."
    ),
    "cust_info_bill_to": (
        "Customer / Bill To party details (name and address if present). Leave empty if not found."
    ),
    "service_description": (
        "Description of goods/services on the invoice line items. Leave empty if not found."
    ),
    "service_period": (
        "Service / billing period if shown (e.g. June 2026, 01/06/2026-30/06/2026). "
        "Leave empty if not found."
    ),
    "bank_iban_info": (
        "Bank name and/or IBAN payment details for settling the invoice. Leave empty if not found."
    ),
    "invoice_issuer": (
        "Official company or person name of the invoice issuer (seller / vendor), "
        "copied EXACTLY as printed on the invoice. Preserve all punctuation, "
        "abbreviations, and legal-form suffixes (e.g. Ο.Ε., Α.Ε., Ι.Κ.Ε., Ε.Π.Ε., O.E., S.A.). "
        "Do not expand, translate, title-case, or otherwise rewrite the name. "
        "Leave empty if not found."
    ),
    "issuer_profession": (
        "Issuer profession / business activity / επάγγελμα as printed near the company "
        "header (e.g. ΥΔΡΑΥΛΙΚΕΣ ΕΡΓΑΣΙΕΣ). Copy exactly as printed; do not translate. "
        "Do not use invoice line-item descriptions. Leave empty if not found."
    ),
    "invoice_issue_date": (
        "Invoice issue date. Return as YYYY-MM-DD."
    ),
    "invoice_recipient": (
        "Invoice recipient / buyer / billed party name. Leave empty if not found."
    ),
    "invoice_source": (
        "Must be exactly one of: 'Solomon Invoice' or 'Third Party Invoice'. "
        "Use 'Solomon Invoice' when issued by Solomon / Invest Greece / Solomon United Realtors. "
        "Use 'Third Party Invoice' when issued by any other company. Leave empty if unclear."
    ),
    "invoice_issuer_tax_id": (
        "Issuer Α.Φ.Μ. (AFM / Tax Number) exactly as printed. Leave empty if not found."
    ),
    "invoice_recipient_tax_id": (
        "Recipient tax ID / Α.Φ.Μ. / VAT number. Leave empty if not found."
    ),
    "invoice_issued_by": (
        "Person who issued/signed/prepared the invoice if shown. Leave empty if not found."
    ),
    "invoice_number": (
        "Invoice number / document number. Leave empty if not found."
    ),
    "amount_excluding_vat": (
        "Net amount excluding VAT. Numeric string with dot decimal. Leave empty if not found."
    ),
    "vat_ratio": (
        "VAT percentage rate as a number without % sign (e.g. 24, 13, 0). Leave empty if not found."
    ),
    "vat_amount": (
        "VAT amount in euros. Numeric string with dot decimal. Leave empty if not found."
    ),
    "trx_payment_method": (
        "Payment method. Must be exactly one of: 'In Person' or 'Bank'. "
        "Use 'Bank' for bank transfer / IBAN / deposit. Use 'In Person' for cash / in-person payment. "
        "Leave empty if unclear."
    ),
    "total_amount": (
        "Total amount including VAT / grand total payable. "
        "Numeric string with dot decimal. Leave empty if not found."
    ),
    "invest_greece_notes": (
        "Any relevant notes, remarks, or payment references from the invoice. "
        "Leave empty if none."
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
    if document_type.strip() == DOCUMENT_TYPE_WATER_BILL:
        return DOCUMENT_TYPE_WATER_BILL
    if document_type.strip() == DOCUMENT_TYPE_BANK_RECEIPT:
        return DOCUMENT_TYPE_BANK_RECEIPT
    if document_type.strip() == DOCUMENT_TYPE_INVOICE:
        return DOCUMENT_TYPE_INVOICE
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
    text = value.strip()
    # Drop time portion if present (e.g. "10/07/2026 14:53" or "10/07/2026 (Now)")
    text = re.split(r"\s+", text, maxsplit=1)[0]
    text = text.strip("()")
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if not parsed:
        return value.strip()

    local = datetime(
        parsed.year, parsed.month, parsed.day, tzinfo=ZoneInfo("Europe/Athens")
    )
    utc = local.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _to_dd_mm_yyyy(value: str) -> str:
    """Format a calendar date as DD/MM/YYYY with no timezone conversion."""
    if not value:
        return ""
    text = value.strip()
    # Drop time portion if present (e.g. "10/07/2026 14:53" or "10/07/2026 (Now)")
    text = re.split(r"\s+", text, maxsplit=1)[0]
    text = text.strip("()")
    parsed: datetime | None = None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    if not parsed:
        return value.strip()
    return parsed.strftime("%d/%m/%Y")


def _parse_amount_two_decimals(value: Any) -> str | None:
    parsed = _parse_amount(value)
    if not parsed:
        return None
    try:
        return f"{float(parsed):.2f}"
    except ValueError:
        return parsed


def _month_year_from_iso(iso_date: str) -> tuple[str, str]:
    if not iso_date:
        return "", ""
    try:
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        local = dt.astimezone(ZoneInfo("Europe/Athens"))
        return MONTH_NAMES[local.month - 1], str(local.year)
    except ValueError:
        return "", ""


def _format_bank_name(value: str) -> str:
    text = _normalize_payee_name(value).strip()
    if not text:
        return ""
    return " ".join(part.capitalize() if part.isupper() or part.islower() else part for part in text.split())


def _format_beneficiary_who_gets_money(name: str, account: str) -> str:
    name = _normalize_payee_name(name).strip()
    account = account.strip()
    if name and account:
        return f"{name} (Account No: {account})"
    if name:
        return name
    if account:
        return f"(Account No: {account})"
    return ""


def _format_invest_greece_notes(transaction_type: str, payment_reference: str) -> str:
    parts: list[str] = []
    if transaction_type.strip():
        parts.append(f"Transaction type: {transaction_type.strip()}")
    if payment_reference.strip():
        parts.append(f"Payment reference: {payment_reference.strip()}")
    return " | ".join(parts)


def _normalize_receipt_number(value: str) -> str:
    return re.sub(r"\s+", "", value.strip())


def _normalize_water_service_period(value: str) -> str:
    return value.strip()


def _normalize_bill_type(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in ("final", "εκκαθαριστικός", "settlement", "clearance"):
        return "Final"
    if lower in ("interim", "ενδιάμεσος", "estimated", "provisional"):
        return "Interim"
    if text in ("Final", "Interim"):
        return text
    return ""


def _normalize_enum(value: str, allowed: frozenset[str]) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    for option in allowed:
        if text.lower() == option.lower():
            return option
    return ""


def _normalize_trx_payer(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    if "invest" in text or "solomon" in text:
        return "Invest Greece"
    if "customer" in text or "client" in text or "πελάτ" in text:
        return "Customer"
    return _normalize_enum(value, TRX_PAYER_VALUES)


def _normalize_invoice_source(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    if "solomon" in text or "invest greece" in text:
        return "Solomon Invoice"
    if "third" in text or "3rd" in text or "external" in text:
        return "Third Party Invoice"
    return _normalize_enum(value, INVOICE_SOURCE_VALUES)


def _normalize_trx_payment_method(value: str) -> str:
    text = (value or "").strip().lower()
    if not text:
        return ""
    if any(token in text for token in ("bank", "iban", "transfer", "έμβασ", "κατάθεσ")):
        return "Bank"
    if any(token in text for token in ("person", "cash", "μετρητ", "ταμείο", "in person")):
        return "In Person"
    return _normalize_enum(value, TRX_PAYMENT_METHOD_VALUES)


def _normalize_provider(value: str) -> str:
    return value.strip()


def _to_latin_transliteration(value: str) -> str:
    """Convert Greek company names to Latin characters (e.g. ΕΥΔΑΠ -> EYDAP)."""
    if not value:
        return ""

    greek_map = {
        "Α": "A", "Β": "B", "Γ": "G", "Δ": "D", "Ε": "E", "Ζ": "Z", "Η": "I", "Θ": "TH",
        "Ι": "I", "Κ": "K", "Λ": "L", "Μ": "M", "Ν": "N", "Ξ": "X", "Ο": "O", "Π": "P",
        "Ρ": "R", "Σ": "S", "Τ": "T", "Υ": "Y", "Φ": "F", "Χ": "CH", "Ψ": "PS", "Ω": "O",
        "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th",
        "ι": "i", "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p",
        "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f", "χ": "ch", "ψ": "ps",
        "ω": "o",
    }

    return "".join(greek_map.get(char, char) for char in value.strip())


def _normalize_payee_name(value: str) -> str:
    return _to_latin_transliteration(_normalize_provider(value))


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


def _extract_water_bill_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    return _extract_with_gemini(
        file_bytes,
        filename=filename,
        schema=WATER_BILL_EXTRACTION_SCHEMA,
        document_description=(
            "water utility bills and invoices from any water provider, "
            "in any layout, language (Greek or English), or format"
        ),
        extra_rules="""
- Templates vary widely across providers — extract by meaning, not by layout or coordinates.
- Do NOT assume a specific company, logo, or invoice template.
- Recognize equivalent terminology rather than exact labels.
- payment_due_date must be YYYY-MM-DD when confidently extracted.
- receipt_number must have all spaces removed.
- water_meter_no is the numeric registry number; water_account_no is the separate alphanumeric account/contract code.
""",
    )


def _extract_bank_receipt_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    return _extract_with_gemini(
        file_bytes,
        filename=filename,
        schema=BANK_RECEIPT_EXTRACTION_SCHEMA,
        document_description=(
            "bank transfer receipts and payment confirmations from any bank, "
            "in any layout, language (Greek or English), or format"
        ),
        extra_rules="""
- Templates vary widely across banks — extract by meaning, not by layout or coordinates.
- Do NOT assume a specific bank or receipt template.
- trx_date and trx_value_date must be YYYY-MM-DD (date only, no time).
- transaction_type is the transfer title/type; payment_reference is the remittance/payment details text.
- beneficiary_name and beneficiary_account are the payee receiving the funds (not the depositor).
""",
    )


def _extract_invoice_with_gemini(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, Any]:
    return _extract_with_gemini(
        file_bytes,
        filename=filename,
        schema=INVOICE_EXTRACTION_SCHEMA,
        document_description=(
            "commercial invoices (τιμολόγια) from any issuer, "
            "in any layout, language (Greek or English), or format"
        ),
        extra_rules="""
- Templates vary widely — extract by meaning, not by layout or coordinates.
- If the file contains multiple invoices, extract ONLY the first invoice.
- Do NOT assume a specific company or template.
- invoice_issuer must be the exact printed legal/company name, including punctuation
  and suffixes such as Ο.Ε. / Α.Ε. / Ι.Κ.Ε. — do not rewrite or normalize the name.
- issuer_profession is the επάγγελμα / activity near the issuer header (not line items).
- trx_payer must be exactly "Customer" or "Invest Greece" (or empty).
- invoice_source must be exactly "Solomon Invoice" or "Third Party Invoice" (or empty).
- trx_payment_method must be exactly "In Person" or "Bank" (or empty).
- invoice_issue_date must be YYYY-MM-DD when confidently extracted.
- Numeric amounts use dot as decimal separator.
""",
    )


def build_electricity_bill_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    provider = _normalize_provider(extracted.get("who_gets_money", ""))
    matching_number = (extracted.get("matching_number") or "").strip()
    receipt_no = (extracted.get("receipt_no") or "").strip()
    previous_balance = _parse_amount_optional(extracted.get("previous_balance_due"))
    total_amount = _parse_amount_optional(extracted.get("total_amount"))
    service_period = _normalize_service_period(extracted.get("service_period", ""))
    due_date = _to_iso_due_date(extracted.get("payment_due_date", ""))

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_ELECTRICITY_BILL,
        FIELD_CATEGORY: "Accrual",
        FIELD_DEBIT_CREDIT: "Debit",
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
        data[FIELD_WHO_GETS_MONEY_ALT] = _normalize_payee_name(provider)
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
    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount

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
        FIELD_DEBIT_CREDIT: "Debit",
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
        data[FIELD_WHO_GETS_MONEY_ALT] = _normalize_payee_name(who_gets_money)
    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount

    return _omit_unextracted_fields(data)


def build_water_bill_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    total_amount = _parse_amount_optional(extracted.get("total_amount"))
    receipt_number = _normalize_receipt_number(extracted.get("receipt_number", ""))
    due_date = _to_iso_due_date(extracted.get("payment_due_date", ""))
    meter_no = (extracted.get("water_meter_no") or "").strip()
    account_no = (extracted.get("water_account_no") or "").strip()
    service_period = _normalize_water_service_period(extracted.get("service_period", ""))
    who_gets_money = _normalize_provider(extracted.get("who_gets_money", ""))
    previous_balance = _parse_amount_optional(extracted.get("previous_balance_due"))
    bill_type = _normalize_bill_type(extracted.get("final_interim_bill", ""))

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_WATER_BILL,
        FIELD_CATEGORY: "Accrual",
        FIELD_DEBIT_CREDIT: "Debit",
    }

    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount
    if receipt_number:
        data[FIELD_MATCHING_NUMBER] = receipt_number
    if due_date:
        data[FIELD_PAYMENT_DUE_DATE] = due_date
    if meter_no:
        data[FIELD_WATER_METER_NO] = meter_no
    if account_no:
        data[FIELD_WATER_ACCOUNT_NO] = account_no
    if service_period:
        data[FIELD_SERVICE_PERIOD] = service_period
    if who_gets_money:
        data[FIELD_WHO_GETS_MONEY_ALT] = _normalize_payee_name(who_gets_money)
    if previous_balance is not None:
        data[FIELD_PREVIOUS_BALANCE_DUE] = previous_balance
        data[FIELD_OVERDUE_PAYMENT] = "Yes" if _amount_is_positive(previous_balance) else "No"
    if bill_type:
        data[FIELD_BILL_TYPE] = bill_type

    return _omit_unextracted_fields(data)


def build_bank_receipt_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    trx_ref = (extracted.get("trx_ref_no") or "").strip()
    trx_date = _to_iso_due_date(extracted.get("trx_date", ""))
    trx_value_date = _to_iso_due_date(extracted.get("trx_value_date", "")) or trx_date
    trx_bank = _format_bank_name(extracted.get("trx_bank", ""))
    total_amount = _parse_amount_two_decimals(extracted.get("total_amount"))
    notes = _format_invest_greece_notes(
        extracted.get("transaction_type", ""),
        extracted.get("payment_reference", ""),
    )
    who_gets_money = _format_beneficiary_who_gets_money(
        extracted.get("beneficiary_name", ""),
        extracted.get("beneficiary_account", ""),
    )
    month, year = _month_year_from_iso(trx_date or trx_value_date)

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_BANK_RECEIPT,
        FIELD_CATEGORY: "Payment",
        FIELD_PAYMENT_DIRECTION: "Incoming Payment",
    }

    if trx_ref:
        data[FIELD_TRX_REF_NO] = trx_ref
    if trx_date:
        data[FIELD_TRX_DATE] = trx_date
    if trx_value_date:
        data[FIELD_TRX_VALUE_DATE] = trx_value_date
    if trx_bank:
        data[FIELD_TRX_BANK] = trx_bank
    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount
    if notes:
        data[FIELD_INVEST_GREECE_NOTES] = notes
    if month:
        data[FIELD_MONTH] = month
    if year:
        data[FIELD_YEAR] = year
    if who_gets_money:
        data[FIELD_WHO_GETS_MONEY_ALT] = who_gets_money

    return _omit_unextracted_fields(data)


def build_invoice_cashflow_data(extracted: dict[str, Any]) -> dict[str, Any]:
    trx_payer = _normalize_trx_payer(extracted.get("trx_payer", ""))
    invoice_source = _normalize_invoice_source(extracted.get("invoice_source", ""))
    payment_method = _normalize_trx_payment_method(extracted.get("trx_payment_method", ""))
    issue_date = _to_dd_mm_yyyy(extracted.get("invoice_issue_date", ""))
    amount_ex_vat = _parse_amount_optional(extracted.get("amount_excluding_vat"))
    vat_amount = _parse_amount_optional(extracted.get("vat_amount"))
    total_amount = _parse_amount_optional(extracted.get("total_amount"))
    vat_ratio = _parse_amount_optional(extracted.get("vat_ratio"))

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: DOCUMENT_TYPE_INVOICE,
    }

    if trx_payer:
        data[FIELD_TRX_PAYER] = trx_payer
    company_address = (extracted.get("company_address") or "").strip()
    if company_address:
        data[FIELD_COMPANY_ADDRESS] = company_address
    tax_office = (extracted.get("tax_office") or "").strip()
    if tax_office:
        data[FIELD_TAX_OFFICE] = tax_office
    bill_to = (extracted.get("cust_info_bill_to") or "").strip()
    if bill_to:
        data[FIELD_CUST_INFO_BILL_TO] = bill_to
    service_description = (extracted.get("service_description") or "").strip()
    if service_description:
        data[FIELD_SERVICE_DESCRIPTION] = service_description
    service_period = (extracted.get("service_period") or "").strip()
    if service_period:
        data[FIELD_SERVICE_PERIOD] = service_period
    bank_iban = (extracted.get("bank_iban_info") or "").strip()
    if bank_iban:
        data[FIELD_BANK_IBAN_INFO] = bank_iban
    issuer = (extracted.get("invoice_issuer") or "").strip()
    if issuer:
        data[FIELD_INVOICE_ISSUER] = issuer
    if issue_date:
        data[FIELD_INVOICE_ISSUE_DATE] = issue_date
    recipient = (extracted.get("invoice_recipient") or "").strip()
    if recipient:
        data[FIELD_INVOICE_RECIPIENT] = recipient
    if invoice_source:
        data[FIELD_INVOICE_SOURCE] = invoice_source
    issuer_tax_id = (extracted.get("invoice_issuer_tax_id") or "").strip()
    if issuer_tax_id:
        data[FIELD_INVOICE_ISSUER_TAX_ID] = issuer_tax_id
    recipient_tax_id = (extracted.get("invoice_recipient_tax_id") or "").strip()
    if recipient_tax_id:
        data[FIELD_INVOICE_RECIPIENT_TAX_ID] = recipient_tax_id
    issued_by = (extracted.get("invoice_issued_by") or "").strip()
    if issued_by:
        data[FIELD_INVOICE_ISSUED_BY] = issued_by
    invoice_number = (extracted.get("invoice_number") or "").strip()
    if invoice_number:
        data[FIELD_INVOICE_NUMBER] = invoice_number
    if amount_ex_vat is not None:
        data[FIELD_AMOUNT_EXCLUDING_VAT] = amount_ex_vat
    if vat_ratio is not None:
        data[FIELD_VAT_RATIO] = vat_ratio
    if vat_amount is not None:
        data[FIELD_VAT_AMOUNT] = vat_amount
    if payment_method:
        data[FIELD_TRX_PAYMENT_METHOD] = payment_method
    if total_amount is not None:
        data[FIELD_TOTAL_AMOUNT] = total_amount
    notes = (extracted.get("invest_greece_notes") or "").strip()
    if notes:
        data[FIELD_INVEST_GREECE_NOTES] = notes

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

    if normalized == DOCUMENT_TYPE_WATER_BILL:
        extracted = _extract_water_bill_with_gemini(file_bytes, filename)
        return build_water_bill_cashflow_data(extracted)

    if normalized == DOCUMENT_TYPE_BANK_RECEIPT:
        extracted = _extract_bank_receipt_with_gemini(file_bytes, filename)
        return build_bank_receipt_cashflow_data(extracted)

    if normalized == DOCUMENT_TYPE_INVOICE:
        extracted = _extract_invoice_with_gemini(file_bytes, filename)
        return build_invoice_cashflow_data(extracted)

    raise ValueError(f"Document type '{document_type}' is not supported.")
