"""Optima bank statement import: parse, suggest, duplicate-check, save cashflows."""

from __future__ import annotations

import logging
import random
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from bson import ObjectId

from services.ledger_report import (
    CASHFLOW_FORM_URL_TEMPLATE,
    CASHFLOW_INDICATOR,
    FIELD_AMOUNT,
    FIELD_CATEGORY,
    FIELD_CASHFLOW_CONTACT,
    FIELD_DATE,
    FIELD_DEBIT_CREDIT,
    FIELD_DESCRIPTION,
    FIELD_PAYMENT_DIRECTION,
    FIELD_PROPERTY,
    PROPERTY_TITLE_FIELD,
    extract_piped_id,
)
from services.recurrent_accruals import (
    CONTACT_NAME_FIELD,
    KOSTAS_MERCHANT_ID,
    KOSTAS_OWNER_ID,
    TRX_DESCRIPTIONS,
    search_contacts,
)

logger = logging.getLogger(__name__)

GREECE_TZ = ZoneInfo("Europe/Athens")

ACCOUNTS_COLLECTION = "bank_import_accounts"
PAYEES_COLLECTION = "bank_import_payees"
IMPORTS_COLLECTION = "bank_statement_imports"

CASHFLOW_FORM_ID = "68c2eb885d8b5ec633d3be86"

FIELD_DOCUMENT_TYPE = "field-1758699529035-xp7lumgx5"
FIELD_TRX_REF_NO = "field-1757604618017-q2mtvmiqp"
FIELD_TRX_VALUE_DATE = "field-1757605079803-icw8ykc19"
FIELD_TRX_BANK = "field-1757605194423-ofbqnqfso"
FIELD_NOTES = "field-1757605632930-2hfg96qgr"
FIELD_TRX_BENEFICIARY = "field-1757605635397-x9kecghp5"
FIELD_RECORDED_BY = "field-1757605870503-s1lu31him"
FIELD_TRX_PAYMENT_METHOD = "field-1758478644281-ct8pck6lk"
FIELD_TRX_RECEIVER = "field-1758478869002-dpm43pot3"
FIELD_TRX_BANK_STATEMENT = "field-1758478917909-jqxoz2s3h"
FIELD_MONTH = "field-1759392145178-qbx06ungl"
FIELD_YEAR = "field-1759392151218-bmx5iidn6"
FIELD_SENDER_ACCOUNT = "field-1784669176413-x7f0x1p33"
FIELD_BENEFICIARY_ACCOUNT = "field-1784669169410-6s5x04ju0"
FIELD_RF_PAYMENT_CODE = "field-1781250415587-3dmfxioiz"

OPTIMA_TITLE = "Ιστορικό συναλλαγών λογαριασμού"
FEE_PREFIXES = (
    "ΕΞΟΔΑ ΜΕΤΑΦΟΡΑΣ",
    "ΕΞΟΔΑ ΠΛΗΡΩΜΗΣ",
    "Χρέωση εξόδων",
)

AUX_PROPERTY = (
    "AUX - İT İS NOT RELATED TO ANY PROPERTY - HER HANGİ BİR MÜLK İLE "
    "BAĞLANTISI YOK - DEN EİNAİ SİNDEDEMENO ME KAPİO AKİNİTO|4626743390357804"
)
FEE_RECEIVER = "ΤΡΑΠΕΖΑ OPTIMA BANK ΑΕ|1103506393340376"
RECORDED_BY_MARKER = "Bank import"

MONTH_NAMES = (
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
)

LEGACY_CATEGORY_MAP = {
    "Rent": "Rent Payment",
    "Electricity": "Electricity Bill",
    "Water": "Water Bill",
    "Internet": "Internet Bill",
}

DEFAULT_ACCOUNTS: list[dict[str, Any]] = [
    {
        "key": "united",
        "label": "Solomon United Realtors",
        "trxBank": "solomon united realtors - optima bank",
        "accountNo": "17000794027197",
        "holderLabel": "SOLOMON UNİTED COMPANY ACCOUNT",
        "holderPid": "1411189864006583",
    },
    {
        "key": "partners297",
        "label": "Solomon Partners 297",
        "trxBank": "Solomon Partners 297 - Optima Bank",
        "accountNo": "17035626027297",
        "holderLabel": "SOLOMON PARTNERS COMPANY ACCOUNT",
        "holderPid": "4045235964447361",
    },
    {
        "key": "partners197",
        "label": "Solomon Partners 197",
        "trxBank": "Solomon Partners 197 - Optima Bank",
        "accountNo": "17035626027197",
        "holderLabel": "SOLOMON PARTNERS COMPANY ACCOUNT",
        "holderPid": "4045235964447361",
    },
    {
        "key": "personal",
        "label": "Kostas personal",
        "trxBank": "kostas arslanoglu personal - optima bank",
        "accountNo": "17000737022197",
        "holderLabel": "kostas arslanoğlu admin merkez",
        "holderPid": "9608276816853062",
    },
]

DEFAULT_PAYEES: list[dict[str, Any]] = [
    {"match": "ZENITH", "category": "Electricity Bill", "receiverLabel": "ΖΕΝΙΘ", "receiverPid": ""},
    {"match": "ΖΕΝΙΘ", "category": "Electricity Bill", "receiverLabel": "ΖΕΝΙΘ", "receiverPid": ""},
    {"match": "ΔΕΗ", "category": "Electricity Bill", "receiverLabel": "ΔΕΗ", "receiverPid": ""},
    {"match": "ΕΥΔΑΠ", "category": "Water Bill", "receiverLabel": "ΕΥΔΑΠ", "receiverPid": ""},
    {"match": "INTERASCO", "category": "Insurance Fees", "receiverLabel": "INTERASCO", "receiverPid": ""},
    {"match": "ΕΦΚΑ", "category": "Social Security Expenses", "receiverLabel": "ΕΦΚΑ", "receiverPid": ""},
    {"match": "e-Παράβολο", "category": "Land Registry Fee", "receiverLabel": "e-Παράβολο", "receiverPid": ""},
    {"match": "GOOGLE ADS", "category": "IT Expenses", "receiverLabel": "Google", "receiverPid": ""},
    {"match": "GOOGLE*WORKSPACE", "category": "IT Expenses", "receiverLabel": "Google", "receiverPid": ""},
    {"match": "FACEBK", "category": "IT Expenses", "receiverLabel": "Facebook", "receiverPid": ""},
    {"match": "FACEBOOK", "category": "IT Expenses", "receiverLabel": "Facebook", "receiverPid": ""},
    {"match": "Vodafone", "category": "Mobile Phone", "receiverLabel": "Vodafone", "receiverPid": ""},
    {"match": "Allianz", "category": "Insurance Fees", "receiverLabel": "Allianz", "receiverPid": ""},
]

GREEK_TO_LATIN = str.maketrans(
    {
        "Α": "A",
        "Β": "V",
        "Γ": "G",
        "Δ": "D",
        "Ε": "E",
        "Ζ": "Z",
        "Η": "I",
        "Θ": "TH",
        "Ι": "I",
        "Κ": "K",
        "Λ": "L",
        "Μ": "M",
        "Ν": "N",
        "Ξ": "X",
        "Ο": "O",
        "Π": "P",
        "Ρ": "R",
        "Σ": "S",
        "Τ": "T",
        "Υ": "Y",
        "Φ": "F",
        "Χ": "CH",
        "Ψ": "PS",
        "Ω": "O",
        "ά": "a",
        "έ": "e",
        "ή": "i",
        "ί": "i",
        "ό": "o",
        "ύ": "y",
        "ώ": "o",
        "ϊ": "i",
        "ϋ": "y",
        "ΐ": "i",
        "ΰ": "y",
        "ς": "s",
        "α": "a",
        "β": "v",
        "γ": "g",
        "δ": "d",
        "ε": "e",
        "ζ": "z",
        "η": "i",
        "θ": "th",
        "ι": "i",
        "κ": "k",
        "λ": "l",
        "μ": "m",
        "ν": "n",
        "ξ": "x",
        "ο": "o",
        "π": "p",
        "ρ": "r",
        "σ": "s",
        "τ": "t",
        "υ": "y",
        "φ": "f",
        "χ": "ch",
        "ψ": "ps",
        "ω": "o",
    }
)


def _build_piped(label: str, pid: str | int) -> str:
    return f"{label}|{pid}"


def holder_piped(account: dict[str, Any]) -> str:
    return _build_piped(account["holderLabel"], account["holderPid"])


def normalize_account_no(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace(" ", "").replace(",", "")
    if text.endswith(".0"):
        text = text[:-2]
    # Prefer an explicit long digit run (Optima account numbers) over
    # stripping digits from timestamps like "03/11/2026 09:48:51".
    match = re.search(r"\d{10,}", text.replace(" ", ""))
    if match:
        return match.group(0)
    digits = re.sub(r"[^\d]", "", text)
    # Reject short / date-like leftovers (ddmmyyyyhhmmss etc.)
    if len(digits) < 10:
        return ""
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}.*", str(value).strip()):
        return ""
    return digits


def parse_greek_amount(raw: Any) -> tuple[float, float]:
    """Return (signed_amount, absolute_amount) from Optima amount text."""
    text = str(raw or "").strip()
    text = text.replace("EUR", "").replace("€", "").strip()
    negative = text.startswith("-") or text.startswith("−")
    text = text.lstrip("-−").strip()
    text = text.replace(".", "").replace(",", ".")
    if not text:
        return 0.0, 0.0
    amount = float(text)
    if negative:
        amount = -amount
    return amount, abs(amount)


def parse_optima_date(raw: Any) -> date | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def athens_midnight_utc(d: date) -> datetime:
    local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=GREECE_TZ)
    return local.astimezone(timezone.utc)


def month_year_from_date(d: date) -> tuple[str, str]:
    return MONTH_NAMES[d.month - 1], str(d.year)


def greek_to_latin_lower(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "").strip()
    translated = cleaned.translate(GREEK_TO_LATIN)
    translated = re.sub(r"\s+", " ", translated).strip().lower()
    return translated


def extract_counterparty_name(details: str, short_desc: str = "") -> str:
    details_s = (details or "").strip()
    short_s = (short_desc or "").strip()
    generic_details = greek_to_latin_lower(details_s) in {
        "agora meso kartas",
        "αγορα μεσω καρτας",
    } or len(details_s) < 5
    source = short_s if (generic_details and short_s) else (details_s or short_s)
    if not source:
        return ""
    # Drop leading transfer verbs from short desc
    for prefix in (
        "ΜΕΤΑΦΟΡΑ ΑΠΟ ",
        "ΜΕΤΑΦΟΡΑ ΠΡΟΣ ",
        "ΠΛΗΡΩΜΗ ",
        "TCR κατάθεση ",
        "TCR ΚΑΤΑΘΕΣΗ ",
    ):
        if source.upper().startswith(prefix.upper()) or source.startswith(prefix):
            source = source[len(prefix) :].strip()
            break
    # Col G usually: NAME - IBAN - BANK - reason
    name = source.split(" - ")[0].strip()
    name = re.sub(r"\.\.\.$", "", name).strip()
    return greek_to_latin_lower(name)


def extract_iban(text: str) -> str | None:
    match = re.search(r"\b([A-Z]{2}\d{2}[A-Z0-9]{10,30})\b", text or "", re.I)
    return match.group(1).upper() if match else None


def extract_rf_code(text: str) -> str | None:
    match = re.search(r"\b(RF\d{10,})\b", text or "", re.I)
    return match.group(1).upper() if match else None


def is_fee_line(short_desc: str) -> bool:
    text = (short_desc or "").strip()
    return any(text.startswith(prefix) for prefix in FEE_PREFIXES)


def _cell(row: list[Any], idx: int) -> str:
    if idx >= len(row):
        return ""
    val = row[idx]
    if val is None:
        return ""
    return str(val).strip()


def parse_optima_workbook(file_bytes: bytes) -> dict[str, Any]:
    """Parse Optima 'Ιστορικό συναλλαγών λογαριασμού' xlsx."""
    import io

    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise ValueError("Empty workbook.")

    title = _cell(list(rows[0]), 0)
    title_ok = (
        OPTIMA_TITLE in title
        or "συναλλαγών" in title.lower()
        or "Ιστορικό" in title
        or "Istoriko" in title.lower()
    )
    if not title_ok:
        raise ValueError(
            "File does not look like an Optima account transaction history export."
        )

    account_no = ""
    closing_balance = None
    period = ""
    # Scan first 6 rows for header labels
    for i, row in enumerate(rows[:6]):
        cells = list(row)
        a = _cell(cells, 0)
        b = _cell(cells, 1)
        label_hints = (
            "Αριθμός" in a
            or "λογαριασμ" in a.lower()
            or "RECEIVER" in a.upper()
            or "ACCOUNT" in a.upper()
        )
        digits_b = normalize_account_no(b)
        digits_a = normalize_account_no(a)
        digits = digits_b or digits_a
        if digits and len(digits) >= 10:
            if label_hints:
                account_no = digits
            elif not account_no and i == 2 and digits_b:
                account_no = digits_b
        if "υπόλοιπο" in a.lower() or "BALANCE" in a.upper() or a.startswith("Διαθέσιμο"):
            try:
                bal_text = b or a.split(":")[-1]
                if bal_text:
                    _, closing_balance = parse_greek_amount(bal_text)
            except Exception:
                pass
        if a.startswith("Περίοδος") or "ΕΡΙΟΔ" in a.upper() or "Period" in a:
            period = b or a.split(":", 1)[-1].strip()

    if not account_no and len(rows) > 2:
        # Row 3 (index 2) value column is the usual Optima account number cell
        account_no = normalize_account_no(_cell(list(rows[2]), 1))

    # Find header row (has date / Ημερομηνία)
    header_idx = None
    for i, row in enumerate(rows[:12]):
        a = _cell(list(row), 0).upper()
        if "ΗΜΕΡΟΜΗΝΙΑ" in a or a.startswith("TRX DATE") or "DATE" == a.strip():
            header_idx = i
            break
    if header_idx is None:
        header_idx = 5

    lines: list[dict[str, Any]] = []
    for row in rows[header_idx + 1 :]:
        cells = list(row)
        date_raw = _cell(cells, 0)
        if not date_raw or not re.search(r"\d", date_raw):
            # Stop on trailing notes without dates
            if lines and not date_raw:
                continue
            continue
        trx_date = parse_optima_date(date_raw)
        if not trx_date:
            continue
        short_desc = _cell(cells, 1)
        ref = _cell(cells, 2)
        amount_raw = _cell(cells, 3)
        balance_raw = _cell(cells, 4)
        value_date = parse_optima_date(_cell(cells, 5)) or trx_date
        details = _cell(cells, 6)

        signed, absolute = parse_greek_amount(amount_raw)
        try:
            _, balance = parse_greek_amount(balance_raw)
        except Exception:
            balance = None

        fee = is_fee_line(short_desc)
        direction = "Incoming Payment" if signed >= 0 else "Outgoing Payment"
        debit_credit = "Credit" if signed >= 0 else "Debit"

        lines.append(
            {
                "trxDate": trx_date.isoformat(),
                "valueDate": value_date.isoformat(),
                "shortDesc": short_desc,
                "details": details,
                "refNo": ref,
                "signedAmount": signed,
                "amount": round(absolute, 2),
                "balance": balance,
                "isFee": fee,
                "paymentDirection": direction,
                "debitCredit": debit_credit,
                "statementText": details or short_desc,
            }
        )

    if not lines:
        raise ValueError("No transaction lines found in the Optima file.")

    # Integrity: newest line balance vs header (best-effort)
    integrity: dict[str, Any] = {
        "lineCount": len(lines),
        "headerBalance": closing_balance,
        "newestBalance": lines[0].get("balance"),
        "newestMatchesHeader": None,
        "runningBalanceOk": None,
    }
    if closing_balance is not None and lines[0].get("balance") is not None:
        integrity["newestMatchesHeader"] = abs(lines[0]["balance"] - closing_balance) < 0.02

    # Running balance oldest → newest
    chronological = list(reversed(lines))
    if chronological and chronological[0].get("balance") is not None:
        ok = True
        prev_balance = chronological[0]["balance"] - chronological[0]["signedAmount"]
        for line in chronological:
            expected = round(prev_balance + line["signedAmount"], 2)
            if line.get("balance") is not None and abs(expected - line["balance"]) > 0.05:
                ok = False
                break
            if line.get("balance") is not None:
                prev_balance = line["balance"]
            else:
                prev_balance = expected
        integrity["runningBalanceOk"] = ok

    return {
        "accountNo": account_no,
        "closingBalance": closing_balance,
        "period": period,
        "lines": lines,
        "integrity": integrity,
    }


async def ensure_default_accounts(console_db) -> list[dict[str, Any]]:
    col = console_db[ACCOUNTS_COLLECTION]
    count = await col.count_documents({})
    if count == 0:
        now = datetime.now(timezone.utc)
        docs = [{**a, "createdAt": now, "updatedAt": now} for a in DEFAULT_ACCOUNTS]
        await col.insert_many(docs)
    return await list_accounts(console_db)


async def list_accounts(console_db) -> list[dict[str, Any]]:
    cursor = console_db[ACCOUNTS_COLLECTION].find({}).sort("key", 1)
    out = []
    async for doc in cursor:
        out.append(serialize_account(doc))
    return out


def serialize_account(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "key": doc.get("key") or "",
        "label": doc.get("label") or "",
        "trxBank": doc.get("trxBank") or "",
        "accountNo": normalize_account_no(doc.get("accountNo")),
        "holderLabel": doc.get("holderLabel") or "",
        "holderPid": str(doc.get("holderPid") or ""),
        "holderPiped": _build_piped(doc.get("holderLabel") or "", doc.get("holderPid") or ""),
    }


async def upsert_account(console_db, payload: dict[str, Any], account_id: str | None = None) -> dict:
    col = console_db[ACCOUNTS_COLLECTION]
    now = datetime.now(timezone.utc)
    fields = {
        "key": (payload.get("key") or "").strip(),
        "label": (payload.get("label") or "").strip(),
        "trxBank": (payload.get("trxBank") or "").strip(),
        "accountNo": normalize_account_no(payload.get("accountNo")),
        "holderLabel": (payload.get("holderLabel") or "").strip(),
        "holderPid": str(payload.get("holderPid") or "").strip(),
        "updatedAt": now,
    }
    if not fields["trxBank"] or not fields["accountNo"] or not fields["holderPid"]:
        raise ValueError("trxBank, accountNo and holderPid are required.")
    if account_id:
        await col.update_one({"_id": ObjectId(account_id)}, {"$set": fields})
        doc = await col.find_one({"_id": ObjectId(account_id)})
    else:
        fields["createdAt"] = now
        if not fields["key"]:
            fields["key"] = fields["accountNo"][-4:]
        result = await col.insert_one(fields)
        doc = await col.find_one({"_id": result.inserted_id})
    return serialize_account(doc)


async def ensure_default_payees(console_db) -> list[dict[str, Any]]:
    col = console_db[PAYEES_COLLECTION]
    count = await col.count_documents({})
    if count == 0:
        now = datetime.now(timezone.utc)
        docs = [{**p, "createdAt": now, "updatedAt": now} for p in DEFAULT_PAYEES]
        await col.insert_many(docs)
    return await list_payees(console_db)


async def list_payees(console_db) -> list[dict[str, Any]]:
    cursor = console_db[PAYEES_COLLECTION].find({}).sort("match", 1)
    out = []
    async for doc in cursor:
        out.append(
            {
                "id": str(doc["_id"]),
                "match": doc.get("match") or "",
                "category": doc.get("category") or "",
                "receiverLabel": doc.get("receiverLabel") or "",
                "receiverPid": str(doc.get("receiverPid") or ""),
            }
        )
    return out


async def upsert_payee(console_db, payload: dict[str, Any], payee_id: str | None = None) -> dict:
    col = console_db[PAYEES_COLLECTION]
    now = datetime.now(timezone.utc)
    fields = {
        "match": (payload.get("match") or "").strip(),
        "category": (payload.get("category") or "").strip(),
        "receiverLabel": (payload.get("receiverLabel") or "").strip(),
        "receiverPid": str(payload.get("receiverPid") or "").strip(),
        "updatedAt": now,
    }
    if not fields["match"] or not fields["category"]:
        raise ValueError("match and category are required.")
    if payee_id:
        await col.update_one({"_id": ObjectId(payee_id)}, {"$set": fields})
        doc = await col.find_one({"_id": ObjectId(payee_id)})
    else:
        fields["createdAt"] = now
        result = await col.insert_one(fields)
        doc = await col.find_one({"_id": result.inserted_id})
    return {
        "id": str(doc["_id"]),
        "match": doc.get("match") or "",
        "category": doc.get("category") or "",
        "receiverLabel": doc.get("receiverLabel") or "",
        "receiverPid": str(doc.get("receiverPid") or ""),
    }


async def delete_payee(console_db, payee_id: str) -> None:
    await console_db[PAYEES_COLLECTION].delete_one({"_id": ObjectId(payee_id)})


def map_legacy_category(category: str | None) -> str | None:
    if not category:
        return None
    mapped = LEGACY_CATEGORY_MAP.get(category, category)
    if mapped in TRX_DESCRIPTIONS:
        return mapped
    return None


def _new_cashflow_pid() -> float:
    return float(random.randint(10**15, 10**16 - 1))


def _prop360_owner_id() -> str:
    import os

    return os.getenv("TRANSFER_NEW_OWNER_ID") or KOSTAS_OWNER_ID


def build_cashflow_data_from_row(
    row: dict[str, Any],
    account: dict[str, Any],
) -> dict[str, Any]:
    trx_date = date.fromisoformat(row["trxDate"][:10])
    value_date = date.fromisoformat((row.get("valueDate") or row["trxDate"])[:10])
    month, year = month_year_from_date(trx_date)
    amount = float(row["amount"])
    direction = row["paymentDirection"]
    is_incoming = direction == "Incoming Payment"

    data: dict[str, Any] = {
        FIELD_DOCUMENT_TYPE: "Bank Transaction",
        FIELD_CATEGORY: "Payment",
        FIELD_TRX_PAYMENT_METHOD: "Bank",
        FIELD_TRX_BANK: account["trxBank"],
        FIELD_RECORDED_BY: RECORDED_BY_MARKER,
        FIELD_DATE: athens_midnight_utc(trx_date),
        FIELD_TRX_VALUE_DATE: athens_midnight_utc(value_date),
        FIELD_MONTH: month,
        FIELD_YEAR: year,
        FIELD_AMOUNT: amount,
        FIELD_PAYMENT_DIRECTION: direction,
        FIELD_DEBIT_CREDIT: row["debitCredit"],
        FIELD_TRX_REF_NO: row.get("refNo") or "",
        FIELD_TRX_BANK_STATEMENT: row.get("trxBankStatement") or "",
        FIELD_NOTES: row.get("investGreeceNotes") or "",
        FIELD_DESCRIPTION: row.get("trxDescription") or "",
        FIELD_TRX_BENEFICIARY: row.get("trxBeneficiary") or "",
        FIELD_TRX_RECEIVER: row.get("trxReceiver") or "",
        FIELD_PROPERTY: row.get("propertyId") or "",
        FIELD_CASHFLOW_CONTACT: row.get("propertyOwner") or "",
        "attachedForms": [],
    }
    acct = account["accountNo"]
    if is_incoming:
        data[FIELD_BENEFICIARY_ACCOUNT] = float(acct) if acct.isdigit() else acct
    else:
        data[FIELD_SENDER_ACCOUNT] = float(acct) if acct.isdigit() else acct
    return data


async def create_bank_import_cashflow(
    prop_db,
    data: dict[str, Any],
    *,
    import_id: str,
    row_index: int,
) -> str:
    owner = _prop360_owner_id()
    now = datetime.now(timezone.utc)
    pid = _new_cashflow_pid()
    for _ in range(5):
        existing = await prop_db.formdatas.find_one({"pid": pid}, {"_id": 1})
        if not existing:
            break
        pid = _new_cashflow_pid()

    data_payload = dict(data)
    data_payload.setdefault("attachedForms", [])
    doc_id = ObjectId()
    doc = {
        "_id": doc_id,
        "formId": CASHFLOW_FORM_ID,
        "indicator": CASHFLOW_INDICATOR,
        "isPublic": False,
        "data": data_payload,
        "metadata": {
            "createdAt": now,
            "updatedAt": now,
            "createdBy": owner,
            "updatedBy": owner,
            "spitogatosPropertyId": None,
            "paymentFormDataIds": [],
            "paymentCount": 0,
            "paymentStatus": "none",
            "paymentTotalAmount": 0,
            "consoleBankImport": True,
            "bankImportSessionId": import_id,
            "bankImportRowIndex": row_index,
        },
        "owner": owner,
        "merchantId": KOSTAS_MERCHANT_ID,
        "sharedWithMerchants": [KOSTAS_MERCHANT_ID],
        "sharedWith": [],
        "status": "active",
        "destructive": False,
        "pid": pid,
        "isWorkflowForm": False,
        "__v": 0,
    }
    await prop_db.formdatas.insert_one(doc)
    return str(doc_id)


async def find_exact_duplicate(
    prop_db,
    *,
    trx_bank: str,
    ref_no: str,
    amount: float,
    direction: str,
) -> dict[str, Any] | None:
    if not ref_no:
        return None
    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        f"data.{FIELD_TRX_BANK}": trx_bank,
        f"data.{FIELD_TRX_REF_NO}": ref_no,
        f"data.{FIELD_PAYMENT_DIRECTION}": direction,
        f"data.{FIELD_AMOUNT}": {"$in": [amount, str(amount), f"{amount:.2f}", f"{amount:.1f}"]},
    }
    doc = await prop_db.formdatas.find_one(query, {"_id": 1, "data": 1})
    if doc:
        return doc
    # Amount sometimes stored as number without trailing zeros issues — broaden
    cursor = prop_db.formdatas.find(
        {
            "indicator": CASHFLOW_INDICATOR,
            "status": "active",
            f"data.{FIELD_TRX_BANK}": trx_bank,
            f"data.{FIELD_TRX_REF_NO}": ref_no,
            f"data.{FIELD_PAYMENT_DIRECTION}": direction,
        },
        {"_id": 1, "data": 1},
    ).limit(20)
    async for candidate in cursor:
        raw = (candidate.get("data") or {}).get(FIELD_AMOUNT)
        try:
            if abs(float(raw) - amount) < 0.005:
                return candidate
        except (TypeError, ValueError):
            continue
    return None


async def find_possible_duplicate(
    prop_db,
    *,
    account_no: str,
    amount: float,
    direction: str,
    trx_date: date,
) -> dict[str, Any] | None:
    start = trx_date - timedelta(days=4)
    end = trx_date + timedelta(days=4)
    start_dt = athens_midnight_utc(start) - timedelta(hours=12)
    end_dt = athens_midnight_utc(end) + timedelta(hours=36)
    bank_regex = {"$regex": re.escape(account_no), "$options": "i"}
    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        f"data.{FIELD_PAYMENT_DIRECTION}": direction,
        f"data.{FIELD_DATE}": {"$gte": start_dt, "$lte": end_dt},
        "$or": [
            {f"data.{FIELD_TRX_BANK}": bank_regex},
            {f"data.{FIELD_SENDER_ACCOUNT}": {"$in": [account_no, float(account_no)]}},
            {f"data.{FIELD_BENEFICIARY_ACCOUNT}": {"$in": [account_no, float(account_no)]}},
        ],
    }
    cursor = prop_db.formdatas.find(query, {"_id": 1, "data": 1}).limit(50)
    async for candidate in cursor:
        raw = (candidate.get("data") or {}).get(FIELD_AMOUNT)
        try:
            if abs(float(raw) - amount) < 0.005:
                return candidate
        except (TypeError, ValueError):
            continue
    return None


async def find_prior_by_iban_or_name(
    prop_db,
    *,
    iban: str | None,
    bank_statement_name: str,
) -> dict[str, Any] | None:
    or_conditions: list[dict[str, Any]] = []
    if iban:
        or_conditions.append(
            {f"data.{FIELD_NOTES}": {"$regex": re.escape(iban), "$options": "i"}}
        )
    if bank_statement_name and len(bank_statement_name) >= 4:
        or_conditions.append(
            {
                f"data.{FIELD_TRX_BANK_STATEMENT}": {
                    "$regex": f"^{re.escape(bank_statement_name)}$",
                    "$options": "i",
                }
            }
        )
    if not or_conditions:
        return None
    doc = await prop_db.formdatas.find_one(
        {
            "indicator": CASHFLOW_INDICATOR,
            "status": "active",
            f"data.{FIELD_DOCUMENT_TYPE}": "Bank Transaction",
            f"data.{FIELD_CATEGORY}": "Payment",
            "$or": or_conditions,
        },
        sort=[(f"data.{FIELD_DATE}", -1)],
    )
    return doc


async def find_property_by_text(prop_db, text: str) -> dict[str, Any] | None:
    # Look for street+number patterns
    candidates = re.findall(
        r"([A-Za-zΑ-Ωα-ωΆ-ώ][A-Za-zΑ-Ωα-ωΆ-ώ\s]{2,40}\s+\d+[A-Za-z]?)",
        text or "",
    )
    for cand in candidates[:5]:
        q = cand.strip()
        if len(q) < 5:
            continue
        doc = await prop_db.formdatas.find_one(
            {
                "indicator": "properties",
                "status": "active",
                f"data.{PROPERTY_TITLE_FIELD}": {
                    "$regex": re.escape(q),
                    "$options": "i",
                },
            },
            {"pid": 1, f"data.{PROPERTY_TITLE_FIELD}": 1},
        )
        if doc:
            pid = doc.get("pid")
            pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
            label = ((doc.get("data") or {}).get(PROPERTY_TITLE_FIELD) or "").strip()
            return {"pid": pid_str, "label": label or pid_str}
    return None


async def find_contact_by_name(prop_db, name: str) -> dict[str, Any] | None:
    cleaned = (name or "").strip()
    if len(cleaned) < 3:
        return None
    # Try latin/greek fragments
    results = await search_contacts(prop_db, cleaned[:40], limit=5)
    if results:
        return results[0]
    # First two tokens
    tokens = cleaned.split()
    if len(tokens) >= 2:
        results = await search_contacts(prop_db, " ".join(tokens[:2]), limit=5)
        if results:
            return results[0]
    return None


async def find_by_rf_code(prop_db, rf: str) -> dict[str, Any] | None:
    if not rf:
        return None
    return await prop_db.formdatas.find_one(
        {
            "indicator": CASHFLOW_INDICATOR,
            "status": "active",
            "$or": [
                {f"data.{FIELD_RF_PAYMENT_CODE}": {"$regex": re.escape(rf), "$options": "i"}},
                {f"data.{FIELD_NOTES}": {"$regex": re.escape(rf), "$options": "i"}},
                {f"data.{FIELD_TRX_REF_NO}": {"$regex": re.escape(rf), "$options": "i"}},
            ],
        },
        sort=[(f"data.{FIELD_DATE}", -1)],
    )


async def owner_for_property(
    prop_db,
    property_piped: str,
    holder_piped_value: str,
) -> str:
    if not property_piped:
        return holder_piped_value
    if extract_piped_id(property_piped) == extract_piped_id(AUX_PROPERTY):
        return holder_piped_value
    # Prefer owner from a recent cashflow on this property
    prop_id = extract_piped_id(property_piped)
    if not prop_id:
        return holder_piped_value
    doc = await prop_db.formdatas.find_one(
        {
            "indicator": CASHFLOW_INDICATOR,
            "status": "active",
            f"data.{FIELD_PROPERTY}": {"$regex": re.escape(prop_id)},
            f"data.{FIELD_CASHFLOW_CONTACT}": {"$exists": True, "$ne": ""},
        },
        sort=[(f"data.{FIELD_DATE}", -1)],
        projection={f"data.{FIELD_CASHFLOW_CONTACT}": 1},
    )
    if doc:
        owner = (doc.get("data") or {}).get(FIELD_CASHFLOW_CONTACT)
        if owner:
            return owner
    return holder_piped_value


def our_account_numbers(accounts: list[dict[str, Any]]) -> set[str]:
    return {normalize_account_no(a["accountNo"]) for a in accounts}


def detect_internal_transfer(text: str, our_accounts: set[str]) -> bool:
    digits = re.findall(r"\d{10,}", text or "")
    for d in digits:
        if normalize_account_no(d) in our_accounts:
            return True
    iban = extract_iban(text or "")
    if iban:
        for acct in our_accounts:
            if acct and acct in iban:
                return True
    return False


async def enrich_line(
    prop_db,
    line: dict[str, Any],
    account: dict[str, Any],
    *,
    payees: list[dict[str, Any]],
    all_accounts: list[dict[str, Any]],
) -> dict[str, Any]:
    holder = holder_piped(account)
    short = line.get("shortDesc") or ""
    details = line.get("details") or ""
    statement = details or short
    iban = extract_iban(statement)
    rf = extract_rf_code(statement)
    bank_stmt_name = extract_counterparty_name(details, short)
    our_accts = our_account_numbers(all_accounts)
    internal = detect_internal_transfer(statement, our_accts - {account["accountNo"]})

    row: dict[str, Any] = {
        **line,
        "status": "new",
        "rule": "",
        "confirmed": False,
        "internalTransfer": internal,
        "trxDescription": "",
        "trxBeneficiary": "",
        "trxReceiver": "",
        "propertyId": "",
        "propertyOwner": "",
        "trxBankStatement": bank_stmt_name,
        "investGreeceNotes": statement,
        "existingCashflowId": None,
        "existingCashflowUrl": None,
        "suggestionComplete": False,
    }

    # Fees
    if line.get("isFee"):
        row.update(
            {
                "status": "fee",
                "rule": "fee_template",
                "trxDescription": "Bank Fees",
                "trxBeneficiary": holder,
                "trxReceiver": FEE_RECEIVER,
                "propertyId": AUX_PROPERTY,
                "propertyOwner": holder,
                "trxBankStatement": "eksoda metaforas",
                "investGreeceNotes": "trapezika eksoda",
                "suggestionComplete": True,
                "confirmed": True,
            }
        )
        # Still run duplicate check below
    else:
        # Personal rules
        upper_stmt = statement.upper()
        if account.get("key") == "personal" or "personal" in (account.get("trxBank") or "").lower():
            if "PERSONAL FUNDING" in upper_stmt or internal:
                row["trxDescription"] = "Transfers Between Accounts"
                row["propertyId"] = AUX_PROPERTY
                row["propertyOwner"] = holder
                row["rule"] = "personal_funding"
                if line["paymentDirection"] == "Incoming Payment":
                    row["trxReceiver"] = holder
                else:
                    row["trxBeneficiary"] = holder
            elif short.upper().startswith("TCR") or "ΚΑΤΑΘΕΣΗ" in short.upper():
                row["trxDescription"] = "Cash Flow"
                row["propertyId"] = AUX_PROPERTY
                row["propertyOwner"] = holder
                row["trxBeneficiary"] = holder
                row["trxReceiver"] = holder
                row["rule"] = "cash_deposit"

        # Known payees
        if not row["rule"]:
            hay = f"{short} {details}".upper()
            for payee in payees:
                match = (payee.get("match") or "").upper()
                if match and match in hay:
                    row["trxDescription"] = map_legacy_category(payee.get("category")) or payee.get(
                        "category"
                    )
                    recv_label = payee.get("receiverLabel") or payee.get("match")
                    recv_pid = payee.get("receiverPid") or ""
                    if recv_pid:
                        row["trxReceiver"] = _build_piped(recv_label, recv_pid)
                    else:
                        contact = await find_contact_by_name(prop_db, recv_label)
                        if contact:
                            row["trxReceiver"] = _build_piped(contact["label"], contact["pid"])
                        else:
                            row["trxReceiver"] = ""
                    row["trxBeneficiary"] = holder
                    row["propertyId"] = AUX_PROPERTY
                    row["propertyOwner"] = holder
                    row["rule"] = f"known_payee:{payee.get('match')}"
                    break

        # Prior counterparty
        if not row["rule"]:
            prior = await find_prior_by_iban_or_name(
                prop_db, iban=iban, bank_statement_name=bank_stmt_name
            )
            if prior:
                pdata = prior.get("data") or {}
                row["trxDescription"] = map_legacy_category(
                    pdata.get(FIELD_DESCRIPTION)
                ) or (pdata.get(FIELD_DESCRIPTION) or "")
                row["trxBeneficiary"] = pdata.get(FIELD_TRX_BENEFICIARY) or ""
                row["trxReceiver"] = pdata.get(FIELD_TRX_RECEIVER) or ""
                row["propertyId"] = pdata.get(FIELD_PROPERTY) or ""
                row["propertyOwner"] = pdata.get(FIELD_CASHFLOW_CONTACT) or ""
                row["rule"] = "prior_counterparty"

        # Address / tenant
        if not row["rule"] or (
            row["rule"].startswith("personal") and not row.get("propertyId")
        ):
            prop = await find_property_by_text(prop_db, statement)
            if prop:
                row["propertyId"] = _build_piped(prop["label"], prop["pid"])
                row["propertyOwner"] = await owner_for_property(
                    prop_db, row["propertyId"], holder
                )
                if not row["trxDescription"]:
                    # Heuristic: incoming with property → rent
                    if line["paymentDirection"] == "Incoming Payment":
                        row["trxDescription"] = "Rent Payment"
                    row["rule"] = row["rule"] or "address_or_tenant"
                # Try tenant/sender name as contact
                name_guess = extract_counterparty_name(details, short)
                contact = await find_contact_by_name(prop_db, name_guess)
                if contact:
                    piped = _build_piped(contact["label"], contact["pid"])
                    if line["paymentDirection"] == "Incoming Payment":
                        row["trxBeneficiary"] = piped
                        row["trxReceiver"] = holder
                    else:
                        row["trxBeneficiary"] = holder
                        row["trxReceiver"] = piped
                elif line["paymentDirection"] == "Incoming Payment":
                    row["trxReceiver"] = holder
                if not row["rule"]:
                    row["rule"] = "address_or_tenant"

        # RF code
        if (not row["rule"] or row["status"] == "new") and rf and not row.get("propertyId"):
            prior = await find_by_rf_code(prop_db, rf)
            if prior:
                pdata = prior.get("data") or {}
                if not row["trxDescription"]:
                    row["trxDescription"] = map_legacy_category(
                        pdata.get(FIELD_DESCRIPTION)
                    ) or (pdata.get(FIELD_DESCRIPTION) or "")
                if not row["propertyId"]:
                    row["propertyId"] = pdata.get(FIELD_PROPERTY) or ""
                if not row["propertyOwner"]:
                    row["propertyOwner"] = pdata.get(FIELD_CASHFLOW_CONTACT) or ""
                if not row["trxBeneficiary"]:
                    row["trxBeneficiary"] = pdata.get(FIELD_TRX_BENEFICIARY) or ""
                if not row["trxReceiver"]:
                    row["trxReceiver"] = pdata.get(FIELD_TRX_RECEIVER) or ""
                row["rule"] = row["rule"] or "rf_code"

        # Direction defaults for contacts
        if line["paymentDirection"] == "Incoming Payment":
            if not row["trxReceiver"]:
                row["trxReceiver"] = holder
        else:
            if not row["trxBeneficiary"]:
                row["trxBeneficiary"] = holder

        if not row["rule"]:
            row["rule"] = "to_choose"
            row["status"] = "to_choose"

        # Fill property owner if property set
        if row.get("propertyId") and not row.get("propertyOwner"):
            row["propertyOwner"] = await owner_for_property(
                prop_db, row["propertyId"], holder
            )

        # Internal transfer category
        if internal and row.get("trxDescription") not in ("Transfers Between Accounts", "Bank Fees"):
            if not row["trxDescription"]:
                row["trxDescription"] = "Transfers Between Accounts"
                row["propertyId"] = row["propertyId"] or AUX_PROPERTY
                row["propertyOwner"] = row["propertyOwner"] or holder
                row["rule"] = "internal_transfer"

        judgement_ok = all(
            [
                row.get("trxDescription"),
                row.get("trxBeneficiary"),
                row.get("trxReceiver"),
                row.get("propertyId"),
                row.get("propertyOwner"),
            ]
        )
        row["suggestionComplete"] = bool(judgement_ok)
        if judgement_ok and row["status"] == "new" and row["rule"] not in ("", "to_choose"):
            pass
        elif not judgement_ok and row["status"] not in ("fee", "skip"):
            row["status"] = "to_choose"

    # Duplicate checks
    exact = await find_exact_duplicate(
        prop_db,
        trx_bank=account["trxBank"],
        ref_no=line.get("refNo") or "",
        amount=float(line["amount"]),
        direction=line["paymentDirection"],
    )
    if exact:
        cid = str(exact["_id"])
        row["status"] = "skip"
        row["existingCashflowId"] = cid
        row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cid)
        row["rule"] = "duplicate_exact"
        return row

    # Also match against Olga-style bank texts containing account number
    if line.get("refNo"):
        bank_regex_doc = await prop_db.formdatas.find_one(
            {
                "indicator": CASHFLOW_INDICATOR,
                "status": "active",
                f"data.{FIELD_TRX_REF_NO}": line["refNo"],
                f"data.{FIELD_PAYMENT_DIRECTION}": line["paymentDirection"],
                f"data.{FIELD_TRX_BANK}": {
                    "$regex": re.escape(account["accountNo"]),
                    "$options": "i",
                },
            },
            {"_id": 1},
        )
        if bank_regex_doc:
            amt = None
            # verify amount via second query fields — reload
            full = await prop_db.formdatas.find_one(
                {"_id": bank_regex_doc["_id"]}, {f"data.{FIELD_AMOUNT}": 1}
            )
            try:
                amt = float((full.get("data") or {}).get(FIELD_AMOUNT))
            except (TypeError, ValueError, AttributeError):
                amt = None
            if amt is not None and abs(amt - float(line["amount"])) < 0.005:
                cid = str(bank_regex_doc["_id"])
                row["status"] = "skip"
                row["existingCashflowId"] = cid
                row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(
                    cashflow_id=cid
                )
                row["rule"] = "duplicate_exact_alt_bank"
                return row

    possible = await find_possible_duplicate(
        prop_db,
        account_no=account["accountNo"],
        amount=float(line["amount"]),
        direction=line["paymentDirection"],
        trx_date=date.fromisoformat(line["trxDate"][:10]),
    )
    if possible and row["status"] not in ("fee",):
        # Only flag as possible double when ref missing or weak
        if not line.get("refNo"):
            cid = str(possible["_id"])
            row["status"] = "possible_double"
            row["existingCashflowId"] = cid
            row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cid)
            row["rule"] = "possible_double"
        elif row["status"] in ("new", "to_choose"):
            cid = str(possible["_id"])
            row["status"] = "possible_double"
            row["existingCashflowId"] = cid
            row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cid)
            if not row.get("rule") or row["rule"] == "to_choose":
                row["rule"] = "possible_double"

    return row


def judgement_fields_complete(row: dict[str, Any]) -> bool:
    return all(
        [
            (row.get("trxDescription") or "").strip(),
            (row.get("trxBeneficiary") or "").strip(),
            (row.get("trxReceiver") or "").strip(),
            (row.get("propertyId") or "").strip(),
            (row.get("propertyOwner") or "").strip(),
        ]
    )


def refresh_judgement_status(row: dict[str, Any]) -> None:
    """Keep new/to_choose in sync with whether judgement fields are filled."""
    status = row.get("status")
    if status in ("fee", "skip", "saved", "possible_double", "ignored"):
        return
    if judgement_fields_complete(row):
        row["status"] = "new"
        row["suggestionComplete"] = True
    else:
        row["status"] = "to_choose"
        row["suggestionComplete"] = False
        row["confirmed"] = False


async def create_import_session(
    console_db,
    prop_db,
    *,
    account: dict[str, Any],
    filename: str,
    file_bytes: bytes,
    saved_by: str = "",
) -> dict[str, Any]:
    parsed = parse_optima_workbook(file_bytes)
    file_account = normalize_account_no(parsed["accountNo"])
    if file_account != account["accountNo"]:
        raise ValueError(
            f"File account {file_account or '(missing)'} does not match "
            f"selected account {account['accountNo']}."
        )

    await ensure_default_payees(console_db)
    payees = await list_payees(console_db)
    all_accounts = await list_accounts(console_db)

    rows: list[dict[str, Any]] = []
    for idx, line in enumerate(parsed["lines"]):
        enriched = await enrich_line(
            prop_db,
            line,
            account,
            payees=payees,
            all_accounts=all_accounts,
        )
        enriched["index"] = idx
        rows.append(enriched)

    # Sort by date ascending for review
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("trxDate") or "", r.get("index", 0)),
    )

    now = datetime.now(timezone.utc)
    doc = {
        "status": "open",
        "filename": filename,
        "accountId": account["id"],
        "accountKey": account.get("key"),
        "accountLabel": account.get("label"),
        "trxBank": account["trxBank"],
        "accountNo": account["accountNo"],
        "holderLabel": account["holderLabel"],
        "holderPid": account["holderPid"],
        "period": parsed.get("period") or "",
        "closingBalance": parsed.get("closingBalance"),
        "integrity": parsed.get("integrity") or {},
        "rows": rows_sorted,
        "savedBy": saved_by or "",
        "createdAt": now,
        "updatedAt": now,
        "lastSaveAt": None,
        "saveStats": None,
    }
    result = await console_db[IMPORTS_COLLECTION].insert_one(doc)
    doc["_id"] = result.inserted_id
    return serialize_import(doc)


def serialize_import(doc: dict[str, Any], *, include_rows: bool = True) -> dict[str, Any]:
    out = {
        "id": str(doc["_id"]),
        "status": doc.get("status") or "open",
        "filename": doc.get("filename") or "",
        "accountId": doc.get("accountId") or "",
        "accountKey": doc.get("accountKey") or "",
        "accountLabel": doc.get("accountLabel") or "",
        "trxBank": doc.get("trxBank") or "",
        "accountNo": doc.get("accountNo") or "",
        "holderLabel": doc.get("holderLabel") or "",
        "holderPid": str(doc.get("holderPid") or ""),
        "period": doc.get("period") or "",
        "closingBalance": doc.get("closingBalance"),
        "integrity": doc.get("integrity") or {},
        "savedBy": doc.get("savedBy") or "",
        "createdAt": _iso(doc.get("createdAt")),
        "updatedAt": _iso(doc.get("updatedAt")),
        "lastSaveAt": _iso(doc.get("lastSaveAt")),
        "saveStats": doc.get("saveStats"),
        "rowCount": len(doc.get("rows") or []),
    }
    if include_rows:
        out["rows"] = doc.get("rows") or []
        out["counts"] = summarize_row_statuses(out["rows"])
    return out


def summarize_row_statuses(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {
        "new": 0,
        "fee": 0,
        "skip": 0,
        "possible_double": 0,
        "to_choose": 0,
        "saved": 0,
        "ignored": 0,
        "total": len(rows),
    }
    for row in rows:
        status = row.get("status") or "new"
        counts[status] = counts.get(status, 0) + 1
    return counts


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.isoformat()
    return str(value)


async def get_import(console_db, import_id: str) -> dict[str, Any] | None:
    doc = await console_db[IMPORTS_COLLECTION].find_one({"_id": ObjectId(import_id)})
    if not doc:
        return None
    return serialize_import(doc)


async def list_imports(console_db, limit: int = 30) -> list[dict[str, Any]]:
    cursor = console_db[IMPORTS_COLLECTION].find({}).sort("createdAt", -1).limit(limit)
    out = []
    async for doc in cursor:
        out.append(serialize_import(doc, include_rows=False))
    return out


async def delete_import(console_db, import_id: str) -> bool:
    """Delete an import session draft. Does not delete Prop360 cashflows already saved."""
    result = await console_db[IMPORTS_COLLECTION].delete_one({"_id": ObjectId(import_id)})
    return result.deleted_count > 0


async def patch_import_rows(
    console_db,
    prop_db,
    import_id: str,
    *,
    rows_patch: list[dict[str, Any]] | None = None,
    confirm_all_suggested: bool = False,
) -> dict[str, Any]:
    col = console_db[IMPORTS_COLLECTION]
    doc = await col.find_one({"_id": ObjectId(import_id)})
    if not doc:
        raise ValueError("Import not found.")
    rows = list(doc.get("rows") or [])
    by_index = {int(r.get("index", i)): i for i, r in enumerate(rows)}
    holder = _build_piped(doc.get("holderLabel") or "", doc.get("holderPid") or "")

    if confirm_all_suggested:
        for row in rows:
            if row.get("status") in ("new", "fee") and judgement_fields_complete(row):
                row["confirmed"] = True
                if row.get("status") == "to_choose" and judgement_fields_complete(row):
                    row["status"] = "new"

    if rows_patch:
        for patch in rows_patch:
            idx = int(patch.get("index"))
            pos = by_index.get(idx)
            if pos is None:
                continue
            row = rows[pos]
            for key in (
                "trxDescription",
                "trxBeneficiary",
                "trxReceiver",
                "propertyId",
                "propertyOwner",
                "trxBankStatement",
                "investGreeceNotes",
                "confirmed",
                "status",
            ):
                if key in patch:
                    row[key] = patch[key]
            if "propertyId" in patch and patch.get("propertyId"):
                row["propertyOwner"] = await owner_for_property(
                    prop_db, row["propertyId"], holder
                )
            if patch.get("ignore"):
                row["status"] = "ignored"
                row["confirmed"] = False
                row["previousStatus"] = patch.get("previousStatus") or row.get(
                    "previousStatus"
                )
            elif patch.get("restoreIgnored"):
                # Bring back into the review queue
                row["status"] = "to_choose"
                row.pop("previousStatus", None)
                refresh_judgement_status(row)
            elif row.get("status") == "possible_double" and patch.get("confirmAsNew"):
                row["status"] = "new"
                row["confirmed"] = True
                refresh_judgement_status(row)
            elif row.get("status") != "ignored":
                # Upgrade/downgrade new <-> to_choose when judgement fields change
                refresh_judgement_status(row)
            rows[pos] = row

    now = datetime.now(timezone.utc)
    await col.update_one(
        {"_id": doc["_id"]},
        {"$set": {"rows": rows, "updatedAt": now}},
    )
    updated = await col.find_one({"_id": doc["_id"]})
    return serialize_import(updated)


async def save_import(
    console_db,
    prop_db,
    import_id: str,
    *,
    saved_by: str = "",
) -> dict[str, Any]:
    col = console_db[IMPORTS_COLLECTION]
    doc = await col.find_one({"_id": ObjectId(import_id)})
    if not doc:
        raise ValueError("Import not found.")

    account = {
        "id": doc.get("accountId"),
        "key": doc.get("accountKey"),
        "label": doc.get("accountLabel"),
        "trxBank": doc.get("trxBank"),
        "accountNo": doc.get("accountNo"),
        "holderLabel": doc.get("holderLabel"),
        "holderPid": doc.get("holderPid"),
    }

    rows = list(doc.get("rows") or [])
    created = 0
    skipped = 0
    left_open = 0
    created_ids: list[str] = []

    for i, row in enumerate(rows):
        status = row.get("status")
        if status in ("skip", "saved", "ignored"):
            skipped += 1
            continue
        if status not in ("new", "fee"):
            left_open += 1
            continue
        if not judgement_fields_complete(row):
            left_open += 1
            continue

        # Re-check exact duplicate before insert
        exact = await find_exact_duplicate(
            prop_db,
            trx_bank=account["trxBank"],
            ref_no=row.get("refNo") or "",
            amount=float(row["amount"]),
            direction=row["paymentDirection"],
        )
        if exact:
            cid = str(exact["_id"])
            row["status"] = "skip"
            row["existingCashflowId"] = cid
            row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cid)
            skipped += 1
            rows[i] = row
            continue

        data = build_cashflow_data_from_row(row, account)
        cashflow_id = await create_bank_import_cashflow(
            prop_db,
            data,
            import_id=import_id,
            row_index=int(row.get("index", i)),
        )
        row["status"] = "saved"
        row["existingCashflowId"] = cashflow_id
        row["existingCashflowUrl"] = CASHFLOW_FORM_URL_TEMPLATE.format(
            cashflow_id=cashflow_id
        )
        created += 1
        created_ids.append(cashflow_id)
        rows[i] = row

    stats = {
        "created": created,
        "skipped": skipped,
        "leftOpen": left_open,
        "createdIds": created_ids,
    }
    now = datetime.now(timezone.utc)
    still_open = any(
        r.get("status") in ("new", "fee", "to_choose", "possible_double") for r in rows
    )
    await col.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "rows": rows,
                "updatedAt": now,
                "lastSaveAt": now,
                "saveStats": stats,
                "savedBy": saved_by or doc.get("savedBy") or "",
                "status": "open" if still_open else "completed",
            }
        },
    )
    updated = await col.find_one({"_id": doc["_id"]})
    result = serialize_import(updated)
    result["saveResult"] = stats
    return result


async def create_simple_contact(
    prop_db,
    *,
    name: str,
) -> dict[str, Any]:
    """Create a minimal Prop360 contact for use as beneficiary/receiver."""
    owner = _prop360_owner_id()
    now = datetime.now(timezone.utc)
    pid = _new_cashflow_pid()
    label = (name or "").strip()
    if len(label) < 2:
        raise ValueError("Contact name is required.")

    existing = await find_contact_by_name(prop_db, label)
    if existing and existing["label"].casefold() == label.casefold():
        return existing

    doc_id = ObjectId()
    doc: dict[str, Any] = {
        "_id": doc_id,
        "indicator": "contacts",
        "isPublic": False,
        "data": {
            CONTACT_NAME_FIELD: label,
            "attachedForms": [],
        },
        "metadata": {
            "createdAt": now,
            "updatedAt": now,
            "createdBy": owner,
            "updatedBy": owner,
            "consoleBankImportContact": True,
        },
        "owner": owner,
        "merchantId": KOSTAS_MERCHANT_ID,
        "sharedWithMerchants": [KOSTAS_MERCHANT_ID],
        "sharedWith": [],
        "status": "active",
        "destructive": False,
        "pid": pid,
        "isWorkflowForm": False,
        "__v": 0,
    }
    form = await prop_db.forms.find_one({"indicator": "contacts"}, {"_id": 1})
    if form:
        doc["formId"] = str(form["_id"])
    await prop_db.formdatas.insert_one(doc)
    pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
    return {"pid": pid_str, "label": label, "mongoId": str(doc_id)}
