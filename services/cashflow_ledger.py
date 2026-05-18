import io
import re
from datetime import datetime
from typing import Any

import pandas as pd

CASHFLOW_INDICATOR = "custom-a462rgbzo"

FIELD_PROPERTY = "field-1757605637506-70de9chi5"
FIELD_DATE = "field-1757605069078-p5plna7qr"
FIELD_DESCRIPTION = "field-1757605219384-fikyy3d7u"
FIELD_AMOUNT = "field-1757605508754-uea4iadqd"
FIELD_DEBIT_CREDIT = "field-1757605718340-ue95ozr9u"


def normalize_property_field(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_property_id(property_field: Any) -> str | None:
    text = normalize_property_field(property_field)
    if not text:
        return None
    if "|" in text:
        prop_id = text.rsplit("|", 1)[-1].strip()
        return prop_id or None
    return text


def _clean_property_name_display(name: str) -> str:
    """Normalize whitespace and literal \\n from property label text."""
    cleaned = name.replace("\\n", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(cleaned.split()).strip()


def extract_property_name(
    property_field: Any, property_id: str | None = None
) -> str | None:
    text = normalize_property_field(property_field)
    if not text:
        return None

    if property_id:
        pid = str(property_id).strip()
        for separator in ("|", "｜", "¦"):
            marker = f"{separator}{pid}"
            idx = text.rfind(marker)
            if idx != -1:
                name = _clean_property_name_display(text[:idx])
                if name:
                    return name

        match = re.search(
            rf"[\|｜¦]\s*{re.escape(pid)}\s*$",
            text,
            flags=re.DOTALL,
        )
        if match:
            name = _clean_property_name_display(text[: match.start()])
            if name:
                return name

    if "|" in text:
        name = _clean_property_name_display(text.rsplit("|", 1)[0])
        if not name:
            return None
        if property_id and name.replace(" ", "") == str(property_id).replace(" ", ""):
            return None
        return name

    if property_id and text.replace(" ", "") == str(property_id).replace(" ", ""):
        return None
    if text.isdigit():
        return None
    return _clean_property_name_display(text) or None


def property_field_regex(property_id: str) -> str:
    escaped = re.escape(property_id)
    return rf"(?:\s|\\n|\r|\n)*\|\s*{escaped}\s*$"


def _pick_longer_property_name(current: str, candidate: str | None) -> str:
    if not candidate:
        return current
    if len(candidate) > len(current):
        return candidate
    return current


PROPERTY_TITLE_FIELD = "field-1741536181001-wd8it2quy"


async def _collect_property_name_from_docs(
    cursor, property_id: str, best: str
) -> str:
    async for doc in cursor:
        data = doc.get("data") or {}
        name = extract_property_name(data.get(FIELD_PROPERTY), property_id)
        best = _pick_longer_property_name(best, name)

        if not best:
            title = normalize_property_field(data.get(PROPERTY_TITLE_FIELD))
            if title and not title.isdigit():
                best = _pick_longer_property_name(
                    best, _clean_property_name_display(title.split("-")[0])
                )
    return best


async def lookup_property_name(prop_db, property_id: str) -> str:
    """Find the longest property name from any formdata referencing this property id."""
    field_regex = {f"data.{FIELD_PROPERTY}": {"$regex": property_field_regex(property_id)}}
    best = ""

    for query in (
        {"status": "active", **field_regex},
        field_regex,
    ):
        cursor = prop_db.formdatas.find(
            query, {f"data.{FIELD_PROPERTY}": 1, f"data.{PROPERTY_TITLE_FIELD}": 1}
        ).sort("_id", -1).limit(200)
        best = await _collect_property_name_from_docs(cursor, property_id, best)
        if best:
            return best

    try:
        pid_float = float(property_id)
        for indicator in ("properties", CASHFLOW_INDICATOR):
            doc = await prop_db.formdatas.find_one(
                {"pid": pid_float, "indicator": indicator},
                {f"data.{FIELD_PROPERTY}": 1, f"data.{PROPERTY_TITLE_FIELD}": 1},
            )
            if not doc:
                continue
            data = doc.get("data") or {}
            best = _pick_longer_property_name(
                best,
                extract_property_name(data.get(FIELD_PROPERTY), property_id),
            )
            title = normalize_property_field(data.get(PROPERTY_TITLE_FIELD))
            if title and not title.isdigit():
                best = _pick_longer_property_name(
                    best, _clean_property_name_display(title.split("-")[0])
                )
            if best:
                return best
    except ValueError:
        pass

    # Last resort: any document whose property field contains this id
    cursor = prop_db.formdatas.find(
        {f"data.{FIELD_PROPERTY}": {"$regex": re.escape(property_id)}},
        {f"data.{FIELD_PROPERTY}": 1},
    ).sort("_id", -1).limit(200)
    return await _collect_property_name_from_docs(cursor, property_id, best)


def parse_amount(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def format_date(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text:
        return ""
    try:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).strftime("%Y-%m-%d")
    except ValueError:
        return text[:10] if len(text) >= 10 else text


def normalize_debit_credit(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"debit", "dr", "d"}:
        return "debit"
    if text in {"credit", "cr", "c"}:
        return "credit"
    if "debit" in text:
        return "debit"
    if "credit" in text:
        return "credit"
    return None


def parse_cashflow_document(doc: dict) -> dict | None:
    data = doc.get("data") or {}
    entry_type = normalize_debit_credit(data.get(FIELD_DEBIT_CREDIT))
    if not entry_type:
        return None

    return {
        "date": format_date(data.get(FIELD_DATE)),
        "description": str(data.get(FIELD_DESCRIPTION) or "").strip(),
        "amount": parse_amount(data.get(FIELD_AMOUNT)),
        "type": entry_type,
        "_sort_date": data.get(FIELD_DATE) or "",
    }


def build_ledger(transactions: list[dict]) -> dict:
    debit_rows = []
    credit_rows = []

    for tx in sorted(transactions, key=lambda row: row.get("_sort_date") or ""):
        row = {
            "date": tx["date"],
            "description": tx["description"],
            "amount": tx["amount"],
        }
        if tx["type"] == "debit":
            debit_rows.append(row)
        else:
            credit_rows.append(row)

    debit_sum = round(sum(row["amount"] for row in debit_rows), 2)
    credit_sum = round(sum(row["amount"] for row in credit_rows), 2)

    return {
        "propertyId": None,
        "propertyName": None,
        "debit": {
            "transactions": debit_rows,
            "sum": debit_sum,
        },
        "credit": {
            "transactions": credit_rows,
            "sum": credit_sum,
        },
    }


async def fetch_ledger_for_property(prop_db, property_id: str) -> dict:
    property_id = str(property_id).strip()
    if not property_id:
        raise ValueError("propertyId is required")

    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        f"data.{FIELD_PROPERTY}": {"$regex": property_field_regex(property_id)},
    }

    cursor = prop_db.formdatas.find(query).sort("_id", -1)
    transactions = []
    property_name = ""

    async for doc in cursor:
        data = doc.get("data") or {}
        property_name = _pick_longer_property_name(
            property_name,
            extract_property_name(data.get(FIELD_PROPERTY), property_id),
        )

        parsed = parse_cashflow_document(doc)
        if parsed:
            transactions.append(parsed)

    resolved_name = await lookup_property_name(prop_db, property_id)
    property_name = _pick_longer_property_name(property_name, resolved_name)

    ledger = build_ledger(transactions)
    ledger["propertyId"] = property_id
    ledger["propertyName"] = property_name
    ledger["transactionCount"] = len(transactions)
    return ledger


def build_ledger_excel(ledger: dict) -> bytes:
    debit_rows = ledger["debit"]["transactions"]
    credit_rows = ledger["credit"]["transactions"]
    max_len = max(len(debit_rows), len(credit_rows), 0)

    property_id = ledger.get("propertyId") or ""
    property_name = ledger.get("propertyName") or ""

    rows: list[list[Any]] = [
        ["Property ID", property_id, "", "", "", "", ""],
        ["Property Name", property_name, "", "", "", "", ""],
        ["", "", "", "", "", "", ""],
        [
            "Date",
            "Transaction Description",
            "Amount",
            "",
            "Date",
            "Transaction Description",
            "Amount",
        ],
        ["DEBIT", "", "", "", "CREDIT", "", ""],
    ]

    for index in range(max_len):
        debit = debit_rows[index] if index < len(debit_rows) else {}
        credit = credit_rows[index] if index < len(credit_rows) else {}
        rows.append(
            [
                debit.get("date", ""),
                debit.get("description", ""),
                debit.get("amount", ""),
                "",
                credit.get("date", ""),
                credit.get("description", ""),
                credit.get("amount", ""),
            ]
        )

    rows.append(
        [
            "",
            "Total",
            ledger["debit"]["sum"],
            "",
            "",
            "Total",
            ledger["credit"]["sum"],
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, header=False, sheet_name="Ledger")
    buffer.seek(0)
    return buffer.getvalue()
