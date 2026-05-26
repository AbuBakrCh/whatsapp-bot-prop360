import io
import math
import re
from datetime import datetime, timezone
from typing import Any

DEFAULT_ACTIVITY_PAGE_SIZE = 10

import pandas as pd

CASHFLOW_INDICATOR = "custom-a462rgbzo"
CASHFLOW_FORM_URL_TEMPLATE = (
    "https://prop360.pro/en/dashboard/forms/custom-a462rgbzo/{cashflow_id}"
)

FIELD_PROPERTY = "field-1757605637506-70de9chi5"
FIELD_DATE = "field-1757605069078-p5plna7qr"
FIELD_DESCRIPTION = "field-1757605219384-fikyy3d7u"
FIELD_AMOUNT = "field-1757605508754-uea4iadqd"
FIELD_DEBIT_CREDIT = "field-1757605718340-ue95ozr9u"
FIELD_CASHFLOW_CONTACT = "field-1761124961032-67lj21506"

GROUP_TYPE_PROPERTY = "property"
GROUP_TYPE_CONTACT = "contact"

ACTIVITY_INDICATOR = "custom-wyey07pb7"
ACTIVITY_FORM_URL_TEMPLATE = (
    "https://prop360.pro/en/dashboard/forms/custom-wyey07pb7/{activity_id}"
)
FIELD_ACTIVITY_DATE = "field-1760213127501-vd61epis6"
FIELD_ACTIVITY_DESCRIPTION = "field-1760213212062-ask5v2fuy"
ACTIVITY_PROPERTY_FIELDS = (
    "field-1760213192233-byk1fbajy",
    "field-1762112936496-lcg46gwiy",
    "field-1762112987608-45lv27qbc",
    "field-1764147281268-oqtfditkd",
    "field-1764147283488-svx61v7j3",
    "field-1764147285842-qbxk0iz1e",
)
ACTIVITY_CLIENT_FIELDS = (
    "field-1760213170764-fhjgcg5u0",
    "field-1762112354057-0rwwvsbo0",
    "field-1762112414711-wp3hdmt1n",
    "field-1764147273289-bqudbub97",
    "field-1764147276663-da6q4ymmr",
    "field-1764147278883-5oxys6rmc",
)


def normalize_piped_field(value: Any) -> str | None:
    return normalize_property_field(value)


def extract_piped_id(piped_field: Any) -> str | None:
    return extract_property_id(piped_field)


def piped_field_regex(piped_id: str) -> str:
    return property_field_regex(piped_id)


def extract_piped_label(piped_field: Any, piped_id: str | None = None) -> str | None:
    return extract_property_name(piped_field, piped_id)


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
        return None

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


def parse_sort_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_date(value: Any) -> str:
    parsed = parse_sort_datetime(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d")
    text = str(value).strip() if value is not None else ""
    return text[:10] if len(text) >= 10 else text


def format_datetime(value: Any) -> str:
    parsed = parse_sort_datetime(value)
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M")
    return ""


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


def doc_matches_property_id(data: dict, property_id: str) -> bool:
    pid = str(property_id).strip()
    for field in ACTIVITY_PROPERTY_FIELDS:
        if extract_piped_id(data.get(field)) == pid:
            return True
    return False


def doc_matches_contact_id(data: dict, contact_id: str) -> bool:
    cid = str(contact_id).strip()
    for field in ACTIVITY_CLIENT_FIELDS:
        if extract_piped_id(data.get(field)) == cid:
            return True
    return False


def doc_matches_group(data: dict, group_type: str, group_id: str) -> bool:
    if group_type == GROUP_TYPE_CONTACT:
        return doc_matches_contact_id(data, group_id)
    return doc_matches_property_id(data, group_id)


def parse_activity_document(doc: dict) -> dict | None:
    data = doc.get("data") or {}
    description = str(data.get(FIELD_ACTIVITY_DESCRIPTION) or "").strip()
    if not description:
        return None

    activity_date = data.get(FIELD_ACTIVITY_DATE)
    activity_id = str(doc.get("_id", ""))
    return {
        "id": activity_id,
        "date": format_datetime(activity_date),
        "description": description,
        "_sort_date": activity_date,
    }


def paginate_activities(
    activities: list[dict], page: int, page_size: int
) -> dict[str, Any]:
    total_count = len(activities)
    page_size = max(1, page_size)
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "rows": activities[start:end],
        "page": page,
        "pageSize": page_size,
        "totalCount": total_count,
        "totalPages": total_pages,
    }


async def fetch_activities(prop_db, group_type: str, group_id: str) -> list[dict]:
    group_id = str(group_id).strip()
    regex = piped_field_regex(group_id)
    if group_type == GROUP_TYPE_CONTACT:
        match_fields = ACTIVITY_CLIENT_FIELDS
    else:
        match_fields = ACTIVITY_PROPERTY_FIELDS

    query = {
        "indicator": ACTIVITY_INDICATOR,
        "status": "active",
        "$or": [{f"data.{field}": {"$regex": regex}} for field in match_fields],
    }

    cursor = prop_db.formdatas.find(query).sort("_id", -1)
    activities = []
    seen_ids: set[str] = set()

    async for doc in cursor:
        doc_id = str(doc.get("_id", ""))
        if doc_id in seen_ids:
            continue

        data = doc.get("data") or {}
        if not doc_matches_group(data, group_type, group_id):
            continue

        seen_ids.add(doc_id)

        parsed = parse_activity_document(doc)
        if parsed:
            activities.append(parsed)

    activities.sort(
        key=lambda row: parse_sort_datetime(row.get("_sort_date"))
        or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return [
        {
            "id": row.get("id", ""),
            "date": row["date"],
            "description": row["description"],
            "url": (
                ACTIVITY_FORM_URL_TEMPLATE.format(activity_id=row["id"])
                if row.get("id")
                else ""
            ),
        }
        for row in activities
    ]


def parse_cashflow_document(doc: dict) -> dict | None:
    data = doc.get("data") or {}
    entry_type = normalize_debit_credit(data.get(FIELD_DEBIT_CREDIT))
    if not entry_type:
        return None

    cashflow_date = data.get(FIELD_DATE)
    cashflow_id = str(doc.get("_id", ""))
    return {
        "id": cashflow_id,
        "date": format_datetime(cashflow_date),
        "description": str(data.get(FIELD_DESCRIPTION) or "").strip(),
        "amount": parse_amount(data.get(FIELD_AMOUNT)),
        "type": entry_type,
        "_sort_date": cashflow_date,
        "url": (
            CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cashflow_id)
            if cashflow_id
            else ""
        ),
    }


def _transaction_sort_key(row: dict) -> datetime:
    return parse_sort_datetime(row.get("_sort_date")) or datetime.min.replace(
        tzinfo=timezone.utc
    )


def build_ledger(transactions: list[dict]) -> dict:
    debit_rows = []
    credit_rows = []

    for tx in sorted(transactions, key=_transaction_sort_key, reverse=True):
        row = {
            "id": tx.get("id", ""),
            "date": tx["date"],
            "description": tx["description"],
            "amount": tx["amount"],
            "url": tx.get("url", ""),
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



async def _collect_contact_name_from_docs(cursor, contact_id: str, best: str) -> str:
    cid = str(contact_id).strip()
    async for doc in cursor:
        data = doc.get("data") or {}
        for field in ACTIVITY_CLIENT_FIELDS + (FIELD_CASHFLOW_CONTACT,):
            raw = data.get(field)
            if extract_piped_id(raw) != cid:
                continue
            name = extract_piped_label(raw, contact_id)
            best = _pick_longer_property_name(best, name)
    return best


async def lookup_contact_name(prop_db, contact_id: str) -> str:
    contact_id = str(contact_id).strip()
    regex = piped_field_regex(contact_id)
    best = ""
    projection = {f"data.{field}": 1 for field in ACTIVITY_CLIENT_FIELDS}
    projection[f"data.{FIELD_CASHFLOW_CONTACT}"] = 1

    for query in (
        {
            "status": "active",
            "indicator": ACTIVITY_INDICATOR,
            "$or": [{f"data.{field}": {"$regex": regex}} for field in ACTIVITY_CLIENT_FIELDS],
        },
        {
            "status": "active",
            "indicator": CASHFLOW_INDICATOR,
            f"data.{FIELD_CASHFLOW_CONTACT}": {"$regex": regex},
        },
    ):
        cursor = prop_db.formdatas.find(query, projection).sort("_id", -1).limit(200)
        best = await _collect_contact_name_from_docs(cursor, contact_id, best)
        if best:
            return best

    cursor = prop_db.formdatas.find(
        {
            "$or": [
                {f"data.{field}": {"$regex": re.escape(contact_id)}}
                for field in ACTIVITY_CLIENT_FIELDS
            ]
            + [{f"data.{FIELD_CASHFLOW_CONTACT}": {"$regex": re.escape(contact_id)}}]
        },
        projection,
    ).sort("_id", -1).limit(200)
    return await _collect_contact_name_from_docs(cursor, contact_id, best)


async def lookup_group_name(prop_db, group_type: str, group_id: str) -> str:
    if group_type == GROUP_TYPE_CONTACT:
        return await lookup_contact_name(prop_db, group_id)
    return await lookup_property_name(prop_db, group_id)


async def fetch_cashflow_transactions(
    prop_db, group_type: str, group_id: str
) -> tuple[list[dict], str]:
    group_id = str(group_id).strip()
    if group_type == GROUP_TYPE_CONTACT:
        field_key = f"data.{FIELD_CASHFLOW_CONTACT}"
    else:
        field_key = f"data.{FIELD_PROPERTY}"

    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        field_key: {"$regex": piped_field_regex(group_id)},
    }

    cursor = prop_db.formdatas.find(query).sort("_id", -1)
    transactions = []
    group_name = ""

    async for doc in cursor:
        data = doc.get("data") or {}
        if group_type == GROUP_TYPE_CONTACT:
            group_name = _pick_longer_property_name(
                group_name,
                extract_piped_label(data.get(FIELD_CASHFLOW_CONTACT), group_id),
            )
        else:
            group_name = _pick_longer_property_name(
                group_name,
                extract_piped_label(data.get(FIELD_PROPERTY), group_id),
            )

        if group_type == GROUP_TYPE_CONTACT:
            if extract_piped_id(data.get(FIELD_CASHFLOW_CONTACT)) != group_id:
                continue

        parsed = parse_cashflow_document(doc)
        if parsed:
            transactions.append(parsed)

    resolved_name = await lookup_group_name(prop_db, group_type, group_id)
    group_name = _pick_longer_property_name(group_name, resolved_name)
    return transactions, group_name


def _apply_group_metadata(ledger: dict, group_type: str, group_id: str, group_name: str) -> dict:
    ledger["groupType"] = group_type
    ledger["groupId"] = group_id
    ledger["groupName"] = group_name
    ledger["propertyId"] = group_id if group_type == GROUP_TYPE_PROPERTY else None
    ledger["propertyName"] = group_name if group_type == GROUP_TYPE_PROPERTY else None
    ledger["contactId"] = group_id if group_type == GROUP_TYPE_CONTACT else None
    ledger["contactName"] = group_name if group_type == GROUP_TYPE_CONTACT else None
    return ledger


async def fetch_ledger_report(
    prop_db,
    group_type: str,
    group_id: str,
    activity_page: int = 1,
    activity_page_size: int = DEFAULT_ACTIVITY_PAGE_SIZE,
    include_all_activities: bool = False,
) -> dict:
    group_type = str(group_type).strip().lower()
    group_id = str(group_id).strip()
    if group_type not in (GROUP_TYPE_PROPERTY, GROUP_TYPE_CONTACT):
        raise ValueError("groupType must be 'property' or 'contact'")
    if not group_id:
        raise ValueError("groupId is required")

    transactions, group_name = await fetch_cashflow_transactions(
        prop_db, group_type, group_id
    )
    all_activities = await fetch_activities(prop_db, group_type, group_id)

    ledger = build_ledger(transactions)
    ledger = _apply_group_metadata(ledger, group_type, group_id, group_name)
    ledger["transactionCount"] = len(transactions)
    ledger["activityCount"] = len(all_activities)
    if include_all_activities:
        ledger["activities"] = {
            "rows": all_activities,
            "page": 1,
            "pageSize": len(all_activities) or DEFAULT_ACTIVITY_PAGE_SIZE,
            "totalCount": len(all_activities),
            "totalPages": 1,
        }
    else:
        ledger["activities"] = paginate_activities(
            all_activities, activity_page, activity_page_size
        )
    return ledger


async def fetch_ledger_for_property(
    prop_db,
    property_id: str,
    activity_page: int = 1,
    activity_page_size: int = DEFAULT_ACTIVITY_PAGE_SIZE,
    include_all_activities: bool = False,
) -> dict:
    """Backward-compatible wrapper for property-scoped ledger."""
    return await fetch_ledger_report(
        prop_db,
        GROUP_TYPE_PROPERTY,
        property_id,
        activity_page=activity_page,
        activity_page_size=activity_page_size,
        include_all_activities=include_all_activities,
    )


def build_ledger_excel(ledger: dict) -> bytes:
    debit_rows = ledger["debit"]["transactions"]
    credit_rows = ledger["credit"]["transactions"]
    max_len = max(len(debit_rows), len(credit_rows), 0)

    group_type = ledger.get("groupType") or ""
    group_id = ledger.get("groupId") or ""
    group_name = ledger.get("groupName") or ""

    excel_width = 9
    rows: list[list[Any]] = [
        ["Group Type", group_type] + [""] * (excel_width - 2),
        ["Group ID", group_id] + [""] * (excel_width - 2),
        ["Group Name", group_name] + [""] * (excel_width - 2),
        [""] * excel_width,
        [
            "Date",
            "Link",
            "Transaction Description",
            "Amount",
            "",
            "Date",
            "Link",
            "Transaction Description",
            "Amount",
        ],
        ["DEBIT", "", "", "", "CREDIT", "", "", "", ""],
    ]

    for index in range(max_len):
        debit = debit_rows[index] if index < len(debit_rows) else {}
        credit = credit_rows[index] if index < len(credit_rows) else {}
        rows.append(
            [
                debit.get("date", ""),
                debit.get("url", ""),
                debit.get("description", ""),
                debit.get("amount", ""),
                "",
                credit.get("date", ""),
                credit.get("url", ""),
                credit.get("description", ""),
                credit.get("amount", ""),
            ]
        )

    rows.append(
        [
            "",
            "",
            "Total",
            ledger["debit"]["sum"],
            "",
            "",
            "",
            "Total",
            ledger["credit"]["sum"],
        ]
    )

    activity_rows = ledger.get("activities", {}).get("rows", [])
    rows.append([""] * excel_width)
    rows.append(["ACTIVITIES"] + [""] * (excel_width - 1))
    rows.append(["Date", "Link", "Activity Description"] + [""] * (excel_width - 3))
    for activity in activity_rows:
        rows.append(
            [
                activity.get("date", ""),
                activity.get("url", ""),
                activity.get("description", ""),
            ]
            + [""] * (excel_width - 3)
        )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, index=False, header=False, sheet_name="Ledger")
    buffer.seek(0)
    return buffer.getvalue()
