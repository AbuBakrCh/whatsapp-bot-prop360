"""Lookup and unshare Property/Contact/Timetable/Cashflow formdatas."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId

from services.ledger_report import (
    ACTIVITY_CLIENT_FIELDS,
    ACTIVITY_INDICATOR,
    ACTIVITY_PROPERTY_FIELDS,
    CASHFLOW_INDICATOR,
    FIELD_ACTIVITY_DESCRIPTION,
    FIELD_AMOUNT,
    FIELD_CASHFLOW_CONTACT,
    FIELD_DESCRIPTION,
    FIELD_PROPERTY,
    PROPERTY_TITLE_FIELD,
)
from services.recurrent_accruals import (
    CONTACT_EMAIL_FIELD,
    CONTACT_NAME_FIELD,
    CONTACT_PHONE_FIELD,
)
from services.sync_cashflow_refs import FIELD_TRX_RECEIVER

ENTITY_TYPE_INDICATORS = {
    "property": ("properties", "Property"),
    "contact": ("contacts", "Contact"),
    "timetable": (ACTIVITY_INDICATOR, "Timetable"),
    "cashflow": (CASHFLOW_INDICATOR, "Cashflow"),
}


def _pid_str(pid: Any) -> str | None:
    if pid is None:
        return None
    if isinstance(pid, float):
        return str(int(pid))
    return str(pid)


def _shared_count(doc: dict) -> int:
    merchants = doc.get("sharedWithMerchants") or []
    return len(merchants) if isinstance(merchants, list) else 0


def _truncate(text: str, max_len: int = 80) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def _parse_object_id(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def _parse_pid_value(pid_raw: Any) -> float | None:
    try:
        return float(pid_raw)
    except (TypeError, ValueError):
        return None


def _pid_or_conditions(q: str, escaped: str) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = [
        {
            "$expr": {
                "$regexMatch": {
                    "input": {"$toString": "$pid"},
                    "regex": escaped,
                    "options": "i",
                }
            }
        }
    ]
    if q.isdigit():
        try:
            conditions.append({"pid": float(q)})
        except ValueError:
            pass
        try:
            conditions.append({"pid": int(q)})
        except ValueError:
            pass
    return conditions


def _result_from_doc(doc: dict, label: str) -> dict:
    pid_str = _pid_str(doc.get("pid")) or ""
    return {
        "mongoId": str(doc["_id"]),
        "pid": pid_str,
        "label": label or pid_str or str(doc["_id"]),
        "sharedWithMerchantsCount": _shared_count(doc),
    }


def _property_label(doc: dict) -> str:
    data = doc.get("data") or {}
    return (data.get(PROPERTY_TITLE_FIELD) or "").strip()


def _contact_label(doc: dict) -> str:
    data = doc.get("data") or {}
    return (data.get(CONTACT_NAME_FIELD) or "").strip()


def _timetable_label(doc: dict) -> str:
    data = doc.get("data") or {}
    for field in ACTIVITY_PROPERTY_FIELDS:
        value = (data.get(field) or "").strip()
        if value:
            return _truncate(value.split("|")[0].strip() or value)
    for field in ACTIVITY_CLIENT_FIELDS:
        value = (data.get(field) or "").strip()
        if value:
            return _truncate(value.split("|")[0].strip() or value)
    desc = (data.get(FIELD_ACTIVITY_DESCRIPTION) or "").strip()
    if desc:
        return _truncate(desc)
    return ""


def _cashflow_label(doc: dict) -> str:
    data = doc.get("data") or {}
    parts: list[str] = []
    desc = (data.get(FIELD_DESCRIPTION) or "").strip()
    if desc:
        parts.append(_truncate(desc, 50))
    prop = (data.get(FIELD_PROPERTY) or "").strip()
    if prop:
        parts.append(_truncate(prop.split("|")[0].strip() or prop, 40))
    amount = data.get(FIELD_AMOUNT)
    if amount not in (None, ""):
        parts.append(str(amount))
    return " · ".join(parts)


async def _find_by_object_id(prop_db, entity_type: str, oid: ObjectId) -> list[dict]:
    indicator, _ = ENTITY_TYPE_INDICATORS[entity_type]
    doc = await prop_db.formdatas.find_one({"_id": oid, "indicator": indicator})
    if not doc:
        return []
    label_fn = {
        "property": _property_label,
        "contact": _contact_label,
        "timetable": _timetable_label,
        "cashflow": _cashflow_label,
    }[entity_type]
    return [_result_from_doc(doc, label_fn(doc))]


async def search_properties_for_unshare(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    field_regex = {"$regex": escaped, "$options": "i"}
    or_conditions = [
        {f"data.{PROPERTY_TITLE_FIELD}": field_regex},
        *_pid_or_conditions(q, escaped),
    ]
    filt = {
        "indicator": "properties",
        "status": "active",
        "$or": or_conditions,
    }
    cursor = prop_db.formdatas.find(
        filt,
        {
            "pid": 1,
            "sharedWithMerchants": 1,
            f"data.{PROPERTY_TITLE_FIELD}": 1,
        },
    ).limit(limit)

    results = []
    seen: set[str] = set()
    async for doc in cursor:
        key = str(doc["_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(_result_from_doc(doc, _property_label(doc)))
    return results


async def search_contacts_for_unshare(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    field_regex = {"$regex": escaped, "$options": "i"}
    or_conditions = [
        {f"data.{CONTACT_NAME_FIELD}": field_regex},
        {f"data.{CONTACT_EMAIL_FIELD}": field_regex},
        {f"data.{CONTACT_PHONE_FIELD}": field_regex},
        *_pid_or_conditions(q, escaped),
    ]
    filt = {
        "indicator": "contacts",
        "status": "active",
        "$or": or_conditions,
    }
    cursor = prop_db.formdatas.find(
        filt,
        {
            "pid": 1,
            "sharedWithMerchants": 1,
            f"data.{CONTACT_NAME_FIELD}": 1,
        },
    ).limit(limit)

    results = []
    seen: set[str] = set()
    async for doc in cursor:
        key = str(doc["_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(_result_from_doc(doc, _contact_label(doc)))
    return results


async def search_timetables_for_unshare(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    field_regex = {"$regex": escaped, "$options": "i"}
    text_fields = (
        FIELD_ACTIVITY_DESCRIPTION,
        *ACTIVITY_PROPERTY_FIELDS,
        *ACTIVITY_CLIENT_FIELDS,
    )
    or_conditions = [
        {f"data.{field}": field_regex} for field in text_fields
    ] + _pid_or_conditions(q, escaped)

    filt = {
        "indicator": ACTIVITY_INDICATOR,
        "status": "active",
        "$or": or_conditions,
    }
    projection = {
        "pid": 1,
        "sharedWithMerchants": 1,
        f"data.{FIELD_ACTIVITY_DESCRIPTION}": 1,
    }
    for field in ACTIVITY_PROPERTY_FIELDS + ACTIVITY_CLIENT_FIELDS:
        projection[f"data.{field}"] = 1

    cursor = prop_db.formdatas.find(filt, projection).limit(limit)
    results = []
    seen: set[str] = set()
    async for doc in cursor:
        key = str(doc["_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(_result_from_doc(doc, _timetable_label(doc)))
    return results


async def search_cashflows_for_unshare(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    field_regex = {"$regex": escaped, "$options": "i"}
    text_fields = (
        FIELD_DESCRIPTION,
        FIELD_PROPERTY,
        FIELD_CASHFLOW_CONTACT,
        FIELD_TRX_RECEIVER,
    )
    or_conditions = [
        {f"data.{field}": field_regex} for field in text_fields
    ] + _pid_or_conditions(q, escaped)

    filt = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        "$or": or_conditions,
    }
    cursor = prop_db.formdatas.find(
        filt,
        {
            "pid": 1,
            "sharedWithMerchants": 1,
            f"data.{FIELD_DESCRIPTION}": 1,
            f"data.{FIELD_PROPERTY}": 1,
            f"data.{FIELD_AMOUNT}": 1,
            f"data.{FIELD_CASHFLOW_CONTACT}": 1,
        },
    ).limit(limit)

    results = []
    seen: set[str] = set()
    async for doc in cursor:
        key = str(doc["_id"])
        if key in seen:
            continue
        seen.add(key)
        results.append(_result_from_doc(doc, _cashflow_label(doc)))
    return results


async def search_entities(
    prop_db,
    entity_type: str,
    query_text: str,
    limit: int = 15,
) -> list[dict]:
    """Search one module by ObjectId, PID, or name/title text."""
    if entity_type not in ENTITY_TYPE_INDICATORS:
        return []

    q = (query_text or "").strip()
    if not q:
        return []

    oid = _parse_object_id(q)
    if oid is not None:
        return await _find_by_object_id(prop_db, entity_type, oid)

    searchers = {
        "property": search_properties_for_unshare,
        "contact": search_contacts_for_unshare,
        "timetable": search_timetables_for_unshare,
        "cashflow": search_cashflows_for_unshare,
    }
    return await searchers[entity_type](prop_db, q, limit)


async def unshare_entity(
    prop_db,
    entity_type: str,
    mongo_id: str | None = None,
    pid: Any = None,
) -> dict:
    """Clear sharedWithMerchants and sharedWith on one formdatas document."""
    if entity_type not in ENTITY_TYPE_INDICATORS:
        return {
            "error": "type must be 'property', 'contact', 'timetable', or 'cashflow'."
        }

    indicator, label = ENTITY_TYPE_INDICATORS[entity_type]
    formdatas_col = prop_db.formdatas
    doc = None

    if mongo_id:
        oid = _parse_object_id(str(mongo_id).strip())
        if not oid:
            return {"error": "Invalid mongoId."}
        doc = await formdatas_col.find_one({"_id": oid, "indicator": indicator})
        if not doc:
            return {"error": f"{label} not found for id {mongo_id}."}
    else:
        pid_value = _parse_pid_value(pid)
        if pid_value is None:
            return {"error": "A valid mongoId or PID is required."}
        doc = await formdatas_col.find_one({"pid": pid_value, "indicator": indicator})
        if not doc:
            return {"error": f"{label} not found for PID {pid_value}."}

    previous_count = _shared_count(doc)
    label_fn = {
        "property": _property_label,
        "contact": _contact_label,
        "timetable": _timetable_label,
        "cashflow": _cashflow_label,
    }[entity_type]
    entity_label = label_fn(doc) or _pid_str(doc.get("pid")) or str(doc["_id"])

    result = await formdatas_col.update_one(
        {"_id": doc["_id"]},
        {
            "$set": {
                "sharedWithMerchants": [],
                "sharedWith": [],
                "metadata.updatedAt": datetime.utcnow(),
                "metadata.updatedBy": "console",
            }
        },
    )

    return {
        "message": f"{label} unshared from all merchants.",
        "type": entity_type,
        "mongoId": str(doc["_id"]),
        "pid": _pid_str(doc.get("pid")),
        "label": entity_label,
        "previousSharedWithMerchantsCount": previous_count,
        "modified": result.modified_count > 0,
    }
