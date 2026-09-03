"""Incomplete cashflow reports: missing contact/owner and/or property attachments."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from services.incomplete_timetables import (
    DETAIL_LIMIT,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    UNKNOWN_USER_ID,
    parse_period,
)
from services.ledger_report import (
    CASHFLOW_FORM_URL_TEMPLATE,
    CASHFLOW_INDICATOR,
    FIELD_CASHFLOW_CONTACT,
    FIELD_DATE,
    FIELD_DESCRIPTION,
    FIELD_PROPERTY,
    format_date,
)

DATE_PATH = f"data.{FIELD_DATE}"
DESC_PATH = f"data.{FIELD_DESCRIPTION}"


def _empty_field_expr(field: str) -> dict[str, Any]:
    path = f"$data.{field}"
    return {"$eq": [{"$ifNull": [path, ""]}, ""]}


def _missing_flags_add_fields() -> dict[str, Any]:
    return {
        "$addFields": {
            "missingContact": _empty_field_expr(FIELD_CASHFLOW_CONTACT),
            "missingProperty": _empty_field_expr(FIELD_PROPERTY),
        }
    }


def _base_incomplete_stages(
    start_utc: datetime | None, end_utc: datetime | None
) -> list[dict[str, Any]]:
    match_filter: dict[str, Any] = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
    }
    if start_utc is not None and end_utc is not None:
        match_filter["metadata.createdAt"] = {
            "$gte": start_utc,
            "$lte": end_utc,
        }

    return [
        {"$match": match_filter},
        _missing_flags_add_fields(),
        {
            "$match": {
                "$or": [
                    {"missingContact": True},
                    {"missingProperty": True},
                ]
            }
        },
    ]


def _normalize_user_id(created_by: Any) -> str:
    if created_by is None:
        return UNKNOWN_USER_ID
    text = str(created_by).strip()
    return text if text else UNKNOWN_USER_ID


def _created_by_match(user_id: str) -> dict[str, Any]:
    if user_id == UNKNOWN_USER_ID:
        return {
            "$or": [
                {"metadata.createdBy": {"$exists": False}},
                {"metadata.createdBy": None},
                {"metadata.createdBy": ""},
            ]
        }
    return {"metadata.createdBy": user_id}


async def list_incomplete_cashflows(
    prop_db,
    *,
    period: str = "yesterday",
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    q: str | None = None,
) -> dict[str, Any]:
    start_utc, end_utc = parse_period(period)
    page = max(1, int(page))
    page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
    skip = (page - 1) * page_size

    pipeline: list[dict[str, Any]] = _base_incomplete_stages(start_utc, end_utc)
    pipeline.extend(
        [
            {
                "$group": {
                    "_id": "$metadata.createdBy",
                    "contactCount": {
                        "$sum": {"$cond": ["$missingContact", 1, 0]}
                    },
                    "propertyCount": {
                        "$sum": {"$cond": ["$missingProperty", 1, 0]}
                    },
                }
            },
            {
                "$lookup": {
                    "from": "users",
                    "localField": "_id",
                    "foreignField": "firebaseId",
                    "as": "creator",
                }
            },
            {
                "$unwind": {
                    "path": "$creator",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "userId": {
                        "$cond": [
                            {
                                "$or": [
                                    {"$eq": ["$_id", None]},
                                    {"$eq": ["$_id", ""]},
                                ]
                            },
                            UNKNOWN_USER_ID,
                            "$_id",
                        ]
                    },
                    "userName": {
                        "$ifNull": ["$creator.displayName", "Unknown"]
                    },
                    "email": {"$ifNull": ["$creator.email", ""]},
                    "contactCount": 1,
                    "propertyCount": 1,
                }
            },
        ]
    )

    search = (q or "").strip()
    if search:
        regex = {"$regex": re.escape(search), "$options": "i"}
        pipeline.append(
            {
                "$match": {
                    "$or": [
                        {"userName": regex},
                        {"email": regex},
                    ]
                }
            }
        )

    pipeline.extend(
        [
            {"$sort": {"userName": 1, "userId": 1}},
            {
                "$facet": {
                    "data": [{"$skip": skip}, {"$limit": page_size}],
                    "meta": [
                        {
                            "$group": {
                                "_id": None,
                                "total": {"$sum": 1},
                                "contactCount": {"$sum": "$contactCount"},
                                "propertyCount": {"$sum": "$propertyCount"},
                            }
                        }
                    ],
                }
            },
        ]
    )

    cursor = prop_db.formdatas.aggregate(pipeline)
    results = await cursor.to_list(length=1)
    facet = results[0] if results else {"data": [], "meta": []}
    meta = facet.get("meta") or []
    totals_doc = meta[0] if meta else {}

    return {
        "data": facet.get("data") or [],
        "total": int(totals_doc.get("total") or 0),
        "page": page,
        "page_size": page_size,
        "totals": {
            "contactCount": int(totals_doc.get("contactCount") or 0),
            "propertyCount": int(totals_doc.get("propertyCount") or 0),
        },
    }


async def get_incomplete_cashflows_for_user(
    prop_db,
    user_id: str,
    *,
    period: str = "yesterday",
) -> dict[str, Any]:
    user_id = _normalize_user_id(user_id)
    start_utc, end_utc = parse_period(period)

    user_doc = None
    if user_id != UNKNOWN_USER_ID:
        user_doc = await prop_db.users.find_one(
            {"firebaseId": user_id},
            {"displayName": 1, "email": 1},
        )

    user_name = (user_doc or {}).get("displayName") or "Unknown"
    email = (user_doc or {}).get("email") or ""

    pipeline: list[dict[str, Any]] = _base_incomplete_stages(start_utc, end_utc)
    pipeline.append({"$match": _created_by_match(user_id)})
    pipeline.extend(
        [
            {"$sort": {"metadata.createdAt": -1, "_id": -1}},
            {"$limit": DETAIL_LIMIT + 1},
            {
                "$project": {
                    "_id": 1,
                    "cashflowDate": f"${DATE_PATH}",
                    "createdAt": "$metadata.createdAt",
                    "description": f"${DESC_PATH}",
                    "missingContact": 1,
                    "missingProperty": 1,
                }
            },
        ]
    )

    cursor = prop_db.formdatas.aggregate(pipeline)
    docs = await cursor.to_list(length=DETAIL_LIMIT + 1)
    truncated = len(docs) > DETAIL_LIMIT
    if truncated:
        docs = docs[:DETAIL_LIMIT]

    cashflows = []
    for doc in docs:
        cashflow_id = str(doc.get("_id", ""))
        display_date = doc.get("cashflowDate") or doc.get("createdAt")
        cashflows.append(
            {
                "id": cashflow_id,
                "date": format_date(display_date),
                "description": str(doc.get("description") or "").strip(),
                "missingContact": bool(doc.get("missingContact")),
                "missingProperty": bool(doc.get("missingProperty")),
                "url": CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cashflow_id),
            }
        )

    return {
        "userId": user_id,
        "userName": user_name,
        "email": email,
        "cashflows": cashflows,
        "truncated": truncated,
    }
