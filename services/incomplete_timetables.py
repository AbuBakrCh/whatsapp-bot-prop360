"""Incomplete timetable (activity) reports: missing contact and/or property attachments."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from services.ledger_report import (
    ACTIVITY_CLIENT_FIELDS,
    ACTIVITY_FORM_URL_TEMPLATE,
    ACTIVITY_INDICATOR,
    ACTIVITY_PROPERTY_FIELDS,
    FIELD_ACTIVITY_DATE,
    FIELD_ACTIVITY_DESCRIPTION,
    GREECE_TZ,
    format_date,
)

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
DETAIL_LIMIT = 500
UNKNOWN_USER_ID = "__unknown__"

ACTIVITY_DATE_PATH = f"data.{FIELD_ACTIVITY_DATE}"
ACTIVITY_DESC_PATH = f"data.{FIELD_ACTIVITY_DESCRIPTION}"


def _empty_field_expr(field: str) -> dict[str, Any]:
    path = f"$data.{field}"
    return {"$eq": [{"$ifNull": [path, ""]}, ""]}


def _all_fields_empty_expr(fields: tuple[str, ...]) -> dict[str, Any]:
    return {"$and": [_empty_field_expr(f) for f in fields]}


def _missing_flags_add_fields() -> dict[str, Any]:
    return {
        "$addFields": {
            "missingContact": _all_fields_empty_expr(ACTIVITY_CLIENT_FIELDS),
            "missingProperty": _all_fields_empty_expr(ACTIVITY_PROPERTY_FIELDS),
        }
    }


def greece_day_bounds_utc(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=GREECE_TZ).astimezone(timezone.utc)
    end = datetime.combine(day, time.max, tzinfo=GREECE_TZ).astimezone(timezone.utc)
    return start, end


def greece_month_bounds_utc(year: int, month: int) -> tuple[datetime, datetime]:
    last_day = calendar.monthrange(year, month)[1]
    start_day = date(year, month, 1)
    end_day = date(year, month, last_day)
    start, _ = greece_day_bounds_utc(start_day)
    _, end = greece_day_bounds_utc(end_day)
    return start, end


def parse_period(period: str) -> tuple[datetime | None, datetime | None]:
    """
    Return UTC inclusive bounds for the period, or (None, None) for all-time.
    Raises ValueError on invalid period.
    """
    text = (period or "yesterday").strip().lower()
    if text == "all":
        return None, None

    now_greece = datetime.now(GREECE_TZ)
    if text == "yesterday":
        yesterday = (now_greece - timedelta(days=1)).date()
        return greece_day_bounds_utc(yesterday)

    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if not match:
        raise ValueError("period must be 'yesterday', 'all', or 'YYYY-MM'")

    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        raise ValueError("period month must be 01-12")
    if year != 2026:
        raise ValueError("month periods are only supported for 2026")

    current = now_greece.date()
    if year > current.year or (year == current.year and month > current.month):
        raise ValueError("period cannot be in the future")

    return greece_month_bounds_utc(year, month)


def _activity_date_parse_stage() -> dict[str, Any]:
    return {
        "$addFields": {
            "parsedActivityDate": {
                "$dateFromString": {
                    "dateString": f"${ACTIVITY_DATE_PATH}",
                    "onError": None,
                    "onNull": None,
                }
            }
        }
    }


def _base_incomplete_stages(
    start_utc: datetime | None, end_utc: datetime | None
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = [
        {
            "$match": {
                "indicator": ACTIVITY_INDICATOR,
                "status": "active",
            }
        },
        _activity_date_parse_stage(),
    ]

    if start_utc is not None and end_utc is not None:
        stages.append(
            {
                "$match": {
                    "parsedActivityDate": {
                        "$gte": start_utc,
                        "$lte": end_utc,
                    }
                }
            }
        )

    stages.extend(
        [
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
    )
    return stages


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


async def list_incomplete_timetables(
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


async def get_incomplete_timetables_for_user(
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
            {"$sort": {"parsedActivityDate": -1, "_id": -1}},
            {"$limit": DETAIL_LIMIT + 1},
            {
                "$project": {
                    "_id": 1,
                    "activityDate": f"${ACTIVITY_DATE_PATH}",
                    "description": f"${ACTIVITY_DESC_PATH}",
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

    activities = []
    for doc in docs:
        activity_id = str(doc.get("_id", ""))
        activities.append(
            {
                "id": activity_id,
                "date": format_date(doc.get("activityDate")),
                "description": str(doc.get("description") or "").strip(),
                "missingContact": bool(doc.get("missingContact")),
                "missingProperty": bool(doc.get("missingProperty")),
                "url": ACTIVITY_FORM_URL_TEMPLATE.format(activity_id=activity_id),
            }
        )

    return {
        "userId": user_id,
        "userName": user_name,
        "email": email,
        "activities": activities,
        "truncated": truncated,
    }
