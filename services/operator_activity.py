"""Operator session active-time rankings (Europe/Athens display, UTC storage)."""

from __future__ import annotations

import calendar
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

GREECE_TZ = ZoneInfo("Europe/Athens")
DETAIL_LIMIT = 500
VALID_VIEWS = ("day", "week", "month")


def greece_day_bounds_utc(
    day: date, *, end_at: datetime | None = None
) -> tuple[datetime, datetime]:
    """Return UTC bounds for a Greece calendar day (activities store UTC)."""
    start_local = datetime.combine(day, time.min, tzinfo=GREECE_TZ)
    if end_at is not None:
        end_local = end_at.astimezone(GREECE_TZ)
        if end_local.date() != day:
            end_local = datetime.combine(day, time.max, tzinfo=GREECE_TZ)
    else:
        end_local = datetime.combine(day, time.max, tzinfo=GREECE_TZ)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def greece_week_bounds_utc(day: date) -> tuple[datetime, datetime, date, date]:
    # Monday = 0 … Sunday = 6
    start_day = day - timedelta(days=day.weekday())
    end_day = start_day + timedelta(days=6)
    start_utc, _ = greece_day_bounds_utc(start_day)
    _, end_utc = greece_day_bounds_utc(end_day)
    return start_utc, end_utc, start_day, end_day


def greece_month_bounds_utc(year: int, month: int) -> tuple[datetime, datetime, date, date]:
    last = calendar.monthrange(year, month)[1]
    start_day = date(year, month, 1)
    end_day = date(year, month, last)
    start_utc, _ = greece_day_bounds_utc(start_day)
    _, end_utc = greece_day_bounds_utc(end_day)
    return start_utc, end_utc, start_day, end_day


def parse_anchor_date(date_str: str | None) -> date:
    now_greece = datetime.now(GREECE_TZ).date()
    if not date_str:
        return now_greece
    try:
        return date.fromisoformat(date_str.strip())
    except ValueError as exc:
        raise ValueError("date must be YYYY-MM-DD") from exc


def resolve_period(
    view: str,
    date_str: str | None = None,
    *,
    end_at_now: bool = False,
) -> dict[str, Any]:
    """
    Resolve a day/week/month window in Greece time.

    Returns start/end as UTC-naive datetimes for Mongo matching on metadata.createdAt.
    Labels and display bounds are Europe/Athens.
    """
    text = (view or "day").strip().lower()
    if text not in VALID_VIEWS:
        raise ValueError("view must be 'day', 'week', or 'month'")

    anchor = parse_anchor_date(date_str)
    now_greece = datetime.now(GREECE_TZ)

    if text == "day":
        end_at = now_greece if end_at_now and anchor == now_greece.date() else None
        start_utc, end_utc = greece_day_bounds_utc(anchor, end_at=end_at)
        start_local = datetime.combine(anchor, time.min, tzinfo=GREECE_TZ)
        end_local = (
            end_at.astimezone(GREECE_TZ)
            if end_at is not None
            else datetime.combine(anchor, time.max, tzinfo=GREECE_TZ)
        )
        label = anchor.strftime("%d %B %Y")
        return {
            "type": "day",
            "start_utc": start_utc,
            "end_utc": end_utc,
            "start": start_local.isoformat(),
            "end": end_local.isoformat(),
            "label": label,
            "anchor": anchor.isoformat(),
        }

    if text == "week":
        start_utc, end_utc, start_day, end_day = greece_week_bounds_utc(anchor)
        start_local = datetime.combine(start_day, time.min, tzinfo=GREECE_TZ)
        end_local = datetime.combine(end_day, time.max, tzinfo=GREECE_TZ)
        label = f"{start_day.strftime('%d %b %Y')} – {end_day.strftime('%d %b %Y')}"
        return {
            "type": "week",
            "start_utc": start_utc,
            "end_utc": end_utc,
            "start": start_local.isoformat(),
            "end": end_local.isoformat(),
            "label": label,
            "anchor": anchor.isoformat(),
        }

    start_utc, end_utc, start_day, end_day = greece_month_bounds_utc(
        anchor.year, anchor.month
    )
    start_local = datetime.combine(start_day, time.min, tzinfo=GREECE_TZ)
    end_local = datetime.combine(end_day, time.max, tzinfo=GREECE_TZ)
    label = start_day.strftime("%B %Y")
    return {
        "type": "month",
        "start_utc": start_utc,
        "end_utc": end_utc,
        "start": start_local.isoformat(),
        "end": end_local.isoformat(),
        "label": label,
        "anchor": anchor.isoformat(),
    }


def format_duration(minutes: int) -> str:
    minutes = max(0, int(minutes or 0))
    hours, mins = divmod(minutes, 60)
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def _to_athens_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(GREECE_TZ).isoformat()


def _format_athens_display(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(GREECE_TZ).strftime("%d %b %Y %H:%M")


async def _load_active_users(prop_db) -> list[dict[str, Any]]:
    cursor = prop_db.users.find(
        {"status": "active", "firebaseId": {"$exists": True, "$ne": None}},
        {
            "firebaseId": 1,
            "displayName": 1,
            "email": 1,
            "role": 1,
            "accountType": 1,
            "photoURL": 1,
        },
    )
    users: list[dict[str, Any]] = []
    async for doc in cursor:
        firebase_id = str(doc.get("firebaseId") or "").strip()
        if not firebase_id:
            continue
        users.append(
            {
                "firebaseId": firebase_id,
                "displayName": (doc.get("displayName") or "").strip() or "Unknown",
                "email": (doc.get("email") or "").strip(),
                "role": doc.get("role"),
                "accountType": doc.get("accountType"),
                "photoURL": doc.get("photoURL"),
            }
        )
    return users


async def _aggregate_session_stats(
    prop_db, start_utc: datetime, end_utc: datetime
) -> dict[str, dict[str, Any]]:
    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "module": "session",
                "action": {"$in": ["active", "login"]},
                "metadata.createdAt": {"$gte": start_utc, "$lte": end_utc},
                "performedBy.firebaseId": {"$exists": True, "$ne": None},
            }
        },
        {
            "$group": {
                "_id": "$performedBy.firebaseId",
                "totalActiveMinutes": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$action", "active"]},
                            {
                                "$ifNull": [
                                    "$metadata.additionalInfo.activeMinutes",
                                    0,
                                ]
                            },
                            0,
                        ]
                    }
                },
                "loginCount": {
                    "$sum": {
                        "$cond": [{"$eq": ["$action", "login"]}, 1, 0]
                    }
                },
                "activeEventCount": {
                    "$sum": {
                        "$cond": [{"$eq": ["$action", "active"]}, 1, 0]
                    }
                },
                "lastActivityAt": {"$max": "$metadata.createdAt"},
            }
        },
    ]

    stats: dict[str, dict[str, Any]] = {}
    async for row in prop_db.activities.aggregate(pipeline):
        firebase_id = str(row.get("_id") or "").strip()
        if not firebase_id:
            continue
        stats[firebase_id] = {
            "totalActiveMinutes": int(row.get("totalActiveMinutes") or 0),
            "loginCount": int(row.get("loginCount") or 0),
            "activeEventCount": int(row.get("activeEventCount") or 0),
            "lastActivityAt": row.get("lastActivityAt"),
        }
    return stats


def _build_rankings(
    users: list[dict[str, Any]], stats: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for user in users:
        fid = user["firebaseId"]
        st = stats.get(fid) or {}
        minutes = int(st.get("totalActiveMinutes") or 0)
        last_at = st.get("lastActivityAt")
        rows.append(
            {
                "firebaseId": fid,
                "displayName": user["displayName"],
                "email": user["email"],
                "role": user.get("role"),
                "accountType": user.get("accountType"),
                "photoURL": user.get("photoURL"),
                "totalActiveMinutes": minutes,
                "durationLabel": format_duration(minutes),
                "loginCount": int(st.get("loginCount") or 0),
                "activeEventCount": int(st.get("activeEventCount") or 0),
                "lastActivityAt": _to_athens_iso(last_at),
                "lastActivityLabel": _format_athens_display(last_at),
            }
        )

    rows.sort(
        key=lambda r: (-r["totalActiveMinutes"], (r["displayName"] or "").lower())
    )

    max_minutes = max((r["totalActiveMinutes"] for r in rows), default=0)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["barPct"] = (
            round(100.0 * row["totalActiveMinutes"] / max_minutes, 1)
            if max_minutes > 0
            else 0.0
        )

    total_minutes = sum(r["totalActiveMinutes"] for r in rows)
    with_activity = sum(1 for r in rows if r["totalActiveMinutes"] > 0)
    top = rows[0] if rows else None
    summary = {
        "totalOperators": len(rows),
        "operatorsWithActivity": with_activity,
        "operatorsWithZero": len(rows) - with_activity,
        "totalActiveMinutes": total_minutes,
        "totalDurationLabel": format_duration(total_minutes),
        "topOperator": (
            {
                "firebaseId": top["firebaseId"],
                "displayName": top["displayName"],
                "email": top["email"],
                "totalActiveMinutes": top["totalActiveMinutes"],
                "durationLabel": top["durationLabel"],
            }
            if top and top["totalActiveMinutes"] > 0
            else None
        ),
    }
    return rows, summary


async def get_operator_activity_report(
    prop_db,
    view: str = "day",
    date_str: str | None = None,
    *,
    end_at_now: bool = False,
) -> dict[str, Any]:
    period = resolve_period(view, date_str, end_at_now=end_at_now)
    users = await _load_active_users(prop_db)
    stats = await _aggregate_session_stats(
        prop_db, period["start_utc"], period["end_utc"]
    )
    rankings, summary = _build_rankings(users, stats)
    return {
        "period": {
            "type": period["type"],
            "start": period["start"],
            "end": period["end"],
            "label": period["label"],
            "anchor": period["anchor"],
            "timezone": "Europe/Athens",
        },
        "summary": summary,
        "rankings": rankings,
    }


async def get_operator_activity_detail(
    prop_db,
    firebase_id: str,
    view: str = "day",
    date_str: str | None = None,
) -> dict[str, Any]:
    firebase_id = (firebase_id or "").strip()
    if not firebase_id:
        raise ValueError("firebase_id is required")

    period = resolve_period(view, date_str, end_at_now=False)
    user_doc = await prop_db.users.find_one(
        {"firebaseId": firebase_id, "status": "active"},
        {
            "firebaseId": 1,
            "displayName": 1,
            "email": 1,
            "role": 1,
            "accountType": 1,
            "photoURL": 1,
        },
    )
    if not user_doc:
        # Still allow detail if they have activity but were deactivated mid-lookback
        user_doc = await prop_db.users.find_one(
            {"firebaseId": firebase_id},
            {
                "firebaseId": 1,
                "displayName": 1,
                "email": 1,
                "role": 1,
                "accountType": 1,
                "photoURL": 1,
            },
        )

    query = {
        "module": "session",
        "action": {"$in": ["active", "login"]},
        "performedBy.firebaseId": firebase_id,
        "metadata.createdAt": {
            "$gte": period["start_utc"],
            "$lte": period["end_utc"],
        },
    }
    cursor = (
        prop_db.activities.find(query)
        .sort("metadata.createdAt", -1)
        .limit(DETAIL_LIMIT + 1)
    )
    raw = await cursor.to_list(length=DETAIL_LIMIT + 1)
    truncated = len(raw) > DETAIL_LIMIT
    raw = raw[:DETAIL_LIMIT]

    events: list[dict[str, Any]] = []
    total_minutes = 0
    login_count = 0
    for doc in raw:
        action = doc.get("action")
        created = (doc.get("metadata") or {}).get("createdAt")
        additional = ((doc.get("metadata") or {}).get("additionalInfo") or {})
        active_minutes = int(additional.get("activeMinutes") or 0) if action == "active" else 0
        if action == "active":
            total_minutes += active_minutes
        if action == "login":
            login_count += 1
        events.append(
            {
                "id": str(doc.get("_id")),
                "action": action,
                "description": doc.get("description"),
                "createdAt": _to_athens_iso(created),
                "createdAtLabel": _format_athens_display(created),
                "ip": (doc.get("metadata") or {}).get("ip"),
                "userAgent": (doc.get("metadata") or {}).get("userAgent"),
                "activeMinutes": active_minutes if action == "active" else None,
                "merchantId": doc.get("merchantId"),
            }
        )

    operator = {
        "firebaseId": firebase_id,
        "displayName": (
            (user_doc or {}).get("displayName")
            or ((raw[0].get("performedBy") or {}).get("displayName") if raw else None)
            or "Unknown"
        ),
        "email": (
            (user_doc or {}).get("email")
            or ((raw[0].get("performedBy") or {}).get("email") if raw else None)
            or ""
        ),
        "role": (user_doc or {}).get("role"),
        "accountType": (user_doc or {}).get("accountType"),
        "photoURL": (user_doc or {}).get("photoURL"),
        "totalActiveMinutes": total_minutes,
        "durationLabel": format_duration(total_minutes),
        "loginCount": login_count,
    }

    return {
        "period": {
            "type": period["type"],
            "start": period["start"],
            "end": period["end"],
            "label": period["label"],
            "anchor": period["anchor"],
            "timezone": "Europe/Athens",
        },
        "operator": operator,
        "events": events,
        "truncated": truncated,
    }
