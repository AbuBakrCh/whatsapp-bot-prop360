"""Recurrent console accrual schedules and Prop360 cashflow generation."""

from __future__ import annotations

import logging
import os
import random
import re
from calendar import monthrange
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from bson import ObjectId

from services.ledger_report import (
    CASHFLOW_INDICATOR,
    FIELD_AMOUNT,
    FIELD_CASHFLOW_CONTACT,
    FIELD_CATEGORY,
    FIELD_DATE,
    FIELD_DEBIT_CREDIT,
    FIELD_PROPERTY,
    PROPERTY_TITLE_FIELD,
    property_field_regex,
)

logger = logging.getLogger(__name__)

GREECE_TZ = ZoneInfo("Europe/Athens")
SCHEDULES_COLLECTION = "recurrent_accrual_schedules"
JOB_ID = "recurrent_accrual_job"

CASHFLOW_FORM_ID = "68c2eb885d8b5ec633d3be86"
KOSTAS_MERCHANT_ID = "34137234-52fe-430c-a97d-df3e16525e71"
# Kostas Firebase UID — same as TRANSFER_NEW_OWNER_ID in whatsapp-bot-prop360.env
KOSTAS_OWNER_ID = "l9C9DQ7nd6VOW4PGqNQL81IWaJM2"

FIELD_DOCUMENT_TYPE = "field-1758699529035-xp7lumgx5"
FIELD_NOTES = "field-1757605632930-2hfg96qgr"
RECURRENT_ACCRUAL_NOTE = "Recurrent accrual created by Kostas Console"

CONTACT_NAME_FIELD = "field-1741774547654-ngd30kdcz"
CONTACT_EMAIL_FIELD = "field-1741774690043-v7jylsjj2"
CONTACT_PHONE_FIELD = "field-1741778098761-w10f6zg6y"

MAX_CATCHUP_ITERATIONS = 24

ACCRUAL_DOCUMENT_TYPES = [
    "Bank Transaction",
    "Invoice",
    "Common Expenses",
    "Electricity Bill",
    "Water Bill",
    "Other",
    "Phone/Internet",
    "Vehicle Fuel",
    "Offer",
    "Private Agreement",
    "Notarial Agreement",
    "Expense List",
    "Notification",
    "E-Paravolo",
    "Management Agreement",
    "After Sale Credit Notification",
    "Water Current Account Card",
    "Employee Advances / Employee Expenses",
    "Land Registry - Ktimatologio",
    "Cash In Hand",
    "Property Sale Accrual",
]

FREQUENCIES = frozenset(
    {"weekly", "monthly", "every_2_months", "quarterly", "yearly"}
)

SCHEDULE_STATUSES = frozenset({"active", "paused", "stopped"})


def _today_greece() -> date:
    return datetime.now(GREECE_TZ).date()


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value.strip()[:10])


def _format_amount(amount: float) -> str:
    return f"{amount:.2f}"


def _build_piped(label: str, pid: str | int) -> str:
    return f"{label}|{pid}"


def advance_due_date(start_anchor: date, frequency: str, current_due: date) -> date:
    if frequency not in FREQUENCIES:
        raise ValueError(f"Unsupported frequency: {frequency}")

    if frequency == "weekly":
        return current_due.fromordinal(current_due.toordinal() + 7)

    months_step = {
        "monthly": 1,
        "every_2_months": 2,
        "quarterly": 3,
        "yearly": 12,
    }[frequency]

    year = current_due.year
    month = current_due.month + months_step
    while month > 12:
        month -= 12
        year += 1

    target_day = start_anchor.day
    last_day = monthrange(year, month)[1]
    day = min(target_day, last_day)
    return date(year, month, day)


def build_cashflow_data(
    *,
    document_type: str,
    amount: float,
    due_date: date,
    property_label: str,
    property_pid: str,
    owner_label: str,
    owner_pid: str,
) -> dict[str, Any]:
    due_str = due_date.isoformat()
    return {
        FIELD_DOCUMENT_TYPE: document_type,
        FIELD_CATEGORY: "Accrual",
        FIELD_DEBIT_CREDIT: "Debit",
        FIELD_AMOUNT: _format_amount(amount),
        FIELD_DATE: due_str,
        FIELD_NOTES: RECURRENT_ACCRUAL_NOTE,
        FIELD_PROPERTY: _build_piped(property_label, property_pid),
        FIELD_CASHFLOW_CONTACT: _build_piped(owner_label, owner_pid),
        "attachedForms": [],
    }


def _prop360_owner_id() -> str:
    return os.getenv("TRANSFER_NEW_OWNER_ID") or KOSTAS_OWNER_ID


def _new_cashflow_pid() -> float:
    """Generate a unique-looking numeric PID matching Prop360 style."""
    return float(random.randint(10**15, 10**16 - 1))


async def create_cashflow_in_prop360(
    prop_db,
    data: dict[str, Any],
    *,
    schedule_id: str,
    period: str,
) -> str:
    """
    Insert a cashflow formdata document directly into Prop360 Mongo.

    Scheduled jobs do not have a short-lived Firebase JWT for the merchant
    HTTP API, so we write the same document shape Prop360 would create.
    """
    owner = _prop360_owner_id()
    now = datetime.now(timezone.utc)
    pid = _new_cashflow_pid()
    for _attempt in range(5):
        existing = await prop_db.formdatas.find_one({"pid": pid}, {"_id": 1})
        if not existing:
            break
        pid = _new_cashflow_pid()

    data_payload = dict(data)
    data_payload.setdefault("attachedForms", [])

    doc_id = ObjectId()
    cashflow_id = str(doc_id)

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
            "consoleRecurrentCashflow": True,
            "recurrentAccrualScheduleId": schedule_id,
            "recurrentAccrualPeriod": period,
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
    logger.info(
        "Created recurrent cashflow %s | schedule=%s period=%s",
        cashflow_id,
        schedule_id,
        period,
    )
    return cashflow_id


async def cashflow_exists_for_period(
    prop_db,
    *,
    schedule_id: str,
    period: str,
    property_pid: str,
) -> bool:
    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        "metadata.recurrentAccrualScheduleId": schedule_id,
        "metadata.recurrentAccrualPeriod": period,
        f"data.{FIELD_PROPERTY}": {"$regex": property_field_regex(str(property_pid))},
    }
    doc = await prop_db.formdatas.find_one(query, {"_id": 1})
    return doc is not None


async def search_contacts(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    """Search active `indicator: contacts` formdatas by name, email, phone, or PID."""
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    field_regex = {"$regex": escaped, "$options": "i"}
    or_conditions: list[dict[str, Any]] = [
        {f"data.{CONTACT_NAME_FIELD}": field_regex},
        {f"data.{CONTACT_EMAIL_FIELD}": field_regex},
        {f"data.{CONTACT_PHONE_FIELD}": field_regex},
        {
            "$expr": {
                "$regexMatch": {
                    "input": {"$toString": "$pid"},
                    "regex": escaped,
                    "options": "i",
                }
            }
        },
    ]

    if q.isdigit():
        try:
            or_conditions.append({"pid": float(q)})
        except ValueError:
            pass
        try:
            or_conditions.append({"pid": int(q)})
        except ValueError:
            pass

    filt = {
        "indicator": "contacts",
        "status": "active",
        "$or": or_conditions,
    }
    cursor = prop_db.formdatas.find(
        filt,
        {
            "pid": 1,
            f"data.{CONTACT_NAME_FIELD}": 1,
        },
    ).limit(limit)

    results = []
    seen_pids: set[str] = set()
    async for doc in cursor:
        pid = doc.get("pid")
        if pid is None:
            continue
        pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
        if pid_str in seen_pids:
            continue
        seen_pids.add(pid_str)
        data = doc.get("data") or {}
        label = (data.get(CONTACT_NAME_FIELD) or "").strip() or pid_str
        results.append(
            {
                "pid": pid_str,
                "label": label,
                "mongoId": str(doc["_id"]),
            }
        )
    return results


async def search_properties(prop_db, query_text: str, limit: int = 15) -> list[dict]:
    """Search active `indicator: properties` formdatas by title or PID."""
    q = query_text.strip()
    if len(q) < 2:
        return []

    escaped = re.escape(q)
    title_regex = {"$regex": escaped, "$options": "i"}
    or_conditions: list[dict[str, Any]] = [
        {f"data.{PROPERTY_TITLE_FIELD}": title_regex},
        {
            "$expr": {
                "$regexMatch": {
                    "input": {"$toString": "$pid"},
                    "regex": escaped,
                    "options": "i",
                }
            }
        },
    ]

    if q.isdigit():
        try:
            or_conditions.append({"pid": float(q)})
        except ValueError:
            pass
        try:
            or_conditions.append({"pid": int(q)})
        except ValueError:
            pass

    filt = {
        "indicator": "properties",
        "status": "active",
        "$or": or_conditions,
    }
    cursor = prop_db.formdatas.find(
        filt,
        {
            "pid": 1,
            f"data.{PROPERTY_TITLE_FIELD}": 1,
        },
    ).limit(limit)

    results = []
    seen_pids: set[str] = set()
    async for doc in cursor:
        pid = doc.get("pid")
        if pid is None:
            continue
        pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
        if pid_str in seen_pids:
            continue
        seen_pids.add(pid_str)
        data = doc.get("data") or {}
        title = (data.get(PROPERTY_TITLE_FIELD) or "").strip()
        label = title or pid_str
        results.append({"pid": pid_str, "label": label})
    return results


async def _generate_for_period(
    prop_db,
    schedule: dict,
    period_date: date,
) -> list[dict]:
    schedule_id = str(schedule["_id"])
    contact = schedule["contact"]
    owner_pid = str(contact["pid"])
    owner_label = contact["label"]
    created_entries: list[dict] = []
    schedules = prop_db[SCHEDULES_COLLECTION]

    for prop in schedule["properties"]:
        property_pid = str(prop["pid"])
        period_str = period_date.isoformat()

        if await cashflow_exists_for_period(
            prop_db,
            schedule_id=schedule_id,
            period=period_str,
            property_pid=property_pid,
        ):
            logger.info(
                "Skip existing | schedule=%s period=%s property=%s",
                schedule_id,
                period_str,
                property_pid,
            )
            continue

        data = build_cashflow_data(
            document_type=schedule["documentType"],
            amount=float(schedule["amount"]),
            due_date=period_date,
            property_label=prop["label"],
            property_pid=property_pid,
            owner_label=owner_label,
            owner_pid=owner_pid,
        )

        cashflow_id = await create_cashflow_in_prop360(
            prop_db,
            data,
            schedule_id=schedule_id,
            period=period_str,
        )

        entry = {
            "period": period_str,
            "propertyPid": property_pid,
            "propertyLabel": prop["label"],
            "cashflowId": cashflow_id,
            "createdAt": datetime.now(timezone.utc),
        }
        created_entries.append(entry)

    if created_entries:
        await schedules.update_one(
            {"_id": schedule["_id"]},
            {"$push": {"generated": {"$each": created_entries}}},
        )

    return created_entries


async def process_one_schedule(
    prop_db,
    schedule: dict,
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """
    Generate all due periods for one active schedule up through `today`
    (Europe/Athens). Advances nextDueDate after each successful period.
    """
    result = {
        "periodsProcessed": 0,
        "cashflowsCreated": 0,
        "error": None,
    }
    if not schedule or schedule.get("status") != "active":
        return result

    today = today or _today_greece()
    schedule_id = str(schedule["_id"])
    schedules = prop_db[SCHEDULES_COLLECTION]

    try:
        start_date = _parse_iso_date(schedule["startDate"])
        next_due = _parse_iso_date(schedule["nextDueDate"])
        end_date = (
            _parse_iso_date(schedule["endDate"]) if schedule.get("endDate") else None
        )
        frequency = schedule["frequency"]

        iterations = 0
        while next_due <= today and iterations < MAX_CATCHUP_ITERATIONS:
            if end_date and next_due > end_date:
                await schedules.update_one(
                    {"_id": schedule["_id"]},
                    {
                        "$set": {
                            "status": "stopped",
                            "updatedAt": datetime.now(timezone.utc),
                        }
                    },
                )
                break

            created = await _generate_for_period(prop_db, schedule, next_due)
            result["periodsProcessed"] += 1
            result["cashflowsCreated"] += len(created)

            next_due = advance_due_date(start_date, frequency, next_due)
            iterations += 1

            await schedules.update_one(
                {"_id": schedule["_id"]},
                {
                    "$set": {
                        "nextDueDate": next_due.isoformat(),
                        "lastRunAt": datetime.now(timezone.utc),
                        "updatedAt": datetime.now(timezone.utc),
                    }
                },
            )
            schedule = await schedules.find_one({"_id": schedule["_id"]})
            if not schedule or schedule.get("status") != "active":
                break

    except Exception as exc:
        logger.exception("Schedule %s failed", schedule_id)
        result["error"] = str(exc)

    return result


async def process_due_schedules(prop_db) -> dict[str, Any]:
    today = _today_greece()
    stats = {
        "schedulesScanned": 0,
        "periodsProcessed": 0,
        "cashflowsCreated": 0,
        "errors": [],
    }

    cursor = prop_db[SCHEDULES_COLLECTION].find({"status": "active"})
    async for schedule in cursor:
        stats["schedulesScanned"] += 1
        one = await process_one_schedule(prop_db, schedule, today=today)
        stats["periodsProcessed"] += one["periodsProcessed"]
        stats["cashflowsCreated"] += one["cashflowsCreated"]
        if one.get("error"):
            stats["errors"].append(
                {"scheduleId": str(schedule["_id"]), "error": one["error"]}
            )

    return stats
