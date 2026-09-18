"""Sync denormalized property/contact labels on cashflow formdatas."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from services.ledger_report import (
    CASHFLOW_INDICATOR,
    FIELD_CASHFLOW_CONTACT,
    FIELD_PROPERTY,
    PROPERTY_TITLE_FIELD,
    extract_piped_id,
    normalize_property_field,
)

logger = logging.getLogger("sync_cashflow_refs")

FIELD_TRX_RECEIVER = "field-1758478869002-dpm43pot3"
CONTACT_NAME_FIELD = "field-1741774547654-ngd30kdcz"

JOB_ID = "sync_cashflow_refs_job"
PAGE_SIZE = 200
STOP_CHECK_EVERY = 50

REF_SPECS: tuple[tuple[str, str, str], ...] = (
    (FIELD_PROPERTY, "properties", PROPERTY_TITLE_FIELD),
    (FIELD_CASHFLOW_CONTACT, "contacts", CONTACT_NAME_FIELD),
    (FIELD_TRX_RECEIVER, "contacts", CONTACT_NAME_FIELD),
)


def _pid_float(pid: str) -> float | None:
    try:
        return float(pid)
    except (TypeError, ValueError):
        return None


def _pid_keys(pid_val: Any) -> list[str]:
    keys: list[str] = []
    try:
        as_float = float(pid_val)
    except (TypeError, ValueError):
        return [str(pid_val)] if pid_val is not None else []
    keys.append(str(as_float))
    if as_float == int(as_float):
        keys.append(str(int(as_float)))
    return keys


def _build_piped(label: str, pid: str) -> str:
    return f"{label}|{pid}"


def _lookup_label(label_by_pid: dict[str, str | None], pid: str) -> tuple[bool, str | None]:
    """Return (found, label). found=False means source missing/inactive."""
    if pid in label_by_pid:
        return True, label_by_pid[pid]
    pid_f = _pid_float(pid)
    if pid_f is None:
        return False, None
    for key in _pid_keys(pid_f):
        if key in label_by_pid:
            return True, label_by_pid[key]
    return False, None


def _desired_value(
    current: Any,
    label_by_pid: dict[str, str | None],
) -> tuple[str | None, str]:
    """
    Return (new_value_or_None_if_unchanged, action).

    action is one of: "update", "clear", "skip".
    """
    text = normalize_property_field(current)
    if not text:
        return None, "skip"

    pid = extract_piped_id(text)
    if not pid:
        return None, "skip"

    found, label = _lookup_label(label_by_pid, pid)
    if not found:
        return "", "clear"

    if not label:
        # Active source exists but has no title/name — leave cashflow as-is
        return None, "skip"

    desired = _build_piped(label, pid)
    if desired == text:
        return None, "skip"
    return desired, "update"


async def _should_stop(db) -> bool:
    control = await db.job_control.find_one({"_id": JOB_ID})
    return bool(control and control.get("status") == "stop")


async def _load_labels(
    prop_db,
    indicator: str,
    label_field: str,
    pids: set[str],
) -> dict[str, str | None]:
    """
    Map pid string -> current title/name for active docs.

    Key present means an active source was found.
    Value None means found but title/name is empty.
    """
    if not pids:
        return {}

    pid_floats: list[float] = []
    for pid in pids:
        pid_f = _pid_float(pid)
        if pid_f is not None:
            pid_floats.append(pid_f)

    if not pid_floats:
        return {}

    labels: dict[str, str | None] = {}
    cursor = prop_db.formdatas.find(
        {
            "indicator": indicator,
            "status": "active",
            "pid": {"$in": pid_floats},
        },
        {"pid": 1, f"data.{label_field}": 1},
    )
    async for doc in cursor:
        pid_val = doc.get("pid")
        if pid_val is None:
            continue
        title = normalize_property_field((doc.get("data") or {}).get(label_field))
        for key in _pid_keys(pid_val):
            labels[key] = title
        for orig in pids:
            orig_f = _pid_float(orig)
            if orig_f is not None and orig_f == float(pid_val):
                labels[orig] = title

    return labels


async def sync_cashflow_refs(prop_db, db) -> dict[str, int]:
    """
    Rewrite cashflow Property / Property Owner / Trx Receiver labels from live
    property/contact docs. Clear the field when the source is missing or inactive.
    """
    stats = {
        "scanned": 0,
        "updated": 0,
        "cleared": 0,
        "skipped": 0,
        "stopped": 0,
    }

    formdatas = prop_db.formdatas
    field_paths = [f"data.{field}" for field, _, _ in REF_SPECS]
    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        "$or": [
            {path: {"$exists": True, "$nin": [None, ""]}} for path in field_paths
        ],
    }
    projection = {
        "_id": 1,
        **{path: 1 for path in field_paths},
    }

    cursor = formdatas.find(query, projection).batch_size(PAGE_SIZE)
    batch: list[dict] = []
    processed_since_stop_check = 0

    async def flush_batch(docs: list[dict]) -> bool:
        """Process a batch. Returns False if stop was requested."""
        nonlocal processed_since_stop_check

        if not docs:
            return True

        property_pids: set[str] = set()
        contact_pids: set[str] = set()

        for doc in docs:
            data = doc.get("data") or {}
            for field, indicator, _ in REF_SPECS:
                pid = extract_piped_id(data.get(field))
                if not pid:
                    continue
                if indicator == "properties":
                    property_pids.add(pid)
                else:
                    contact_pids.add(pid)

        property_labels = await _load_labels(
            prop_db, "properties", PROPERTY_TITLE_FIELD, property_pids
        )
        contact_labels = await _load_labels(
            prop_db, "contacts", CONTACT_NAME_FIELD, contact_pids
        )

        labels_by_indicator = {
            "properties": property_labels,
            "contacts": contact_labels,
        }

        for doc in docs:
            if processed_since_stop_check >= STOP_CHECK_EVERY:
                processed_since_stop_check = 0
                if await _should_stop(db):
                    stats["stopped"] = 1
                    return False

            processed_since_stop_check += 1
            stats["scanned"] += 1
            data = doc.get("data") or {}
            sets: dict[str, Any] = {}

            for field, indicator, _ in REF_SPECS:
                current = data.get(field)
                if not normalize_property_field(current):
                    continue
                new_value, action = _desired_value(
                    current, labels_by_indicator[indicator]
                )
                if action == "skip" or new_value is None:
                    stats["skipped"] += 1
                    continue
                sets[f"data.{field}"] = new_value
                if action == "clear":
                    stats["cleared"] += 1
                else:
                    stats["updated"] += 1

            if not sets:
                continue

            sets["metadata.updatedAt"] = datetime.utcnow()
            await formdatas.update_one({"_id": doc["_id"]}, {"$set": sets})

        return True

    async for doc in cursor:
        batch.append(doc)
        if len(batch) >= PAGE_SIZE:
            if not await flush_batch(batch):
                logger.info("Stop signal received during sync_cashflow_refs")
                return stats
            batch = []

    if batch:
        if not await flush_batch(batch):
            logger.info("Stop signal received during sync_cashflow_refs")
            return stats

    logger.info(
        "sync_cashflow_refs done | scanned=%s updated=%s cleared=%s skipped=%s stopped=%s",
        stats["scanned"],
        stats["updated"],
        stats["cleared"],
        stats["skipped"],
        stats["stopped"],
    )
    return stats
