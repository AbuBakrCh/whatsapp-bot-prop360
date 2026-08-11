"""Email property owners about newly created Common Expenses cashflows."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2
from services.ledger_report import (
    CASHFLOW_INDICATOR,
    FIELD_CASHFLOW_CONTACT,
    FIELD_PROPERTY,
    extract_piped_id,
    extract_piped_label,
)

scheduler = AsyncIOScheduler()

logger = logging.getLogger("common_expenses_owner_email_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

FIELD_DOCUMENT_TYPE = "field-1758699529035-xp7lumgx5"
FIELD_ATTACHMENT = "field-1780858164714-qr9of1dwh"
FIELD_CONTACT_EMAIL = "field-1741774690043-v7jylsjj2"
DOCUMENT_TYPE_COMMON_EXPENSES = "Common Expenses"
PROP360_IMAGE_API = "https://prop360.pro/api/image"
LOOKBACK_MINUTES = 15
DOWNLOAD_TIMEOUT_SECONDS = 60

EMAIL_BODY_TEMPLATE = """Sayın {customer_name},

{address} adresinde bulunan evinize ait ortak alan aidatı bildirimi ekte bilgilerinize sunulmaktadır.

Aidatla ilgili detaylara ekte yer alan bildirim üzerinden ulaşabilirsiniz.

Bilgilerinize sunar, iyi günler dileriz.

Bilgi Notu: Bu mesaj bilgilendirme amacıyla gönderilmiştir. Evin kirada olması hâlinde, olağan kullanıma ilişkin ortak alan aidatları kiracıya; binada gerçekleştirilen demirbaş, yenileme ve kalıcı nitelikteki harcamalar ise ev sahibine aittir.
"""


def _is_job_enabled() -> bool:
    return os.getenv("COMMON_EXPENSES_OWNER_EMAIL_JOB_ENABLED", "true").lower() == "true"


def _first_attachment(data: dict[str, Any]) -> dict[str, Any] | None:
    raw = data.get(FIELD_ATTACHMENT) or []
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    return first if isinstance(first, dict) else None


async def _download_pdf_attachment(
    file_entry: dict[str, Any],
) -> tuple[str, bytes, str, str] | None:
    key = (file_entry.get("key") or "").strip()
    if not key:
        return None

    filename = (
        (file_entry.get("originalName") or "").strip()
        or (file_entry.get("fileName") or "").strip()
        or "common-expenses.pdf"
    )
    url = f"{PROP360_IMAGE_API}?key={quote(key, safe='/')}"

    try:
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
            response = await client.get(url, follow_redirects=True)
        if response.status_code != 200 or not response.content:
            logger.warning(
                "Attachment download failed | status=%s | key=%s",
                response.status_code,
                key,
            )
            return None
        return (filename, response.content, "application", "pdf")
    except Exception:
        logger.exception("Attachment download error | key=%s", key)
        return None


async def _resolve_owner_email(prop_db, owner_pid: str) -> str | None:
    try:
        pid_float = float(owner_pid)
    except (TypeError, ValueError):
        return None

    contact_doc = await prop_db.formdatas.find_one(
        {
            "pid": pid_float,
            "indicator": "contacts",
            "status": "active",
        },
        {f"data.{FIELD_CONTACT_EMAIL}": 1},
    )
    if not contact_doc:
        return None

    email = (contact_doc.get("data") or {}).get(FIELD_CONTACT_EMAIL)
    if not email:
        return None
    email = str(email).strip()
    return email or None


async def send_common_expenses_owner_emails(prop_db):
    if not _is_job_enabled():
        logger.info("Common expenses owner email job disabled")
        return

    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=LOOKBACK_MINUTES)

    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        f"data.{FIELD_DOCUMENT_TYPE}": DOCUMENT_TYPE_COMMON_EXPENSES,
        "metadata.createdAt": {"$gte": since},
        "metadata.commonExpensesOwnerEmailSentAt": {"$exists": False},
    }

    cursor = prop_db.formdatas.find(query)
    processed = 0
    sent = 0
    skipped = 0

    async for doc in cursor:
        processed += 1
        cashflow_id = str(doc.get("_id"))
        data = doc.get("data") or {}

        owner_raw = data.get(FIELD_CASHFLOW_CONTACT)
        owner_pid = extract_piped_id(owner_raw)
        customer_name = extract_piped_label(owner_raw, owner_pid)
        address = extract_piped_label(data.get(FIELD_PROPERTY))

        if not owner_pid or not customer_name:
            skipped += 1
            logger.info(
                "Skipped cashflow %s | reason=missing_owner",
                cashflow_id,
            )
            continue

        if not address:
            skipped += 1
            logger.info(
                "Skipped cashflow %s | reason=missing_address",
                cashflow_id,
            )
            continue

        owner_email = await _resolve_owner_email(prop_db, owner_pid)
        if not owner_email:
            skipped += 1
            logger.info(
                "Skipped cashflow %s | reason=missing_email | owner_pid=%s",
                cashflow_id,
                owner_pid,
            )
            continue

        subject = f"{address} - Ortak Aidat Bildirimi"
        body = EMAIL_BODY_TEMPLATE.format(
            customer_name=customer_name,
            address=address,
        )

        attachments: list[tuple[str, bytes, str, str]] = []
        file_entry = _first_attachment(data)
        if file_entry:
            downloaded = await _download_pdf_attachment(file_entry)
            if downloaded:
                attachments.append(downloaded)
            else:
                logger.warning(
                    "Sending body only (download failed) | cashflow=%s",
                    cashflow_id,
                )
        else:
            logger.warning(
                "Sending body only (no attachment field) | cashflow=%s",
                cashflow_id,
            )

        try:
            await asyncio.to_thread(
                send_email_v2,
                [owner_email],
                subject,
                body,
                None,
                None,
                attachments or None,
            )
        except Exception:
            logger.exception(
                "Failed to send email | cashflow=%s | to=%s",
                cashflow_id,
                owner_email,
            )
            continue

        await prop_db.formdatas.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "metadata.commonExpensesOwnerEmailSentAt": datetime.now(
                        timezone.utc
                    )
                }
            },
        )
        sent += 1
        logger.info(
            "Sent common expenses owner email | cashflow=%s | to=%s | attached=%s",
            cashflow_id,
            owner_email,
            bool(attachments),
        )

    logger.info(
        "Common expenses owner email job completed | processed=%s | sent=%s | skipped=%s",
        processed,
        sent,
        skipped,
    )


def start_common_expenses_owner_email_scheduler(prop_db):
    """
    Starts the common expenses owner email scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        send_common_expenses_owner_emails,
        CronTrigger(minute="*/15"),
        args=[prop_db],
        id="common_expenses_owner_email_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("Common expenses owner email scheduler started (every 15 minutes)")
