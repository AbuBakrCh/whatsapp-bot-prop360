"""Email property owners about newly created utility cashflows."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
FIELD_CATEGORY = "field-1780751488281-e84mgqaeo"
FIELD_ATTACHMENT = "field-1780858164714-qr9of1dwh"
FIELD_CONTACT_EMAIL = "field-1741774690043-v7jylsjj2"
DOCUMENT_TYPE_COMMON_EXPENSES = "Common Expenses"
DOCUMENT_TYPE_ELECTRICITY_BILL = "Electricity Bill"
DOCUMENT_TYPE_WATER_BILL = "Water Bill"
CATEGORY_ACCRUAL = "Accrual"
PROP360_IMAGE_API = "https://prop360.pro/api/image"
LOOKBACK_MINUTES = 15
DOWNLOAD_TIMEOUT_SECONDS = 60
DEFAULT_CC = ["ka@investgreece.gr"]


def _merge_emails(*lists: list[str]) -> list[str]:
    """Order-preserving merge with case-insensitive dedupe."""
    seen: set[str] = set()
    result: list[str] = []
    for emails in lists:
        for email in emails or []:
            cleaned = str(email).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(cleaned)
    return result


@dataclass(frozen=True)
class OwnerEmailJobConfig:
    job_id: str
    document_type: str
    sent_at_field: str
    enabled_env: str
    subject_template: str
    body_template: str
    default_filename: str
    log_label: str


COMMON_EXPENSES_EMAIL_BODY = """Sayın {customer_name},

{address} adresinde bulunan evinize ait ortak alan aidatı bildirimi ekte bilgilerinize sunulmaktadır.

Aidatla ilgili detaylara ekte yer alan bildirim üzerinden ulaşabilirsiniz.

Bilgilerinize sunar, iyi günler dileriz.

Bilgi Notu: Bu mesaj bilgilendirme amacıyla gönderilmiştir. Evin kirada olması hâlinde, olağan kullanıma ilişkin ortak alan aidatları kiracıya; binada gerçekleştirilen demirbaş, yenileme ve kalıcı nitelikteki harcamalar ise ev sahibine aittir.
"""

ELECTRICITY_EMAIL_BODY = """Sayın {customer_name},

{address} adresinde bulunan evinize ait elektrik faturası bildirimi ekte bilgilerinize sunulmaktadır.

Faturaya ilişkin detaylara ekte yer alan bildirim üzerinden ulaşabilirsiniz.

Bilgilerinize sunar, iyi günler dileriz.
"""

WATER_EMAIL_BODY = """Sayın {customer_name},

{address} adresinde bulunan evinize ait su faturası bildirimi ekte bilgilerinize sunulmaktadır.

Faturaya ilişkin detaylara ekte yer alan bildirim üzerinden ulaşabilirsiniz.

Bilgilerinize sunar, iyi günler dileriz.
"""

JOB_CONFIGS = [
    OwnerEmailJobConfig(
        job_id="common_expenses_owner_email_job",
        document_type=DOCUMENT_TYPE_COMMON_EXPENSES,
        sent_at_field="commonExpensesOwnerEmailSentAt",
        enabled_env="COMMON_EXPENSES_OWNER_EMAIL_JOB_ENABLED",
        subject_template="{address} - Ortak Aidat Bildirimi",
        body_template=COMMON_EXPENSES_EMAIL_BODY,
        default_filename="common-expenses.pdf",
        log_label="common expenses owner email",
    ),
    OwnerEmailJobConfig(
        job_id="electricity_bill_owner_email_job",
        document_type=DOCUMENT_TYPE_ELECTRICITY_BILL,
        sent_at_field="electricityBillOwnerEmailSentAt",
        enabled_env="ELECTRICITY_BILL_OWNER_EMAIL_JOB_ENABLED",
        subject_template="{address} - Elektrik Faturasi Bildirimi",
        body_template=ELECTRICITY_EMAIL_BODY,
        default_filename="electricity-bill.pdf",
        log_label="electricity bill owner email",
    ),
    OwnerEmailJobConfig(
        job_id="water_bill_owner_email_job",
        document_type=DOCUMENT_TYPE_WATER_BILL,
        sent_at_field="waterBillOwnerEmailSentAt",
        enabled_env="WATER_BILL_OWNER_EMAIL_JOB_ENABLED",
        subject_template="{address} - Su Faturasi Bildirimi",
        body_template=WATER_EMAIL_BODY,
        default_filename="water-bill.pdf",
        log_label="water bill owner email",
    ),
]


async def get_job_email_recipients(db, job_id: str) -> tuple[list[str], list[str]]:
    if db is None:
        return [], []
    doc = await db.job_control.find_one({"_id": job_id})
    if not doc:
        return [], []
    to_list = [str(e).strip() for e in (doc.get("to") or []) if str(e).strip()]
    cc_list = [str(e).strip() for e in (doc.get("cc") or []) if str(e).strip()]
    return to_list, cc_list


def _is_job_enabled(config: OwnerEmailJobConfig) -> bool:
    return os.getenv(config.enabled_env, "true").lower() == "true"


def _first_attachment(data: dict[str, Any]) -> dict[str, Any] | None:
    raw = data.get(FIELD_ATTACHMENT) or []
    if not isinstance(raw, list) or not raw:
        return None
    first = raw[0]
    return first if isinstance(first, dict) else None


async def _download_pdf_attachment(
    file_entry: dict[str, Any],
    default_filename: str,
) -> tuple[str, bytes, str, str] | None:
    key = (file_entry.get("key") or "").strip()
    if not key:
        return None

    filename = (
        (file_entry.get("originalName") or "").strip()
        or (file_entry.get("fileName") or "").strip()
        or default_filename
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


async def send_owner_bill_emails(
    prop_db,
    db=None,
    config: OwnerEmailJobConfig | None = None,
):
    if config is None:
        config = JOB_CONFIGS[0]

    if not _is_job_enabled(config):
        logger.info("%s job disabled", config.log_label)
        return

    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=LOOKBACK_MINUTES)
    db_to, db_cc = await get_job_email_recipients(db, config.job_id)

    query = {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        f"data.{FIELD_DOCUMENT_TYPE}": config.document_type,
        f"data.{FIELD_CATEGORY}": CATEGORY_ACCRUAL,
        "metadata.createdAt": {"$gte": since},
        f"metadata.{config.sent_at_field}": {"$exists": False},
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

        subject = config.subject_template.format(address=address)
        body = config.body_template.format(
            customer_name=customer_name,
            address=address,
        )

        attachments: list[tuple[str, bytes, str, str]] = []
        file_entry = _first_attachment(data)
        if file_entry:
            downloaded = await _download_pdf_attachment(
                file_entry, config.default_filename
            )
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

        to_list = _merge_emails([owner_email], db_to)
        cc_list = _merge_emails(DEFAULT_CC, db_cc)

        try:
            await asyncio.to_thread(
                send_email_v2,
                to_list,
                subject,
                body,
                cc_list,
                None,
                attachments or None,
            )
        except Exception:
            logger.exception(
                "Failed to send email | cashflow=%s | to=%s | cc=%s",
                cashflow_id,
                to_list,
                cc_list,
            )
            continue

        await prop_db.formdatas.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    f"metadata.{config.sent_at_field}": datetime.now(timezone.utc)
                }
            },
        )
        sent += 1
        logger.info(
            "Sent %s | cashflow=%s | to=%s | cc=%s | attached=%s",
            config.log_label,
            cashflow_id,
            to_list,
            cc_list,
            bool(attachments),
        )

    logger.info(
        "%s job completed | processed=%s | sent=%s | skipped=%s",
        config.log_label,
        processed,
        sent,
        skipped,
    )


def start_common_expenses_owner_email_scheduler(db, prop_db):
    """
    Starts the owner utility email schedulers.
    Call this once during FastAPI startup.
    """
    for config in JOB_CONFIGS:
        scheduler.add_job(
            send_owner_bill_emails,
            CronTrigger(minute="*/15"),
            args=[prop_db, db, config],
            id=config.job_id,
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=300,
        )
    if not scheduler.running:
        scheduler.start()
    logger.info("Owner utility email schedulers started (every 15 minutes)")
