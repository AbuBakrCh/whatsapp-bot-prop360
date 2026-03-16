import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("ide_expiry_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


IDE_FIELD = "field-1750790530531-lymraol5r"
IDE_END = "column-1750791190622-1fsx4x0fe"
IDE_EXPIRY_DAYS = int(os.getenv("IDE_EXPIRY_DAYS", "15"))


def _is_ide_job_enabled():
    return os.getenv("IDE_EXPIRY_JOB_ENABLED", "true").lower() == "true"


async def send_ide_email(title_full, title, record_url, ide_end_date_utc):
    """
    Sends IDE expiry notification email
    ide_end_date_utc: datetime in UTC
    """

    greece_tz = ZoneInfo("Europe/Athens")
    ide_end_greece = ide_end_date_utc.astimezone(greece_tz)

    subject = f"Property IDE Expiry Alert – {title}"

    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height:1.6;">
            <p>Hi Operator,</p>

            <p>
            This is an intimation that the IDE for the following property
            is expected to expire within the next <b>{IDE_EXPIRY_DAYS} days</b>.
            </p>

            <p>
            <b>Title:</b> {title_full}<br>
            <b>IDE Expiry Date (Greece Time):</b> {ide_end_greece.strftime("%d %B %Y, %H:%M %Z")}<br>
            <b>Property URL:</b>
            <a href="{record_url}">{record_url}</a>
            </p>

            <p>
            Kindly review the property details and take the necessary action if required.
            </p>

            <p>Best regards,<br>
            Kostas</p>
        </body>
    </html>
    """

    logger.info(
        "Sending IDE expiry email | title=%s | expiry=%s | url=%s",
        title,
        ide_end_greece,
        record_url,
    )

    loop = asyncio.get_running_loop()

    await loop.run_in_executor(
        None,
        send_email_v2,
        [
            "ka@investgreece.gr",
            "info@investgreece.gr"
        ],
        subject,
        body,
        None
    )


async def ide_expiry_check(prop_db):

    if not _is_ide_job_enabled():
        logger.info("IDE expiry job disabled")
        return

    logger.info("Starting IDE expiry job")

    today = datetime.now(timezone.utc)
    cutoff = today + timedelta(days=IDE_EXPIRY_DAYS)

    formdatas_col = prop_db.formdatas

    cursor = formdatas_col.find(
        {
            "indicator": "properties",
            "status": "active",
            "metadata.ideExpiryNotified": {"$ne": True},
            f"data.{IDE_FIELD}": {"$exists": True}
        }
    )

    processed = 0
    notified = 0

    async for doc in cursor:

        ide_records = doc.get("data", {}).get(IDE_FIELD, [])

        if not ide_records:
            continue

        for ide in ide_records:

            if not ide:
                continue

            end_date_str = ide.get(IDE_END)

            if not end_date_str:
                continue

            try:
                end_date = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00")
                )
            except Exception:
                logger.warning("Invalid IDE date format | %s", end_date_str)
                continue

            if today <= end_date <= cutoff:

                title_full = doc.get(
                    "data", {}
                ).get(
                    "field-1741536181001-wd8it2quy",
                    "IDE Expiring Soon"
                )

                title = title_full.split("-")[0].strip()

                record_id = doc["_id"]

                record_url = f"https://prop360.pro/dashboard/forms/properties/{record_id}"

                await send_ide_email(title_full, title, record_url, end_date)

                await formdatas_col.update_one(
                    {"_id": record_id},
                    {
                        "$set": {
                            "metadata.ideExpiryNotified": True,
                            "metadata.ideExpiryNotifiedAt": datetime.utcnow()
                        }
                    }
                )

                notified += 1
                break

        processed += 1

    logger.info(
        "IDE expiry job completed | processed=%s | notified=%s",
        processed,
        notified
    )


def start_ide_expiry_scheduler(prop_db):
    """
    Starts IDE expiry email scheduler.
    Call this once during FastAPI startup.
    """

    scheduler.add_job(
        ide_expiry_check,
        CronTrigger(
            hour=10,
            minute=10,
            timezone=ZoneInfo("Europe/Athens")
        ),
        args=[prop_db],
        id="ide_expiry_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )

    scheduler.start()