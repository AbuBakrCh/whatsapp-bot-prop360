import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("passport_expiry_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

PASSPORT_FIELD = "field-1741779403411-cgccurwgk"
PASSPORT_EXPIRY_DAYS = int(os.getenv("PASSPORT_EXPIRY_DAYS", "15"))  # configurable
PASSPORT_JOB_ENABLED = "PASSPORT_EXPIRY_JOB_ENABLED"

async def send_passport_email(person_name, person_url, passport_end_date_utc):
    """
    Sends passport expiry notification email
    passport_end_date_utc: datetime in UTC
    """
    greece_tz = ZoneInfo("Europe/Athens")
    passport_end_greece = passport_end_date_utc.astimezone(greece_tz)

    subject = f"Passport Expiry Alert – {person_name}"

    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height:1.6;">
            <p>Hi Operator,</p>

            <p>
            This is an intimation that the passport for the following individual
            is expected to expire within the next <b>{PASSPORT_EXPIRY_DAYS} days</b>.
            </p>

            <p>
            <b>Person Name:</b> {person_name}<br>
            <b>Passport Expiry Date (Greece Time):</b> {passport_end_greece.strftime("%d %B %Y, %H:%M %Z")}<br>
            <b>Profile URL:</b> 
            <a href="{person_url}">{person_url}</a>
            </p>

            <p>
            Kindly review the passport details and take the necessary action if required.
            </p>

            <p>Best regards,<br>
            Kostas</p>
        </body>
    </html>
    """

    logger.info(
        "Sending passport expiry email | name=%s | expiry=%s | url=%s",
        person_name,
        passport_end_greece,
        person_url,
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


async def passport_expiry_check(prop_db):
    if os.getenv(PASSPORT_JOB_ENABLED, "true").lower() != "true":
        logger.info("Passport expiry job disabled")
        return

    logger.info("Starting passport expiry job")

    today = datetime.now(timezone.utc)
    cutoff = today + timedelta(days=PASSPORT_EXPIRY_DAYS)

    formdatas_col = prop_db.formdatas

    cursor = formdatas_col.find(
        {
            "indicator": "contacts",
            "status": "active",
            "metadata.passportExpiryNotified": {"$ne": True},
            f"data.{PASSPORT_FIELD}": {"$exists": True}
        }
    )

    processed = 0
    notified = 0

    async for doc in cursor:
        passport_date_str = doc.get("data", {}).get(PASSPORT_FIELD)
        if not passport_date_str:
            continue

        try:
            passport_end_date = datetime.fromisoformat(passport_date_str.replace("Z", "+00:00"))
        except Exception:
            logger.warning("Invalid passport date format | %s", passport_date_str)
            continue

        if today <= passport_end_date <= cutoff:
            person_name = doc.get("data", {}).get("field-1741774547654-ngd30kdcz", "Passport Expiring Soon")
            person_id = doc["_id"]

            person_url = f"https://prop360.pro/dashboard/forms/contacts/{person_id}"

            await send_passport_email(person_name, person_url, passport_end_date)

            await formdatas_col.update_one(
                {"_id": person_id},
                {
                    "$set": {
                        "metadata.passportExpiryNotified": True,
                        "metadata.passportExpiryNotifiedAt": datetime.utcnow()
                    }
                }
            )

            notified += 1

        processed += 1

    logger.info(
        "Passport expiry job completed | processed=%s | notified=%s",
        processed,
        notified
    )


def start_passport_expiry_scheduler(prop_db):
    """
    Starts passport expiry email scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        passport_expiry_check,
        CronTrigger(
            hour=10,
            minute=5,
            timezone=ZoneInfo("Europe/Athens")
        ),
        args=[prop_db],
        id="passport_expiry_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()