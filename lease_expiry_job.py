import logging
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("lease_expiry_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


LEASE_FIELD = "field-1751378159230-z600hychf"
LEASE_END = "column-1751378319963-1aaste5j5"
LEASE_EXPIRY_DAYS = int(os.getenv("LEASE_EXPIRY_DAYS", "15"))

def _is_job_enabled():
    return os.getenv("LEASE_EXPIRY_JOB_ENABLED", "true").lower() == "true"

async def send_email(property_title_full, property_title, property_url, lease_end_date_utc):
    """
    Sends lease expiry notification email
    lease_end_date_utc: datetime in UTC
    """

    # Convert to Greece time
    greece_tz = ZoneInfo("Europe/Athens")
    lease_end_greece = lease_end_date_utc.astimezone(greece_tz)

    subject = f"Lease Expiry Alert – {property_title}"

    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height:1.6;">
            <p>Hi Operator,</p>

            <p>
            This is an intimation that the lease agreement for the following property
            is expected to expire within the next <b>{LEASE_EXPIRY_DAYS} days</b>.
            </p>

            <p>
            <b>Property Title:</b> {property_title_full}<br>
            <b>Lease Expiry Date (Greece Time):</b> {lease_end_greece.strftime("%d %B %Y, %H:%M %Z")}<br>
            <b>Property URL:</b> 
            <a href="{property_url}">{property_url}</a>
            </p>

            <p>
            Kindly review the lease details and take the necessary action if required.
            </p>

            <p>Best regards,<br>
            Kostas</p>
        </body>
    </html>
    """

    logger.info(
        "Sending lease expiry email | title=%s | expiry=%s | url=%s",
        property_title,
        lease_end_greece,
        property_url,
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


async def lease_expiry_check(prop_db):

    if not _is_job_enabled():
        logger.info("Lease expiry job disabled")
        return

    logger.info("Starting lease expiry job")

    today = datetime.now(timezone.utc)
    cutoff = today + timedelta(days=LEASE_EXPIRY_DAYS)

    formdatas_col = prop_db.formdatas

    cursor = formdatas_col.find(
        {
            "indicator": "properties",
            "status": "active",
            "metadata.leaseExpiryNotified": {"$ne": True},
            f"data.{LEASE_FIELD}": {"$exists": True}
        }
    )

    processed = 0
    notified = 0

    async for doc in cursor:

        leases = doc.get("data", {}).get(LEASE_FIELD, [])
        if not leases:
            continue

        for lease in leases:

            if not lease:
                continue

            end_date_str = lease.get(LEASE_END)

            if not end_date_str:
                continue

            try:
                end_date = datetime.fromisoformat(
                    end_date_str.replace("Z", "+00:00")
                )
            except Exception:
                logger.warning("Invalid date format | %s", end_date_str)
                continue

            if today <= end_date <= cutoff: #Is the lease expiring between today and 15 days from now?

                property_title_full = doc.get("data", {}).get("field-1741536181001-wd8it2quy", "Property Lease Expiring Soon")
                property_title = property_title_full.split("-")[0].strip()
                property_id = doc["_id"]

                property_url = f"https://prop360.pro/dashboard/forms/properties/{property_id}"

                await send_email(property_title_full, property_title, property_url, end_date)

                await formdatas_col.update_one(
                    {"_id": property_id},
                    {
                        "$set": {
                            "metadata.leaseExpiryNotified": True,
                            "metadata.leaseExpiryNotifiedAt": datetime.utcnow()
                        }
                    }
                )

                notified += 1
                break

        processed += 1

    logger.info(
        "Lease expiry job completed | processed=%s | notified=%s",
        processed,
        notified
    )


def start_lease_expiry_scheduler(prop_db):
    """
    Starts follow-up email scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        lease_expiry_check,
        CronTrigger(
            hour=10,
            minute=0,
            timezone=ZoneInfo("Europe/Athens")
        ),
        args=[prop_db],
        id="lease_expiry_check",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()
