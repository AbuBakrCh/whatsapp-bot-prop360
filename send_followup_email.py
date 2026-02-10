import logging

from datetime import datetime, timezone as dt_timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from pytz import timezone
import os
import smtplib
from email.message import EmailMessage

scheduler = AsyncIOScheduler()

logger = logging.getLogger("send_followup_email")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

FOLLOWUP_CC = ["ka@investgreece.gr"]

async def send_followup_emails(prop_db):
    """
    Task: send_followup_emails
    - Sends follow-up emails up to allowed count
    - Never sends more than configured limit
    """

    logger.info("Starting send_followup_emails task")

    formdatas_col = prop_db.formdatas
    users_col = prop_db.users

    now = datetime.now(dt_timezone.utc)
    now_iso = now.isoformat()

    query = {
        "indicator": "custom-wyey07pb7",
        "data.field-1768290921328-dwajjl0vo": "Yes",
        "data.field-1761124691669-lxtkrz3jv": "In Progress",
        "data.field-1768290928665-jgy6150eb": {"$lte": now_iso}
    }

    cursor = formdatas_col.find(query)
    processed = 0
    skipped = 0

    async for doc in cursor:
        data = doc.get("data", {})

        max_followups = int(
            data.get("field-1770024881104-0f0xns6rp", 0) or 0
        )

        sent_count = int(doc.get("followupEmailSentCount", 0))

        if max_followups <= 0:
            skipped += 1
            continue

        if sent_count >= max_followups:
            skipped += 1
            continue

        created_by = doc.get("metadata", {}).get("createdBy")
        if not created_by:
            skipped += 1
            continue

        user = await users_col.find_one(
            {"firebaseId": created_by},
            {"email": 1}
        )

        if not user or not user.get("email"):
            skipped += 1
            continue

        email = user["email"]
        doc_id = str(doc["_id"])

        activity_desc = (
            data.get("field-1760213212062-ask5v2fuy", "")
            .strip()
        )

        subject = "Action Required: Pending Activity"

        body = f"""
        <p>Hello,</p>

        <p>The following activity requires your attention:</p>

        <blockquote>
        {activity_desc}
        </blockquote>

        <p>Please provide an update as soon as possible.</p>

        <p>You can view and update the activity using the link below:</p>

        <p>
        <a href="https://prop360.pro/dashboard/forms/custom-wyey07pb7/{doc_id}">
        View Activity
        </a>
        </p>

        <p>Best regards,<br/>
        Kostas</p>
        """

        try:
            send_email(
                to=email,
                subject=subject,
                body=body,
                cc=FOLLOWUP_CC
            )

            await formdatas_col.update_one(
                {
                    "_id": doc["_id"],
                    "followupEmailSentCount": sent_count
                },
                {
                    "$set": {
                        "lastFollowupEmailAt": now
                    },
                    "$inc": {
                        "followupEmailSentCount": 1
                    }
                }
            )

            processed += 1
            logger.info(
                "Follow-up sent to %s (%s/%s)",
                email,
                sent_count + 1,
                max_followups
            )

        except Exception as e:
            logger.exception("Failed to send follow-up to %s", email)

    logger.info(
        "send_followup_emails completed | processed=%s | skipped=%s",
        processed,
        skipped
    )

    return {
        "processed": processed,
        "skipped": skipped
    }

def start_followup_email_scheduler(prop_db):
    """
    Starts follow-up email scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        send_followup_emails,
        CronTrigger(
            hour=9,
            minute=0,
            timezone=timezone("Europe/Athens")
        ),
        args=[prop_db],
        id="send_followup_emails_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()

def send_email(to: str, subject: str, body: str, cc: list[str] | None = None):
    """
    Send an email via Gmail SMTP.
    """
    email_address = os.getenv("EMAIL_ADDRESS")
    email_app_password = os.getenv("EMAIL_APP_PASSWORD")
    if not email_address or not email_app_password:
        raise ValueError("Missing EMAIL_ADDRESS or EMAIL_APP_PASSWORD environment variables")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_address
    msg["To"] = to

    if cc:
        msg["Cc"] = ", ".join(cc)

    msg["Bcc"] = "m.abubakr916@gmail.com"

    # Detect HTML content
    if "<" in body and ">" in body:
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(email_address, email_app_password)
        smtp.send_message(msg)


