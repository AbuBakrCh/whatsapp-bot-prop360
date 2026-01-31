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
    - Finds eligible custom-wyey07pb7 forms
    - Sends follow-up email
    """

    logger.info("Starting send_followup_emails task")

    formdatas_col = prop_db.formdatas
    users_col = prop_db.users

    now_iso = datetime.now(dt_timezone.utc).isoformat()

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
            doc.get("data", {})
            .get("field-1760213212062-ask5v2fuy", "")
            .strip()
        )

        subject = "Takip Gerekiyor: Bekleyen Aktivite"

        body = f"""
        <p>Merhaba,</p>

        <p>Aşağıdaki aktivite hakkında bir takip yapılması gerekmektedir:</p>

        <blockquote>
        {activity_desc}
        </blockquote>

        <p>Lütfen en kısa sürede bir güncelleme sağlayınız.</p>

        <p>Aktiviteyi görüntülemek ve güncellemek için aşağıdaki linki kullanabilirsiniz:</p>

        <p>
        <a href="https://prop360.pro/dashboard/forms/custom-wyey07pb7/{doc_id}">
        Aktiviteyi Görüntüle
        </a>
        </p>

        <p>Saygılarımızla,<br/>
        Kostas</p>
        """

        send_email(
            to=email,
            subject=subject,
            body=body,
            cc=FOLLOWUP_CC
        )

        processed += 1

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


