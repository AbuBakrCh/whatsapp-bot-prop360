import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import Any
from zoneinfo import ZoneInfo

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.ai_service import generate_text_with_model

scheduler = AsyncIOScheduler()

logger = logging.getLogger("send_activity_emails")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


async def send_daily_activity_emails(prop_db, db):
    try:
        forms_col = prop_db.formdatas
        email_log = db.email_log

        # Greece timezone
        greece_tz = ZoneInfo("Europe/Athens")

        # Current time in Greece
        now_greece = datetime.now(greece_tz)

        # Yesterday in Greece time
        yesterday_greece = (now_greece - timedelta(days=1)).date()
        start_greece = datetime.combine(
            yesterday_greece, datetime.min.time(), tzinfo=greece_tz
        )
        end_greece = datetime.combine(
            yesterday_greece, datetime.max.time(), tzinfo=greece_tz
        )

        # Convert to UTC for MongoDB query
        start_utc = start_greece.astimezone(pytz.UTC)
        end_utc = end_greece.astimezone(pytz.UTC)

        # Aggregation pipeline
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "indicator": "custom-wyey07pb7",
                    "metadata.createdAt": {
                        "$gte": start_utc,
                        "$lte": end_utc
                    }
                }
            },
            {
                "$project": {
                    "indicator": 1,
                    "activityDescription": "$data.field-1760213212062-ask5v2fuy",
                    "pairs": [
                        {"client": "$data.field-1760213170764-fhjgcg5u0",
                         "property": "$data.field-1760213192233-byk1fbajy"},
                        {"client": "$data.field-1762112354057-0rwwvsbo0",
                         "property": "$data.field-1762112936496-lcg46gwiy"},
                        {"client": "$data.field-1762112414711-wp3hdmt1n",
                         "property": "$data.field-1762112987608-45lv27qbc"},
                        {"client": "$data.field-1764147273289-bqudbub97",
                         "property": "$data.field-1764147281268-oqtfditkd"},
                        {"client": "$data.field-1764147276663-da6q4ymmr",
                         "property": "$data.field-1764147283488-svx61v7j3"},
                        {"client": "$data.field-1764147278883-5oxys6rmc",
                         "property": "$data.field-1764147285842-qbxk0iz1e"},
                    ]
                }
            },
            {"$unwind": "$pairs"},
            {
                "$match": {
                    "pairs.client": {"$ne": None},
                    "pairs.property": {"$ne": None}
                }
            },

            {
                "$group": {
                    "_id": "$pairs.client",
                    "indicator": {"$first": "$indicator"},
                    "activities": {
                        "$push": {
                            "property": "$pairs.property",
                            "description": "$activityDescription"
                        }
                    }
                }
            }
        ]

        cursor = forms_col.aggregate(pipeline)
        results = await cursor.to_list(length=None)

        for res in results:
            client = res["_id"]

            existing = await email_log.find_one({
                "recipientId": client,
                "type": "daily-activity-summary",
                "periodStart": start_utc,
                "periodEnd": end_utc,
                "emailSent": True
            })

            if existing:
                continue

            activities = res.get("activities", [])

            # Group activities by property
            property_groups: dict[str, list[str]] = {}
            for act in activities:
                prop_raw = act.get("property")
                desc = act.get("description")
                if prop_raw and desc:
                    prop_name = prop_raw.split("|")[0].strip()
                    property_groups.setdefault(prop_name, []).append(desc)

            # Build activity text grouped by property
            activity_lines = []
            for prop, descs in property_groups.items():
                desc_text = "\n- ".join(descs)
                activity_lines.append(f"Property: {prop}\n- {desc_text}")

            activities_text = "\n\n".join(activity_lines)

            prompt = f"""
            You are an assistant writing a professional email to a property owner.

            The purpose of this email is to clearly inform the property owner about all tasks performed yesterday and any updates regarding their properties.

            EMAIL STRUCTURE (MUST FOLLOW EXACTLY):

            1) Opening greeting:
            Sayın Mülk Sahibi,

            2) Intro sentence explaining that the email provides a summary of all tasks performed and updates from yesterday.

            3) Activity summary grouped by property:
            - Group activities under each property.
            - Rewrite each activity in a professional, neutral tone.
            - **Remove the actor/person from the activity entirely**; focus only on what was done.
            - Convert statements like "I did X" or "I will do Y" into "X was performed" or "Y is scheduled".
            - Keep all factual details exactly as they are.
            - Make it concise, clear, and informative, as if reporting to the property owner.

            4) Aesthetic and formatting instructions:
            - Format the email as visually appealing HTML.
            - Use headings for property names.
            - Use bullet points for activities.
            - Apply subtle styling such as bold property names, spacing between sections, and clear readable font.
            - Ensure it is fully HTML, suitable to be sent via email clients.
            - DO NOT include any markdown, code fences, or triple backticks. Only generate HTML content.
            
            Translate the entire email into Turkish.

            5) Closing:
            İlginiz için teşekkür ederim.
            Saygılarımla,
            Kostas

            Activities:
            {activities_text}
            """
            summary_text = generate_text_with_model(prompt)

            # Fetch client email
            client_email = None
            try:
                if "|" in client:
                    pid = client.split("|")[-1].strip()
                    pid_float = float(pid)

                    contact_doc = await forms_col.find_one({"pid": pid_float})

                    client_email = contact_doc.get(
                        "data", {}
                    ).get("field-1741774690043-v7jylsjj2")
            except:
                pass

            if not client_email:
                continue

            send_email_v2(
                to=client_email,
                subject="Dünkü mülk faaliyetlerinin özeti",
                body=summary_text
            )

            await email_log.update_one(
                {
                    "recipientId": client,
                    "type": "daily-activity-summary",
                    "periodStart": start_utc,
                    "periodEnd": end_utc
                },
                {
                    "$set": {
                        "emailSent": True,
                        "emailSentAt": datetime.utcnow(),
                        "recipientEmail": client_email
                    }
                },
                upsert=True
            )

    except Exception as e:
        print("Job failed:", e)

def start_daily_activity_emails_scheduler(prop_db, db):
    scheduler.add_job(
        send_daily_activity_emails,
        CronTrigger(
            hour=9,
            minute=0,
            timezone=pytz.timezone("Europe/Athens")
        ),
        args=[prop_db, db],
        id="send_daily_activity_emails_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()

def send_email_v2(to: str, subject: str, body: str, cc: list[str] | None = None):
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