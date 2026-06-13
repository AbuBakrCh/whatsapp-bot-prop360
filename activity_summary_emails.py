import asyncio
import logging
import traceback
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("activity_summary_emails")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

GREECE_TZ = ZoneInfo("Europe/Athens")


def get_next_hourly_run_greece() -> datetime:
    """Next top-of-hour in Greece time, as UTC-naive datetime for MongoDB."""
    now_greece = datetime.now(GREECE_TZ)
    candidate = now_greece.replace(minute=0, second=0, microsecond=0)
    if now_greece > candidate:
        candidate += timedelta(hours=1)
    return candidate.astimezone(pytz.UTC).replace(tzinfo=None)


DISCLAIMER_HTML = """
<div style="margin-top:40px; padding-top:15px; border-top:1px solid #ddd;
            font-size:12px; color:#777; line-height:1.6;">
    Bu ileti yalnızca bilgilendirme amacıyla hazırlanmıştır. İçeriğin hazırlanmasında
    makul özen gösterilmiş olmakla birlikte, doğruluğu, eksiksizliği veya güvenilirliği
    konusunda açık ya da zımni herhangi bir beyan veya garanti verilmemektedir.
    Herhangi bir mali tutar, tarih veya detay bağımsız olarak teyit edilmelidir.
    Herhangi bir tereddüt veya tutarsızlık durumunda, herhangi bir işlem yapmadan
    önce lütfen bilgileri Kostas ile doğrulayınız. Bu e-postaya dayanılarak alınan
    kararlar sonucunda doğrudan veya dolaylı olarak ortaya çıkabilecek herhangi
    bir kayıp, zarar veya sonuçtan gönderici sorumlu tutulamaz.
</div>
"""


def format_property_name(property_id: str | None) -> str:
    if not property_id:
        return "Mülk"
    return property_id.split("|")[0].strip() or property_id


def _plain_summary_to_html(summary_text: str) -> str:
    if "<" in summary_text and ">" in summary_text:
        return summary_text

    lines = summary_text.splitlines()
    html_parts: list[str] = []
    in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        if stripped.startswith("•") or stripped.startswith("-") or stripped.startswith("*"):
            if not in_list:
                html_parts.append('<ul style="margin:8px 0; padding-left:20px;">')
                in_list = True
            item = stripped.lstrip("•-* ").strip()
            html_parts.append(f'<li style="margin-bottom:6px;">{item}</li>')
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f'<p style="margin:8px 0;">{stripped}</p>')

    if in_list:
        html_parts.append("</ul>")

    return "\n".join(html_parts)


def format_activity_summary_email(property_id: str | None, summary_text: str) -> str:
    property_name = format_property_name(property_id)
    body_html = _plain_summary_to_html(summary_text or "")

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,Helvetica,sans-serif; color:#333; line-height:1.6; margin:0; padding:20px; background:#f4f4f4;">
  <div style="max-width:640px; margin:0 auto; background:#ffffff; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <div style="background:#2d6a4f; color:#ffffff; padding:20px 24px;">
      <h1 style="margin:0; font-size:20px; font-weight:600;">Mülk Faaliyet Özeti</h1>
      <p style="margin:8px 0 0; font-size:14px; opacity:0.9;">{property_name}</p>
    </div>
    <div style="padding:24px;">
      {body_html}
    </div>
    <div style="padding:0 24px 24px;">
      {DISCLAIMER_HTML}
    </div>
  </div>
</body>
</html>"""


async def send_ready_activity_summary_emails(db):
    logger.info("Starting send_ready_activity_summary_emails task")

    summary_col = db.property_activity_summary
    cursor = summary_col.find({
        "status": "ready to send",
        "clientEmail": {"$exists": True, "$nin": [None, ""]},
        "summary": {"$exists": True, "$nin": [None, ""]},
    })

    sent_count = 0
    async for doc in cursor:
        summary_id = doc["_id"]
        client_email = doc.get("clientEmail")
        property_id = doc.get("propertyId")
        summary_text = doc.get("summary", "")

        property_name = format_property_name(property_id)
        subject = f"Mülk faaliyet özeti – {property_name}"
        body = format_activity_summary_email(property_id, summary_text)

        try:
            await asyncio.to_thread(send_email_v2, [client_email], subject, body)
        except Exception as e:
            logger.error("Failed to send summary %s to %s: %s", summary_id, client_email, e)
            traceback.print_exc()
            continue

        await summary_col.update_one(
            {"_id": summary_id},
            {"$set": {
                "status": "sent",
                "emailSentAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
            }}
        )
        sent_count += 1
        logger.info("Sent activity summary %s to %s", summary_id, client_email)

    logger.info("send_ready_activity_summary_emails completed, sent %d emails", sent_count)


def start_activity_summary_emails_scheduler(db):
    scheduler.add_job(
        send_ready_activity_summary_emails,
        CronTrigger(minute=0, timezone=pytz.timezone("Europe/Athens")),
        args=[db],
        id="send_ready_activity_summary_emails_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()
