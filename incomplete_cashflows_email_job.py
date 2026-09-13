"""Daily email of incomplete cashflow stats.

TEMPORARY: period is since-2026-09-07 → today. Revert PERIOD to "yesterday" later.
"""

from __future__ import annotations

import asyncio
import html
import logging
import traceback
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2
from services.incomplete_cashflows import MAX_PAGE_SIZE, list_incomplete_cashflows

scheduler = AsyncIOScheduler()

logger = logging.getLogger("incomplete_cashflows_email_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

GREECE_TZ = ZoneInfo("Europe/Athens")
JOB_ID = "incomplete_cashflows_email_job"
EMAIL_LOG_TYPE = "incomplete-cashflows-daily"
RECIPIENT = "ka@investgreece.gr"
# TEMPORARY catch-up; revert to "yesterday" later.
PERIOD = "since-2026-09-07"
TEMP_RANGE_START = date(2026, 9, 7)


async def fetch_all_incomplete_for_period(prop_db) -> dict:
    page = 1
    all_rows: list[dict] = []
    totals = {"contactCount": 0, "propertyCount": 0}
    total_users = 0

    while True:
        result = await list_incomplete_cashflows(
            prop_db,
            period=PERIOD,
            page=page,
            page_size=MAX_PAGE_SIZE,
        )
        rows = result.get("data") or []
        all_rows.extend(rows)
        totals = result.get("totals") or totals
        total_users = int(result.get("total") or 0)

        if not rows or len(all_rows) >= total_users:
            break
        page += 1
        if page > 500:
            logger.warning("Stopped paging incomplete cashflows after 500 pages")
            break

    return {
        "data": all_rows,
        "total": total_users,
        "totals": totals,
    }


def _period_label() -> str:
    today = datetime.now(GREECE_TZ).date()
    return (
        f"{TEMP_RANGE_START.strftime('%d %B %Y')} – {today.strftime('%d %B %Y')}"
    )


def format_incomplete_cashflows_email(payload: dict, date_label: str) -> str:
    rows = payload.get("data") or []
    totals = payload.get("totals") or {}
    contact_total = int(totals.get("contactCount") or 0)
    property_total = int(totals.get("propertyCount") or 0)
    user_count = int(payload.get("total") or len(rows))

    if not rows:
        table_html = (
            '<p style="margin:16px 0;color:#555;">'
            "No incomplete cashflows found for this period."
            "</p>"
        )
    else:
        row_html = []
        for row in rows:
            name = html.escape(str(row.get("userName") or "Unknown"))
            email = html.escape(str(row.get("email") or ""))
            contact_count = int(row.get("contactCount") or 0)
            property_count = int(row.get("propertyCount") or 0)
            user_cell = name
            if email:
                user_cell += (
                    f'<br><span style="color:#777;font-size:12px;">{email}</span>'
                )
            row_html.append(
                "<tr>"
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;">{user_cell}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;text-align:right;">{contact_count}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;text-align:right;">{property_count}</td>'
                "</tr>"
            )

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;margin-top:16px;font-size:14px;">
          <thead>
            <tr style="background:#f3f4f6;text-align:left;">
              <th style="padding:10px 12px;">User</th>
              <th style="padding:10px 12px;text-align:right;">Contact</th>
              <th style="padding:10px 12px;text-align:right;">Property</th>
            </tr>
          </thead>
          <tbody>
            {''.join(row_html)}
          </tbody>
          <tfoot>
            <tr style="background:#f9fafb;font-weight:600;">
              <td style="padding:10px 12px;">Total ({user_count} users)</td>
              <td style="padding:10px 12px;text-align:right;">{contact_total}</td>
              <td style="padding:10px 12px;text-align:right;">{property_total}</td>
            </tr>
          </tfoot>
        </table>
        """

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,Helvetica,sans-serif;color:#333;line-height:1.5;margin:0;padding:20px;background:#f4f4f4;">
  <div style="max-width:720px;margin:0 auto;background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <div style="background:#1f4e79;color:#ffffff;padding:20px 24px;">
      <h1 style="margin:0;font-size:20px;font-weight:600;">Incomplete Cashflows</h1>
      <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Since 7 Sep — {html.escape(date_label)}</p>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 8px;">
        Agents with cashflows missing contact/owner and/or property attachments.
      </p>
      <p style="margin:0;color:#555;font-size:13px;">
        Contact = cashflows with no contact/owner attached.
        Property = cashflows with no property attached.
        Missing both increments both counts.
      </p>
      {table_html}
    </div>
  </div>
</body>
</html>"""


async def send_incomplete_cashflows_daily_email(prop_db, db):
    logger.info("Starting incomplete cashflows daily email job")

    await db.job_control.update_one(
        {"_id": JOB_ID},
        {"$setOnInsert": {"status": "start"}},
        upsert=True,
    )

    result = await db.job_control.update_one(
        {"_id": JOB_ID, "running": {"$ne": True}},
        {"$set": {"running": True}},
    )
    if result.modified_count == 0:
        logger.info("Job already running; skipping")
        return

    try:
        control = await db.job_control.find_one({"_id": JOB_ID})
        if control and control.get("status") == "stop":
            logger.info("Job status is stop; skipping send")
            return

        # TEMPORARY dateKey namespace so prior yesterday logs do not block catch-up.
        today = datetime.now(GREECE_TZ).date()
        date_key = f"temp-since-2026-09-07-{today.isoformat()}"

        existing = await db.email_log.find_one(
            {
                "type": EMAIL_LOG_TYPE,
                "dateKey": date_key,
                "recipientEmail": RECIPIENT,
                "emailSent": True,
            }
        )
        if existing:
            logger.info("Already sent incomplete cashflows email for %s", date_key)
            return

        payload = await fetch_all_incomplete_for_period(prop_db)
        date_label = _period_label()
        subject = f"Incomplete Cashflows — {date_label}"
        body = format_incomplete_cashflows_email(payload, date_label)

        await asyncio.to_thread(send_email_v2, [RECIPIENT], subject, body)

        await db.email_log.update_one(
            {
                "type": EMAIL_LOG_TYPE,
                "dateKey": date_key,
                "recipientEmail": RECIPIENT,
            },
            {
                "$set": {
                    "emailSent": True,
                    "emailSentAt": datetime.utcnow(),
                    "userCount": payload.get("total") or 0,
                    "totals": payload.get("totals") or {},
                }
            },
            upsert=True,
        )
        logger.info(
            "Sent incomplete cashflows email to %s | users=%s | totals=%s",
            RECIPIENT,
            payload.get("total"),
            payload.get("totals"),
        )
    except Exception as exc:
        logger.error("Incomplete cashflows email job failed: %s", exc)
        traceback.print_exc()
    finally:
        await db.job_control.update_one(
            {"_id": JOB_ID},
            {"$set": {"running": False}},
        )
        logger.info("Job running flag cleared")


def start_incomplete_cashflows_email_scheduler(prop_db, db):
    scheduler.add_job(
        send_incomplete_cashflows_daily_email,
        CronTrigger(
            hour=10,
            minute=46,
            timezone=pytz.timezone("Europe/Athens"),
        ),
        args=[prop_db, db],
        id="send_incomplete_cashflows_daily_email_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    if not scheduler.running:
        scheduler.start()
    logger.info(
        "Incomplete cashflows daily email scheduled (10:00 Europe/Athens) → %s",
        RECIPIENT,
    )
