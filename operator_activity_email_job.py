"""Daily Operator Active Time report email (19:00 Europe/Athens)."""

from __future__ import annotations

import asyncio
import html
import logging
import traceback
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2
from services.operator_activity import (
    GREECE_TZ,
    format_duration,
    get_operator_activity_report,
)

scheduler = AsyncIOScheduler()

logger = logging.getLogger("operator_activity_email_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

JOB_ID = "operator_activity_email_job"
EMAIL_LOG_TYPE = "operator-activity-daily"
RECIPIENT = "ka@investgreece.gr"


def format_operator_activity_email(payload: dict) -> str:
    period = payload.get("period") or {}
    summary = payload.get("summary") or {}
    rankings = payload.get("rankings") or []
    date_label = html.escape(str(period.get("label") or ""))

    total_duration = html.escape(
        str(summary.get("totalDurationLabel") or format_duration(0))
    )
    with_activity = int(summary.get("operatorsWithActivity") or 0)
    total_ops = int(summary.get("totalOperators") or 0)
    top = summary.get("topOperator") or {}
    top_name = html.escape(str(top.get("displayName") or "—"))
    top_duration = html.escape(str(top.get("durationLabel") or "—"))

    if not rankings:
        table_html = (
            '<p style="margin:16px 0;color:#555;">'
            "No active operators found for this day."
            "</p>"
        )
    else:
        row_html = []
        for row in rankings:
            name = html.escape(str(row.get("displayName") or "Unknown"))
            email = html.escape(str(row.get("email") or ""))
            rank = int(row.get("rank") or 0)
            duration = html.escape(str(row.get("durationLabel") or "0m"))
            logins = int(row.get("loginCount") or 0)
            bar_pct = max(0.0, min(100.0, float(row.get("barPct") or 0)))
            user_cell = name
            if email:
                user_cell += (
                    f'<br><span style="color:#777;font-size:12px;">{email}</span>'
                )
            bar = (
                '<div style="background:#e5e7eb;border-radius:4px;height:8px;'
                'min-width:80px;">'
                f'<div style="background:#1f4e79;border-radius:4px;height:8px;'
                f'width:{bar_pct}%;"></div></div>'
            )
            row_html.append(
                "<tr>"
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;'
                f'text-align:center;font-weight:600;">{rank}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;">'
                f"{user_cell}</td>"
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;'
                f'text-align:right;white-space:nowrap;">{duration}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;'
                f'width:140px;">{bar}</td>'
                f'<td style="padding:10px 12px;border-bottom:1px solid #eee;'
                f'text-align:right;">{logins}</td>'
                "</tr>"
            )

        table_html = f"""
        <table style="width:100%;border-collapse:collapse;margin-top:16px;font-size:14px;">
          <thead>
            <tr style="background:#f3f4f6;text-align:left;">
              <th style="padding:10px 12px;text-align:center;width:48px;">#</th>
              <th style="padding:10px 12px;">Operator</th>
              <th style="padding:10px 12px;text-align:right;">Active</th>
              <th style="padding:10px 12px;"> </th>
              <th style="padding:10px 12px;text-align:right;">Logins</th>
            </tr>
          </thead>
          <tbody>
            {''.join(row_html)}
          </tbody>
        </table>
        """

    return f"""<!DOCTYPE html>
<html>
<body style="font-family:Arial,Helvetica,sans-serif;color:#333;line-height:1.5;margin:0;padding:20px;background:#f4f4f4;">
  <div style="max-width:760px;margin:0 auto;background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
    <div style="background:#1f4e79;color:#ffffff;padding:20px 24px;">
      <h1 style="margin:0;font-size:20px;font-weight:600;">Operator Active Time</h1>
      <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">
        {date_label} · Europe/Athens
      </p>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 16px;color:#555;font-size:13px;">
        Timezone shown in Greece timezone.
      </p>
      <table style="width:100%;border-collapse:separate;border-spacing:8px 0;margin:0 -8px 8px;">
        <tr>
          <td style="background:#f3f4f6;border-radius:8px;padding:14px 16px;width:33%;vertical-align:top;">
            <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.04em;">Total active</div>
            <div style="font-size:22px;font-weight:700;color:#1f4e79;margin-top:4px;">{total_duration}</div>
          </td>
          <td style="background:#f3f4f6;border-radius:8px;padding:14px 16px;width:33%;vertical-align:top;">
            <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.04em;">With activity</div>
            <div style="font-size:22px;font-weight:700;color:#1f4e79;margin-top:4px;">{with_activity}<span style="font-size:14px;font-weight:500;color:#6b7280;"> / {total_ops}</span></div>
          </td>
          <td style="background:#f3f4f6;border-radius:8px;padding:14px 16px;width:33%;vertical-align:top;">
            <div style="font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:0.04em;">Top operator</div>
            <div style="font-size:16px;font-weight:700;color:#1f4e79;margin-top:4px;">{top_name}</div>
            <div style="font-size:13px;color:#555;">{top_duration}</div>
          </td>
        </tr>
      </table>
      {table_html}
    </div>
  </div>
</body>
</html>"""


async def send_operator_activity_daily_email(prop_db, db):
    logger.info("Starting operator activity daily email job")

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

        today_greece = datetime.now(GREECE_TZ).date()
        date_key = today_greece.isoformat()

        existing = await db.email_log.find_one(
            {
                "type": EMAIL_LOG_TYPE,
                "dateKey": date_key,
                "recipientEmail": RECIPIENT,
                "emailSent": True,
            }
        )
        if existing:
            logger.info("Already sent operator activity email for %s", date_key)
            return

        payload = await get_operator_activity_report(
            prop_db,
            view="day",
            date_str=date_key,
            end_at_now=True,
        )
        period_label = (payload.get("period") or {}).get("label") or date_key
        subject = f"Operator Active Time — {period_label}"
        body = format_operator_activity_email(payload)

        await asyncio.to_thread(send_email_v2, [RECIPIENT], subject, body)

        summary = payload.get("summary") or {}
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
                    "operatorCount": summary.get("totalOperators") or 0,
                    "totalActiveMinutes": summary.get("totalActiveMinutes") or 0,
                    "operatorsWithActivity": summary.get("operatorsWithActivity") or 0,
                }
            },
            upsert=True,
        )
        logger.info(
            "Sent operator activity email to %s | operators=%s | minutes=%s",
            RECIPIENT,
            summary.get("totalOperators"),
            summary.get("totalActiveMinutes"),
        )
    except Exception as exc:
        logger.error("Operator activity email job failed: %s", exc)
        traceback.print_exc()
    finally:
        await db.job_control.update_one(
            {"_id": JOB_ID},
            {"$set": {"running": False}},
        )
        logger.info("Job running flag cleared")


def start_operator_activity_email_scheduler(prop_db, db):
    scheduler.add_job(
        send_operator_activity_daily_email,
        CronTrigger(
            hour=19,
            minute=0,
            timezone=pytz.timezone("Europe/Athens"),
        ),
        args=[prop_db, db],
        id="send_operator_activity_daily_email_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    if not scheduler.running:
        scheduler.start()
    logger.info(
        "Operator activity daily email scheduled (19:00 Europe/Athens) → %s",
        RECIPIENT,
    )
