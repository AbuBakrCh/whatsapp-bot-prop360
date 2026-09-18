"""Daily job to sync cashflow property/contact reference labels."""

from __future__ import annotations

import logging
import traceback
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.sync_cashflow_refs import JOB_ID, sync_cashflow_refs

scheduler = AsyncIOScheduler()

logger = logging.getLogger("sync_cashflow_refs_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


async def run_sync_cashflow_refs_job(prop_db, db):
    logger.info("Starting sync cashflow refs job")

    await db.job_control.update_one(
        {"_id": JOB_ID},
        {"$setOnInsert": {"status": "start"}},
        upsert=True,
    )

    result = await db.job_control.update_one(
        {"_id": JOB_ID, "running": {"$ne": True}},
        {
            "$set": {
                "running": True,
                "startedAt": datetime.utcnow(),
                "lastError": None,
            }
        },
    )
    if result.modified_count == 0:
        logger.info("Job already running; skipping")
        return

    stats = None
    error_msg = None
    try:
        control = await db.job_control.find_one({"_id": JOB_ID})
        if control and control.get("status") == "stop":
            logger.info("Job status is stop; skipping run")
            stats = {"skipped": True, "reason": "stop"}
            return

        stats = await sync_cashflow_refs(prop_db, db)
        logger.info("Sync cashflow refs finished | %s", stats)
    except Exception as exc:
        error_msg = str(exc)
        logger.error("Sync cashflow refs job failed: %s", exc)
        traceback.print_exc()
    finally:
        finished_at = datetime.utcnow()
        update = {
            "running": False,
            "finishedAt": finished_at,
        }
        if stats is not None:
            update["lastResult"] = stats
        if error_msg is not None:
            update["lastError"] = error_msg
        await db.job_control.update_one(
            {"_id": JOB_ID},
            {"$set": update},
        )
        logger.info("Job running flag cleared")


def start_sync_cashflow_refs_scheduler(prop_db, db):
    scheduler.add_job(
        run_sync_cashflow_refs_job,
        CronTrigger(
            hour=8,
            minute=55,
            timezone=pytz.timezone("Europe/Athens"),
        ),
        args=[prop_db, db],
        id="sync_cashflow_refs_daily_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    if not scheduler.running:
        scheduler.start()
    logger.info("Sync cashflow refs scheduled (08:55 Europe/Athens)")
