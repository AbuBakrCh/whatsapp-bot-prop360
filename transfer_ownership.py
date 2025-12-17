import logging
import os
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Create scheduler instance
scheduler = AsyncIOScheduler()

# --------------------------------------------------
# Logger setup
# --------------------------------------------------

logger = logging.getLogger("transfer_ownership")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def _get_cutoff_date() -> datetime:
    cutoff_str = os.getenv("TRANSFER_CUTOFF_DATE")
    if not cutoff_str:
        raise ValueError("TRANSFER_CUTOFF_DATE env var is missing")

    # Expecting ISO format: 2025-12-15T00:00:00Z
    return datetime.fromisoformat(cutoff_str.replace("Z", "+00:00"))

async def transfer_ownership(prop_db):
    """
    Task: transfer_ownership
    - Finds all documents updated after 15 Dec 2025
    - Adds original merchantId to sharedWithMerchants
    - Sets new owner
    - Sets new merchantId
    - Duplicate executions are safe
    """
    logger.info("Starting transfer_ownership task")

    NEW_OWNER_ID = os.getenv("TRANSFER_NEW_OWNER_ID")
    NEW_MERCHANT_ID = os.getenv("TRANSFER_NEW_MERCHANT_ID")

    if not NEW_OWNER_ID:
        raise ValueError("TRANSFER_NEW_OWNER_ID env var is missing")

    if not NEW_MERCHANT_ID:
        raise ValueError("TRANSFER_NEW_MERCHANT_ID env var is missing")

    CUTOFF_DATE = _get_cutoff_date()

    logger.info(
        "Transfer config loaded | cutoff=%s | new_owner=%s | new_merchant=%s",
        CUTOFF_DATE.isoformat(),
        NEW_OWNER_ID,
        NEW_MERCHANT_ID,
    )

    formdatas_col = prop_db.formdatas

    update_pipeline = [
        {
            "$set": {
                "metadata.previousOwner": {
                    "$ifNull": ["$metadata.previousOwner", "$owner"]
                },
                "metadata.previousMerchantId": {
                    "$ifNull": ["$metadata.previousMerchantId", "$merchantId"]
                },
                "metadata.ownershipTransferredAt": {
                    "$ifNull": ["$metadata.ownershipTransferredAt", "$$NOW"]
                }
            }
        },
        {
            "$set": {
                "sharedWithMerchants": {
                    "$setUnion": [
                        {"$ifNull": ["$sharedWithMerchants", []]},
                        ["$merchantId"]
                    ]
                }
            }
        },
        {
            "$set": {
                "owner": NEW_OWNER_ID,
                "merchantId": NEW_MERCHANT_ID
            }
        }
    ]

    logger.info("Executing ownership transfer update")

    result = await formdatas_col.update_many(
        {
            "metadata.createdAt": {"$gt": CUTOFF_DATE},
            "indicator": {"$in": ["contacts", "properties", "custom-wyey07pb7"]},
            "owner": {"$ne": NEW_OWNER_ID},
            "merchantId": {"$ne": NEW_MERCHANT_ID}
        },
        update_pipeline
    )

    logger.info(
        "Transfer completed | matched=%s | modified=%s",
        result.matched_count,
        result.modified_count,
    )

    return {
        "matched": result.matched_count,
        "modified": result.modified_count
    }

def start_scheduler(prop_db):
    """
    Starts the background scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        transfer_ownership,
        CronTrigger(minute="*/10"),
        args=[prop_db],  # every 10 minutes
        id="transfer_ownership_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,  # 5 minutes grace
    )
    scheduler.start()
