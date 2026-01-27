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

def _is_job_enabled() -> bool:
    return os.getenv("TRANSFER_OWNERSHIP_ENABLED", "true").lower() == "true"

async def transfer_ownership(prop_db):
    """
    Task: transfer_ownership
    - Finds all documents updated after 15 Dec 2025
    - Adds original merchantId to sharedWithMerchants
    - Sets new owner
    - Sets new merchantId
    - Duplicate executions are safe
    """
    if not _is_job_enabled():
        logger.info("transfer_ownership is disabled. Skipping execution.")
        return {"skipped": True}

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
    users_col = prop_db.users

    update_pipeline = [
        {
            "$set": {
                "metadata.ownershipTransferred": True,
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
            "$or": [
                {"metadata.ownershipTransferred": {"$exists": False}},
                {"metadata.ownershipTransferred": False}
            ]
        },
        update_pipeline
    )

    logger.info(
        "Transfer completed | matched=%s | modified=%s",
        result.matched_count,
        result.modified_count,
    )


    # --------------------------------------------------
    # Step 2: Custom text prefix for custom-wyey07pb7
    # --------------------------------------------------

    logger.info("Processing custom-wyey07pb7 text updates")
    cursor = formdatas_col.find(
        {
            "metadata.createdAt": {"$gt": CUTOFF_DATE},
            "metadata.ownershipTransferred": {"$eq": True},
            "indicator": "custom-wyey07pb7",
            "owner": NEW_OWNER_ID,
            "merchantId": NEW_MERCHANT_ID,
            "metadata.customPrefixAdded": {"$ne": True}
        }
    )
    updated_custom_docs = 0

    async for doc in cursor:
        prev_owner = doc["metadata"]["previousOwner"]

        user = await users_col.find_one(
            {"firebaseId": prev_owner},
            {"displayName": 1, "email": 1}
        )

        if not user:
            logger.warning(
                "Previous merchant user not found | firebaseId=%s",
                prev_owner
            )
            continue

        name = user.get("displayName", "Unknown")
        email = user.get("email", "unknown@email")

        prefix = f"{name} ({email}) wrote:\n\n"

        old_text = (
            doc.get("data", {})
            .get("field-1760213212062-ask5v2fuy", "")
        )

        if old_text.startswith(prefix):
            continue

        await formdatas_col.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "data.field-1760213212062-ask5v2fuy": prefix + old_text,
                    "metadata.customPrefixAdded": True
                }
            }
        )

        updated_custom_docs += 1

    logger.info(
        "Custom text updates completed | updated=%s",
        updated_custom_docs
    )

    return {
        "matched": result.matched_count,
        "modified": result.modified_count,
        "updated_custom_docs": updated_custom_docs
    }

def start_scheduler(prop_db):
    """
    Starts the background scheduler.
    Call this once during FastAPI startup.
    """
    scheduler.add_job(
        transfer_ownership,
        CronTrigger(minute="*/20"),
        args=[prop_db],  # every 20 minutes
        id="transfer_ownership_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,  # 5 minutes grace
    )
    scheduler.start()
