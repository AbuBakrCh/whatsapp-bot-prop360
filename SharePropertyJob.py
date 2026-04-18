import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("share_property_job")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def build_query(filter_doc):
    query = {
        "indicator": "properties",
        "status": "active",
        "spitogatos": {"$exists": True}
    }

    last_shared = filter_doc.get("lastSharedAt")
    if last_shared:
        query["metadata.createdAt"] = {"$gt": last_shared}

    # -------------------------
    # Simple filters
    # -------------------------
    if filter_doc.get("purpose"):
        query["spitogatos.purpose"] = filter_doc["purpose"]

    if filter_doc.get("category"):
        query["spitogatos.category"] = filter_doc["category"]

    if filter_doc.get("area"):
        query["spitogatos.area"] = filter_doc["area"]

    # -------------------------
    # Advanced filters (price/surface)
    # -------------------------
    and_conditions = []

    # Price
    price_filter = filter_doc.get("price", {})
    if price_filter.get("min") is not None or price_filter.get("max") is not None:
        price_condition = {}

        if price_filter.get("min") is not None:
            price_condition["$gte"] = price_filter["min"]

        if price_filter.get("max") is not None:
            price_condition["$lte"] = price_filter["max"]

        and_conditions.append({
            "$or": [
                {"spitogatos.price": price_condition},
                {"spitogatos.price": {"$exists": False}}
            ]
        })

    # Surface
    surface_filter = filter_doc.get("surface", {})
    if surface_filter.get("min") is not None or surface_filter.get("max") is not None:
        surface_condition = {}

        if surface_filter.get("min") is not None:
            surface_condition["$gte"] = surface_filter["min"]

        if surface_filter.get("max") is not None:
            surface_condition["$lte"] = surface_filter["max"]

        and_conditions.append({
            "$or": [
                {"spitogatos.surface": surface_condition},
                {"spitogatos.surface": {"$exists": False}}
            ]
        })

    # Attach conditions if any
    if and_conditions:
        query["$and"] = and_conditions

    return query

async def send_properties_email(email, properties):
    subject = f"Prop 360 - New Matching Properties ({len(properties)})"

    rows = ""
    for p in properties:
        property_id = str(p.get("_id"))
        url = f"https://prop360.pro/en/dashboard/forms/properties/{property_id}"

        title = (
                    p.get("data", {})
                    .get("field-1741536181001-wd8it2quy")
                ) or "Property"

        rows += f"""
        <p>
            <b>{title}</b><br>
            <a href="{url}">{url}</a>
        </p>
        """

    body = f"""
    <html>
      <body>
        <p>Hi,</p>
        <p>We found new properties matching your criteria:</p>
        {rows}
        <p>Best regards</p>
      </body>
    </html>
    """

    send_email_v2([email], subject, body, None)


async def share_property_job(prop_db):

    logger.info("Starting property match job")

    filters_col = prop_db.propertyfilters
    properties_col = prop_db.formdatas

    cursor = filters_col.find({})

    processed = 0
    matched_clients = 0

    async for filter_doc in cursor:

        email = filter_doc.get("clientEmail")
        if not email:
            continue

        query = build_query(filter_doc)

        properties_cursor = properties_col.find(query).sort("metadata.createdAt", 1)

        properties = []
        latest_created_at = None

        async for prop in properties_cursor:
            properties.append(prop)

            created_at = prop.get("metadata", {}).get("createdAt")
            if created_at:
                latest_created_at = created_at

        if not properties:
            continue

        # -------------------------
        # Send email
        # -------------------------
        await send_properties_email(email, properties)

        # -------------------------
        # Update checkpoint
        # -------------------------
        await filters_col.update_one(
            {"_id": filter_doc["_id"]},
            {
                "$set": {
                    "lastSharedAt": latest_created_at,
                    "lastSharedCount": len(properties),
                    "lastSharedAtUpdated": datetime.utcnow()
                }
            }
        )

        matched_clients += 1
        processed += 1

    logger.info(
        "Property match job completed | processed=%s | matched_clients=%s",
        processed,
        matched_clients
    )

def start_property_match_scheduler(prop_db):
    scheduler.add_job(
        share_property_job,
        CronTrigger(hour=11, minute=0, timezone=ZoneInfo("Europe/Athens")),
        args=[prop_db],
        id="share_property_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()