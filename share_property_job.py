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
    source = filter_doc.get("source", "spitogatos")

    # -------------------------
    # Base query
    # -------------------------
    query = {
        "indicator": "properties",
        "status": "active"
    }

    # -------------------------
    # Source handling
    # -------------------------
    if source == "spitogatos":
        query["spitogatos"] = {"$exists": True}

    elif source == "prop360":
        query["spitogatos"] = {"$exists": False}

    # both → no restriction

    # -------------------------
    # Incremental processing
    # -------------------------
    last_shared = filter_doc.get("lastSharedAt")
    if last_shared:
        query["metadata.createdAt"] = {"$gt": last_shared}

    # -------------------------
    # Field-level conditions (AND across fields)
    # -------------------------
    and_conditions = []

    purpose = filter_doc.get("purpose")
    category = filter_doc.get("category")
    area = filter_doc.get("area")

    # -------------------------
    # PURPOSE
    # -------------------------
    if purpose:
        conds = []

        if source in ["spitogatos", "both"]:
            conds.append({"spitogatos.purpose": purpose})

        if source in ["prop360", "both"]:
            conds.append({
                "data.field-1741536151680-7tt7lah7d": {
                    "$regex": f"^{purpose}$",
                    "$options": "i"
                }
            })

        and_conditions.append({"$or": conds})

    # -------------------------
    # CATEGORY
    # -------------------------
    category_map = {
        "home": "Residential",
        "commercial": "Commercial",
        "land": "Project",
        "other": "Project",
        "new-development": "Project",
        "student-housing": "Project"
    }

    category_prop = category_map.get(category)

    if category:
        conds = []

        if source in ["spitogatos", "both"]:
            conds.append({"spitogatos.category": category})

        if source in ["prop360", "both"] and category_prop:
            conds.append({
                "data.field-1741536164363-ai4m3m5r3": category_prop
            })

        and_conditions.append({"$or": conds})

    # -------------------------
    # AREA
    # -------------------------
    if area and area.strip():
        conds = []

        normalized_area = area.replace("-", " ")

        if source in ["spitogatos", "both"]:
            conds.append({"spitogatos.area": area})

        if source in ["prop360", "both"]:
            conds.append({
                "data.field-1744021392093-03a295o25": {
                    "$regex": normalized_area,
                    "$options": "i"
                }
            })

        and_conditions.append({"$or": conds})

    # -------------------------
    # PRICE
    # -------------------------
    price_filter = filter_doc.get("price", {})
    if price_filter.get("min") is not None or price_filter.get("max") is not None:

        spit_cond = {}
        prop_cond = {}

        if price_filter.get("min") is not None:
            spit_cond["$gte"] = price_filter["min"]
            prop_cond["$gte"] = price_filter["min"]

        if price_filter.get("max") is not None:
            spit_cond["$lte"] = price_filter["max"]
            prop_cond["$lte"] = price_filter["max"]

        conds = []

        if source in ["spitogatos", "both"]:
            conds.append({"spitogatos.price": spit_cond})
            conds.append({"spitogatos.price": {"$exists": False}})

        if source in ["prop360", "both"]:
            conds.append({"data.field-1741536272085-yi74oirib": prop_cond})
            conds.append({"data.field-1741536272085-yi74oirib": {"$exists": False}})

        and_conditions.append({"$or": conds})

    # -------------------------
    # SURFACE
    # -------------------------
    surface_filter = filter_doc.get("surface", {})
    if surface_filter.get("min") is not None or surface_filter.get("max") is not None:

        spit_cond = {}
        prop_cond = {}

        if surface_filter.get("min") is not None:
            spit_cond["$gte"] = surface_filter["min"]
            prop_cond["$gte"] = surface_filter["min"]

        if surface_filter.get("max") is not None:
            spit_cond["$lte"] = surface_filter["max"]
            prop_cond["$lte"] = surface_filter["max"]

        conds = []

        if source in ["spitogatos", "both"]:
            conds.append({"spitogatos.surface": spit_cond})
            conds.append({"spitogatos.surface": {"$exists": False}})

        if source in ["prop360", "both"]:
            conds.append({"data.field-1741783862474-a5ordcxh2": prop_cond})
            conds.append({"data.field-1741783862474-a5ordcxh2": {"$exists": False}})

        and_conditions.append({"$or": conds})

    # -------------------------
    # Attach conditions
    # -------------------------
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


async def share_property_job(db, prop_db):
    logger.info("Starting property match job")

    control = await db.job_control.find_one(
        {"_id": "share_property_job"}
    )

    if not control or control.get("status") != "enable":
        logger.info("[Share Property Job] Skipped — job is disabled")
        return

    filters_col = prop_db.propertyfilters
    properties_col = prop_db.formdatas

    cursor = filters_col.find({})

    processed = 0
    matched_clients = 0
    properties_processed = 0

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

        properties_processed += len(properties)
        matched_clients += 1
        processed += 1

    logger.info(
        "Property match job completed | processed=%s | matched_clients=%s | properties_processed=%s",
        processed,
        matched_clients,
        properties_processed
    )

def start_property_match_scheduler(db, prop_db):
    scheduler.add_job(
        share_property_job,
        CronTrigger(hour=11, minute=0, timezone=ZoneInfo("Europe/Athens")),
        args=[db, prop_db],
        id="share_property_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()