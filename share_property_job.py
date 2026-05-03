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

        data = p.get("data", {})

        title = data.get("field-1741536181001-wd8it2quy") or "Property"
        price = data.get("field-1741536272085-yi74oirib") or "-"
        area = data.get("field-1741783862474-a5ordcxh2") or "-"
        listing_type = data.get("field-1741536151680-7tt7lah7d") or "-"

        rows += f"""
        <tr>
          <td style="padding:40px 30px;">

            <!-- Details -->
            <div style="text-align:center;">
              <h3 style="margin:0; font-size:20px;">{title}</h3>

              <p style="margin:12px 0 6px; color:#333; font-size:14px;">
                <b>Price:</b> {price}
              </p>

              <p style="margin:6px 0; color:#333; font-size:14px;">
                <b>Surface:</b> {area} m²
              </p>

              <p style="margin:6px 0; color:#333; font-size:14px;">
                <b>Purpose:</b> {listing_type}
              </p>

              <div style="margin-top:18px;">
                <a href="{url}" style="background:#000; color:#fff; text-decoration:none; padding:10px 22px; border-radius:6px; font-size:13px;">
                  View Details
                </a>
              </div>
            </div>

          </td>
        </tr>

        <!-- Divider -->
        <tr>
          <td style="padding:0 30px;">
            <div style="height:1px; background:#eee;"></div>
          </td>
        </tr>
        """

    body = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0; padding:0; background:#f4f4f4; font-family:Arial, sans-serif;">

    <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;">
    <tr>
    <td align="center">

    <table width="650" cellpadding="0" cellspacing="0" style="background:#ffffff; border-radius:12px; overflow:hidden;">

      <!-- Header -->
      <tr>
        <td style="background:#c9a54c; color:white; text-align:center; padding:35px;">
          <h2 style="margin:0;">PROP 360</h2>
          <p style="margin:8px 0 0;">New Matching Properties</p>
        </td>
      </tr>

      {rows}

      <!-- CTA -->
      <tr>
        <td align="center" style="padding:50px 30px;">
          <a href="https://prop360.pro" style="background:#c9a54c; color:white; text-decoration:none; padding:18px 45px; border-radius:6px; font-weight:bold;">
            VIEW ALL PROPERTIES
          </a>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="background:#fafafa; text-align:center; padding:25px; font-size:12px; color:#888;">
          © 2026 Prop 360<br>
          Unsubscribe anytime (contact: ka@investgreece.gr)
        </td>
      </tr>

    </table>

    </td>
    </tr>
    </table>

    </body>
    </html>
    """

    send_email_v2([email], subject, body, None, bcc=["ka@investgreece.gr"])


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

        properties_cursor = properties_col.find(query).sort("metadata.createdAt", 1).limit(5)

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
        #CronTrigger(minute="*", timezone=ZoneInfo("Europe/Athens")),
        args=[db, prop_db],
        id="share_property_job",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.start()