import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from services.commons import send_email_v2

scheduler = AsyncIOScheduler()

logger = logging.getLogger("share_property_job")
logger.setLevel(logging.INFO)

BATCH_SIZE = 5

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def _make_numeric_range_expr(field, min_val, max_val):
    """
    Cast a string field to double before comparing.
    Returns null (no match) if the field is empty, null, or non-numeric.
    """
    safe_cast = {
        "$convert": {
            "input": f"${field}",
            "to": "double",
            "onError": None,   # non-numeric strings → null → excluded from range
            "onNull": None     # missing/null fields → null → excluded from range
        }
    }

    comparisons = []
    if min_val is not None:
        comparisons.append({"$gte": [safe_cast, float(min_val)]})
    if max_val is not None:
        comparisons.append({"$lte": [safe_cast, float(max_val)]})

    expr = comparisons[0] if len(comparisons) == 1 else {"$and": comparisons}
    return {"$expr": expr}


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
            conds.append({"data.field-1741536164363-ai4m3m5r3": category_prop})
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
    # PRICE  (string fields → cast with $toDouble)
    # -------------------------
    price_filter = filter_doc.get("price", {})
    p_min = price_filter.get("min")
    p_max = price_filter.get("max")

    if p_min is not None or p_max is not None:
        conds = []
        if source in ["spitogatos", "both"]:
            conds.append(_make_numeric_range_expr("spitogatos.price", p_min, p_max))
        if source in ["prop360", "both"]:
            conds.append(_make_numeric_range_expr(
                "data.field-1741536272085-yi74oirib", p_min, p_max
            ))
        and_conditions.append({"$or": conds})

    # -------------------------
    # SURFACE  (string fields → cast with $toDouble)
    # -------------------------
    surface_filter = filter_doc.get("surface", {})
    s_min = surface_filter.get("min")
    s_max = surface_filter.get("max")

    if s_min is not None or s_max is not None:
        conds = []
        if source in ["spitogatos", "both"]:
            conds.append(_make_numeric_range_expr("spitogatos.surface", s_min, s_max))
        if source in ["prop360", "both"]:
            conds.append(_make_numeric_range_expr(
                "data.field-1741783862474-a5ordcxh2", s_min, s_max
            ))
        and_conditions.append({"$or": conds})

    # -------------------------
    # Attach conditions
    # -------------------------
    if and_conditions:
        query["$and"] = and_conditions

    return query

async def send_properties_email(email, properties):
    subject = f"Prop 360 - New Matching Properties ({len(properties)})"

    def img_tag(src, height="220px", radius_sides="all"):
        if radius_sides == "all":
            radius = "border-radius:8px;"
        elif radius_sides == "top":
            radius = "border-radius:8px 8px 0 0;"
        elif radius_sides == "bottom-right":
            radius = "border-radius:0 0 8px 0;"
        elif radius_sides == "top-right":
            radius = "border-radius:0 8px 0 0;"
        return (
            f'<div style="width:100%;height:{height};overflow:hidden;{radius}">'
            f'<img src="{src}" width="100%" height="100%" '
            f'style="display:block;object-fit:cover;object-position:center;">'
            f'</div>'
        )

    def build_image_html(img1, img2, img3):
        if img1 and img2 and img3:
            return f"""
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="55%" style="vertical-align:top; padding-right:4px;">
                  {img_tag(img1, height="300px", radius_sides="all")}
                </td>
                <td width="45%" style="vertical-align:top; padding-left:4px;">
                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td style="padding-bottom:4px;">
                        {img_tag(img2, height="148px", radius_sides="top-right")}
                      </td>
                    </tr>
                    <tr>
                      <td style="padding-top:0;">
                        {img_tag(img3, height="148px", radius_sides="bottom-right")}
                      </td>
                    </tr>
                  </table>
                </td>
              </tr>
            </table>
            """
        elif img1 and img2:
            return f"""
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="50%" style="padding-right:4px;">
                  {img_tag(img1, height="220px")}
                </td>
                <td width="50%" style="padding-left:4px;">
                  {img_tag(img2, height="220px")}
                </td>
              </tr>
            </table>
            """
        elif img1:
            return img_tag(img1, height="260px")
        return ""

    def badge(text, color="#c9a54c"):
        return (
            f'<span style="display:inline-block;background:{color}1a;color:{color};'
            f'border:1px solid {color}55;border-radius:20px;'
            f'padding:3px 12px;font-size:12px;font-weight:600;">{text}</span>'
        )

    rows = ""
    for p in properties:
        property_id = str(p.get("_id"))
        url = f"https://prop360.pro/en/dashboard/forms/properties/{property_id}"
        data = p.get("data", {})

        title = data.get("field-1741536181001-wd8it2quy") or "Property"
        price = data.get("field-1741536272085-yi74oirib") or "-"
        surface = data.get("field-1741783862474-a5ordcxh2") or "-"
        listing_type = data.get("field-1741536151680-7tt7lah7d") or "-"

        raw_images = data.get("field-1741536446663-7s5bcmilv") or []
        image_urls = [
            f"https://prop360.pro/api/image?key={img.get('key')}"
            for img in raw_images if img.get("key")
        ]

        img1 = image_urls[0] if len(image_urls) > 0 else None
        img2 = image_urls[1] if len(image_urls) > 1 else None
        img3 = image_urls[2] if len(image_urls) > 2 else None

        image_html = build_image_html(img1, img2, img3)

        purpose_badge = badge(listing_type)
        price_display = f"€{price}" if price != "-" else "-"

        rows += f"""
        <tr>
          <td style="padding:36px 32px 28px;">

            {image_html}

            <table width="100%" cellpadding="0" cellspacing="0" style="margin-top:20px;">
              <tr>
                <td>
                  <h3 style="margin:0 0 8px;font-size:18px;font-weight:700;color:#111;line-height:1.3;">{title}</h3>
                  <div style="margin-bottom:14px;">{purpose_badge}</div>

                  <table width="100%" cellpadding="0" cellspacing="0">
                    <tr>
                      <td width="50%" style="padding:10px 14px;background:#f9f7f2;border-radius:8px;">
                        <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;">Price</div>
                        <div style="font-size:17px;font-weight:700;color:#c9a54c;">{price_display}</div>
                      </td>
                      <td width="4px"></td>
                      <td width="50%" style="padding:10px 14px;background:#f9f7f2;border-radius:8px;">
                        <div style="font-size:11px;color:#999;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;">Surface</div>
                        <div style="font-size:17px;font-weight:700;color:#111;">{surface} m²</div>
                      </td>
                    </tr>
                  </table>

                  <div style="margin-top:18px;">
                    <a href="{url}"
                       style="display:inline-block;background:#111;color:#fff;text-decoration:none;
                              padding:11px 28px;border-radius:6px;font-size:13px;font-weight:600;
                              letter-spacing:0.3px;">
                      View Property →
                    </a>
                  </div>
                </td>
              </tr>
            </table>

          </td>
        </tr>

        <tr>
          <td style="padding:0 32px;">
            <div style="height:1px;background:#f0ece3;"></div>
          </td>
        </tr>
        """

    body = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#f2ede4;font-family:Arial,sans-serif;">

    <table width="100%" cellpadding="0" cellspacing="0" style="padding:48px 0;">
    <tr>
    <td align="center">

    <table width="640" cellpadding="0" cellspacing="0"
           style="background:#ffffff;border-radius:16px;overflow:hidden;
                  box-shadow:0 2px 20px rgba(0,0,0,0.07);">

      <!-- Header -->
      <tr>
        <td style="background:#111;color:white;text-align:center;padding:36px 32px;">
          <div style="font-size:11px;letter-spacing:3px;color:#c9a54c;margin-bottom:10px;font-weight:600;">REAL ESTATE</div>
          <div style="font-size:26px;font-weight:800;letter-spacing:1px;">PROP 360</div>
          <div style="margin-top:10px;font-size:14px;color:#aaa;">
            {len(properties)} new matching {'property' if len(properties) == 1 else 'properties'} found for you
          </div>
        </td>
      </tr>

      {rows}

      <!-- CTA -->
      <tr>
        <td align="center" style="padding:44px 32px 48px;">
          <a href="https://prop360.pro"
             style="display:inline-block;background:#c9a54c;color:white;text-decoration:none;
                    padding:16px 48px;border-radius:8px;font-weight:700;font-size:14px;
                    letter-spacing:0.5px;">
            VIEW ALL PROPERTIES
          </a>
        </td>
      </tr>

      <!-- Footer -->
      <tr>
        <td style="background:#f9f7f2;text-align:center;padding:24px 32px;
                   font-size:12px;color:#aaa;border-top:1px solid #f0ece3;">
          © 2026 Prop 360 &nbsp;·&nbsp;
          <a href="mailto:ka@investgreece.gr" style="color:#aaa;text-decoration:underline;">Unsubscribe</a>
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

        # -------------------------
        # Exclude already sent properties
        # -------------------------
        sent_ids = filter_doc.get("sentPropertyIds", [])

        if sent_ids:
            query["_id"] = {"$nin": sent_ids}

        # -------------------------
        # Fetch latest unseen properties
        # -------------------------
        properties_cursor = (
            properties_col
            .find(query)
            .sort("metadata.createdAt", -1)  # newest first
            .limit(BATCH_SIZE)
        )

        properties = []
        new_sent_ids = []

        async for prop in properties_cursor:
            properties.append(prop)
            new_sent_ids.append(prop["_id"])

        if not properties:
            continue

        # Optional: reverse so email shows oldest → newest (nicer UX)
        properties.reverse()

        # -------------------------
        # Send email
        # -------------------------
        await send_properties_email(email, properties)

        # -------------------------
        # Update filter document
        # -------------------------
        await filters_col.update_one(
            {"_id": filter_doc["_id"]},
            {
                "$push": {
                    "sentPropertyIds": {
                        "$each": new_sent_ids
                    }
                },
                "$set": {
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