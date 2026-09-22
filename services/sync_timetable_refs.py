"""Sync denormalized property/contact labels on timetable (activity) formdatas."""

from __future__ import annotations

from services.ledger_report import (
    ACTIVITY_CLIENT_FIELDS,
    ACTIVITY_INDICATOR,
    ACTIVITY_PROPERTY_FIELDS,
    PROPERTY_TITLE_FIELD,
)
from services.sync_cashflow_refs import (
    CONTACT_NAME_FIELD,
    sync_piped_refs,
)

JOB_ID = "sync_timetable_refs_job"

REF_SPECS = tuple(
    (field, "properties", PROPERTY_TITLE_FIELD)
    for field in ACTIVITY_PROPERTY_FIELDS
) + tuple(
    (field, "contacts", CONTACT_NAME_FIELD) for field in ACTIVITY_CLIENT_FIELDS
)


async def sync_timetable_refs(prop_db, db) -> dict[str, int]:
    """
    Rewrite timetable property/contact Label|pid fields from live titles.
    Clear the field when the source is missing or inactive.
    """
    return await sync_piped_refs(
        prop_db,
        db,
        job_id=JOB_ID,
        indicator=ACTIVITY_INDICATOR,
        ref_specs=REF_SPECS,
        log_name="sync_timetable_refs",
    )
