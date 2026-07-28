"""Browse-assist Accrual ↔ Payment matching by equal amount within Greece periods."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timezone
from math import ceil
from typing import Any

from services.ledger_report import (
    CASHFLOW_FORM_URL_TEMPLATE,
    CASHFLOW_INDICATOR,
    FIELD_AMOUNT,
    FIELD_CATEGORY,
    FIELD_CASHFLOW_CONTACT,
    FIELD_DESCRIPTION,
    FIELD_PROPERTY,
    GREECE_TZ,
    format_date,
    parse_amount,
)

FIELD_DOCUMENT_TYPE = "field-1758699529035-xp7lumgx5"
FIELD_MATCHING_NUMBER = "field-1779680035201-hbd3ywbb1"

CATEGORY_ACCRUAL = "Accrual"
CATEGORY_PAYMENT = "Payment"

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

CASHFLOW_PROJECTION = {
    "_id": 1,
    "metadata.createdAt": 1,
    f"data.{FIELD_AMOUNT}": 1,
    f"data.{FIELD_DOCUMENT_TYPE}": 1,
    f"data.{FIELD_DESCRIPTION}": 1,
    f"data.{FIELD_CASHFLOW_CONTACT}": 1,
    f"data.{FIELD_PROPERTY}": 1,
    f"data.{FIELD_MATCHING_NUMBER}": 1,
}


def greece_period_to_utc(start: date, end: date) -> tuple[datetime, datetime]:
    """Convert Greece calendar day bounds to UTC datetimes for metadata.createdAt."""
    if end < start:
        raise ValueError("End date must be on or after start date.")
    start_greece = datetime.combine(start, time.min, tzinfo=GREECE_TZ)
    end_greece = datetime.combine(end, time.max, tzinfo=GREECE_TZ)
    return (
        start_greece.astimezone(timezone.utc),
        end_greece.astimezone(timezone.utc),
    )


def parse_yyyy_mm_dd(value: str, field_name: str) -> date:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required (YYYY-MM-DD).")
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD.") from exc


def round_amount_cents(amount: float) -> float:
    return round(float(amount), 2)


def amount_string_variants(raw_value: Any, rounded: float) -> set[str]:
    variants: set[str] = set()
    if raw_value is not None:
        text = str(raw_value).strip()
        if text:
            variants.add(text)
    dotted = f"{rounded:.2f}"
    variants.add(dotted)
    variants.add(dotted.replace(".", ","))
    if rounded == int(rounded):
        variants.add(str(int(rounded)))
    return variants


def _label_from_piped(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if "|" in text:
        return text.rsplit("|", 1)[0].strip()
    return text


def serialize_cashflow_row(doc: dict) -> dict[str, Any]:
    data = doc.get("data") or {}
    cashflow_id = str(doc.get("_id", ""))
    created_at = (doc.get("metadata") or {}).get("createdAt")
    raw_amount = data.get(FIELD_AMOUNT)
    amount = round_amount_cents(parse_amount(raw_amount))
    return {
        "id": cashflow_id,
        "amount": amount,
        "createdAt": format_date(created_at),
        "documentType": str(data.get(FIELD_DOCUMENT_TYPE) or "").strip(),
        "description": str(data.get(FIELD_DESCRIPTION) or "").strip(),
        "contact": _label_from_piped(data.get(FIELD_CASHFLOW_CONTACT)),
        "property": _label_from_piped(data.get(FIELD_PROPERTY)),
        "matchingNumber": str(data.get(FIELD_MATCHING_NUMBER) or "").strip(),
        "url": (
            CASHFLOW_FORM_URL_TEMPLATE.format(cashflow_id=cashflow_id)
            if cashflow_id
            else ""
        ),
        "_raw_amount": raw_amount,
    }


def _strip_internal(row: dict) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _period_payload(start: date, end: date) -> dict[str, str]:
    return {
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
        "timezone": "Europe/Athens",
    }


def _category_period_query(
    category: str,
    start_utc: datetime,
    end_utc: datetime,
) -> dict[str, Any]:
    return {
        "indicator": CASHFLOW_INDICATOR,
        "status": "active",
        "metadata.createdAt": {"$gte": start_utc, "$lte": end_utc},
        f"data.{FIELD_CATEGORY}": category,
    }


async def fetch_accrual_payment_matches(
    db,
    accrual_start_date: str,
    accrual_end_date: str,
    payment_start_date: str,
    payment_end_date: str,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    accrual_start = parse_yyyy_mm_dd(accrual_start_date, "accrualStartDate")
    accrual_end = parse_yyyy_mm_dd(accrual_end_date, "accrualEndDate")
    payment_start = parse_yyyy_mm_dd(payment_start_date, "paymentStartDate")
    payment_end = parse_yyyy_mm_dd(payment_end_date, "paymentEndDate")

    page = max(1, int(page))
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))

    accrual_start_utc, accrual_end_utc = greece_period_to_utc(accrual_start, accrual_end)
    payment_start_utc, payment_end_utc = greece_period_to_utc(payment_start, payment_end)

    accrual_query = _category_period_query(
        CATEGORY_ACCRUAL, accrual_start_utc, accrual_end_utc
    )
    total_count = await db.formdatas.count_documents(accrual_query)
    total_pages = max(1, ceil(total_count / page_size)) if total_count else 1
    page = min(page, total_pages)
    skip = (page - 1) * page_size

    accrual_docs = (
        await db.formdatas.find(accrual_query, CASHFLOW_PROJECTION)
        .sort("metadata.createdAt", -1)
        .skip(skip)
        .limit(page_size)
        .to_list(length=page_size)
    )
    accruals = [serialize_cashflow_row(doc) for doc in accrual_docs]

    amount_variants: set[str] = set()
    for accrual in accruals:
        amount_variants |= amount_string_variants(
            accrual.get("_raw_amount"), accrual["amount"]
        )

    payments: list[dict] = []
    if accruals and amount_variants:
        payment_query = {
            **_category_period_query(
                CATEGORY_PAYMENT, payment_start_utc, payment_end_utc
            ),
            f"data.{FIELD_AMOUNT}": {"$in": list(amount_variants)},
        }
        payment_docs = await db.formdatas.find(
            payment_query, CASHFLOW_PROJECTION
        ).to_list(length=None)
        payments = [serialize_cashflow_row(doc) for doc in payment_docs]

    payments_by_amount: dict[float, list[dict]] = defaultdict(list)
    for payment in payments:
        payments_by_amount[payment["amount"]].append(payment)

    result_accruals: list[dict] = []
    with_candidates = 0
    without_candidates = 0

    for accrual in accruals:
        candidates = [
            _strip_internal(p) for p in payments_by_amount.get(accrual["amount"], [])
        ]
        if candidates:
            with_candidates += 1
        else:
            without_candidates += 1
        result_accruals.append(
            {
                **_strip_internal(accrual),
                "candidates": candidates,
            }
        )

    return {
        "period": {
            "accruals": _period_payload(accrual_start, accrual_end),
            "payments": _period_payload(payment_start, payment_end),
        },
        "accruals": result_accruals,
        "pagination": {
            "page": page,
            "pageSize": page_size,
            "totalCount": total_count,
            "totalPages": total_pages,
        },
        "summary": {
            "accrualCount": total_count,
            "paymentCount": len(payments),
            "withCandidates": with_candidates,
            "withoutCandidates": without_candidates,
        },
    }
