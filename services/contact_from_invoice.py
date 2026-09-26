"""Create or match Prop360 contacts from uploaded invoice documents."""

from __future__ import annotations

import re
from typing import Any

from services.cashflow_document_extractor import (
    _extract_invoice_with_gemini,
    detect_file_type,
)

# Contact form field IDs
FIELD_FULL_NAME = "field-1741774547654-ngd30kdcz"
FIELD_CORPORATE_NAME = "field-1741774642959-g8l3j9yme"
FIELD_CONTACT_TYPE = "field-1741778607900-agpsipbof"
FIELD_TAX_NUMBER = "field-1741896055075-46mjpj2qf"
FIELD_EMAIL = "field-1741774690043-v7jylsjj2"
FIELD_PHONE = "field-1741778098761-w10f6zg6y"
FIELD_CONTACT_ROLE = "field-1741778629128-t3lelya4h"
FIELD_ADDRESS = "field-1741778662831-kkrtmk0rq"
FIELD_PROFESSION = "field-1751377453325-eif6cg1yp"

CONTACT_NOTES_CONSOLE = "Created by console"
CONTACT_TYPE_CORPORATE = "Corporate"
CONTACT_ROLE_SUPPLIER = "Supplier"

SUPPORTED_FILE_TYPES = frozenset({"pdf", "jpeg", "jpg", "png"})


def normalize_tax_id(value: str | None) -> str:
    """Strip to alphanumeric characters only (AFM / tax ID)."""
    if not value:
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", str(value)).upper()


def normalize_email(value: str | None) -> str:
    """Lowercase + trim for email comparison (not for storage)."""
    if not value:
        return ""
    return str(value).strip().casefold()


def normalize_phone(value: str | None) -> str:
    """Digits only for phone comparison (not for storage)."""
    if not value:
        return ""
    return re.sub(r"\D", "", str(value))


def _preserve_printed_text(value: Any) -> str:
    """Keep printed invoice text; only trim outer whitespace."""
    if value is None:
        return ""
    return str(value).strip()


# Common Greek/Latin company legal-form suffixes, restored when the model drops dots.
# Longer forms first so e.g. Ο.Ε.Ε. is not reduced to Ο.Ε.
_LEGAL_FORM_SUFFIX_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\bΟ\.?\s*Ε\.?\s*Ε\.?\s*$"), "Ο.Ε.Ε."),
    (re.compile(r"(?i)\bΙ\.?\s*Κ\.?\s*Ε\.?\s*$"), "Ι.Κ.Ε."),
    (re.compile(r"(?i)\bΕ\.?\s*Π\.?\s*Ε\.?\s*$"), "Ε.Π.Ε."),
    (re.compile(r"(?i)\bΟ\.?\s*Ε\.?\s*$"), "Ο.Ε."),
    (re.compile(r"(?i)\bΑ\.?\s*Ε\.?\s*$"), "Α.Ε."),
    (re.compile(r"(?i)\bO\.?\s*E\.?\s*$"), "O.E."),
    (re.compile(r"(?i)\bS\.?\s*A\.?\s*$"), "S.A."),
)

# Standalone single-letter initials missing a trailing period (e.g. "Α" → "Α.").
_INITIAL_WITHOUT_DOT = re.compile(
    r"(?<!\S)([A-Za-zΑ-Ωα-ωΆΈΉΊΌΎΏάέήίόύώ])(?!\.)(?=\s|$)"
)


def restore_issuer_name_punctuation(name: str) -> str:
    """
    Restore common abbreviation dots that OCR/LLM often strips from Greek company names.

    Example: "ΖΑΧΟΣ Α ΣΙΑ ΟΕ" → "ΖΑΧΟΣ Α. ΣΙΑ Ο.Ε."
    Names that already include correct punctuation are left unchanged.
    """
    text = _preserve_printed_text(name)
    if not text:
        return text

    for pattern, replacement in _LEGAL_FORM_SUFFIX_PATTERNS:
        if pattern.search(text):
            text = pattern.sub(replacement, text)
            break

    text = _INITIAL_WITHOUT_DOT.sub(r"\1.", text)
    return text


def _unwrap_field_value(value: Any) -> str:
    """Normalize Prop360 field values (including merge [[wrapped]] strings)."""
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        value = value[0]
    text = str(value).strip()
    if text.startswith("[[") and text.endswith("]]"):
        text = text[2:-2].strip()
    return text


def _tax_id_match_pattern(tax_id: str) -> str | None:
    """Regex matching a tax ID ignoring non-alphanumeric separators."""
    digits = normalize_tax_id(tax_id)
    if not digits:
        return None
    return r"^\W*" + r"\W*".join(re.escape(ch) for ch in digits) + r"\W*$"


def _email_match_pattern(email: str) -> str | None:
    """Case-insensitive exact email match."""
    normalized = normalize_email(email)
    if not normalized or "@" not in normalized:
        return None
    return r"^\s*" + re.escape(normalized) + r"\s*$"


def _phone_match_pattern(phone: str) -> str | None:
    """Regex matching a phone number ignoring non-digit separators."""
    digits = normalize_phone(phone)
    # Too short to be uniquely identifying
    if len(digits) < 7:
        return None
    return r"^\D*" + r"\D*".join(re.escape(ch) for ch in digits) + r"\D*$"


def issuer_fields_from_invoice_extraction(extracted: dict[str, Any]) -> dict[str, str]:
    """Map a raw Gemini invoice extraction dict to contact issuer fields."""
    return {
        "issuer": restore_issuer_name_punctuation(extracted.get("invoice_issuer")),
        "taxId": _preserve_printed_text(extracted.get("invoice_issuer_tax_id")),
        "email": _preserve_printed_text(extracted.get("issuer_email")),
        "phone": _preserve_printed_text(extracted.get("issuer_phone")),
        "address": _preserve_printed_text(extracted.get("company_address")),
        "profession": _preserve_printed_text(extracted.get("issuer_profession")),
    }


def extract_issuer_from_invoice(
    file_bytes: bytes, filename: str | None = None
) -> dict[str, str]:
    """Extract issuer fields from an invoice PDF/image via Gemini."""
    file_type = detect_file_type(file_bytes, filename)
    if file_type is None or file_type not in SUPPORTED_FILE_TYPES:
        raise ValueError(
            "File type not supported. Upload a PDF, JPEG, or PNG invoice."
        )

    extracted = _extract_invoice_with_gemini(file_bytes, filename=filename)
    return issuer_fields_from_invoice_extraction(extracted)


def build_contact_payload_from_invoice(extracted: dict[str, str]) -> dict[str, Any]:
    """Build Prop360 integration create payload."""
    issuer = _preserve_printed_text(extracted.get("issuer"))
    tax_id = _preserve_printed_text(extracted.get("taxId"))
    email = _preserve_printed_text(extracted.get("email"))
    phone = _preserve_printed_text(extracted.get("phone"))
    address = _preserve_printed_text(extracted.get("address"))
    profession = _preserve_printed_text(extracted.get("profession"))

    data: dict[str, Any] = {
        FIELD_CONTACT_TYPE: CONTACT_TYPE_CORPORATE,
        "Contact Notes": CONTACT_NOTES_CONSOLE,
        FIELD_CONTACT_ROLE: CONTACT_ROLE_SUPPLIER,
    }
    if issuer:
        data["Full Name"] = issuer
        data["Corporate Name"] = issuer
    if tax_id:
        data[FIELD_TAX_NUMBER] = tax_id
    if email:
        data[FIELD_EMAIL] = email
    if phone:
        data[FIELD_PHONE] = phone
    if address:
        data["Address"] = address
    if profession:
        data["Profession"] = profession

    return {"data": data}


def _contact_summary(doc: dict[str, Any]) -> dict[str, str]:
    data = doc.get("data") or {}
    return {
        "fullName": _unwrap_field_value(data.get(FIELD_FULL_NAME)),
        "corporateName": _unwrap_field_value(data.get(FIELD_CORPORATE_NAME)),
        "taxNumber": _unwrap_field_value(data.get(FIELD_TAX_NUMBER)),
        "email": _unwrap_field_value(data.get(FIELD_EMAIL)),
        "phone": _unwrap_field_value(data.get(FIELD_PHONE)),
        "address": _unwrap_field_value(data.get(FIELD_ADDRESS)),
        "profession": _unwrap_field_value(data.get(FIELD_PROFESSION)),
    }


async def find_existing_contact(
    prop_db: Any,
    tax_id: str | None = None,
    email: str | None = None,
    phone: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Find an active contact by Tax Number, then email, then phone.

    Tax IDs may be stored as strings (console) or ints (imports). $regex only
    matches strings, so we also exact-match the numeric form.

    Returns (document, matched_by) where matched_by is
    'tax_number' | 'email' | 'phone' | None.
    """
    base_filter: dict[str, Any] = {
        "indicator": "contacts",
        "status": "active",
    }

    digits = normalize_tax_id(tax_id or "")
    # Skip placeholder AFMs (e.g. 000000000) — they are not unique suppliers
    if digits and not set(digits) <= {"0"}:
        tax_clauses: list[dict[str, Any]] = [
            {f"data.{FIELD_TAX_NUMBER}": digits},
        ]

        # Imported contacts often store AFM as int; $regex would miss those
        if digits.isdigit():
            tax_clauses.append({f"data.{FIELD_TAX_NUMBER}": int(digits)})

        tax_pattern = _tax_id_match_pattern(digits)
        if tax_pattern:
            tax_clauses.append(
                {f"data.{FIELD_TAX_NUMBER}": {"$regex": tax_pattern, "$options": "i"}}
            )

        doc = await prop_db.formdatas.find_one({**base_filter, "$or": tax_clauses})
        if doc:
            return doc, "tax_number"

    email_pattern = _email_match_pattern(email or "")
    if email_pattern:
        doc = await prop_db.formdatas.find_one(
            {
                **base_filter,
                f"data.{FIELD_EMAIL}": {"$regex": email_pattern, "$options": "i"},
            }
        )
        if doc:
            return doc, "email"

    phone_pattern = _phone_match_pattern(phone or "")
    if phone_pattern:
        doc = await prop_db.formdatas.find_one(
            {
                **base_filter,
                f"data.{FIELD_PHONE}": {"$regex": phone_pattern},
            }
        )
        if doc:
            return doc, "phone"

    return None, None


def serialize_contact_result(
    *,
    status: str,
    contact_id: str,
    matched_by: str | None,
    extracted: dict[str, str],
    contact: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "contactId": contact_id,
        "matchedBy": matched_by,
        "extracted": {
            "issuer": extracted.get("issuer") or "",
            "taxId": extracted.get("taxId") or "",
            "email": extracted.get("email") or "",
            "phone": extracted.get("phone") or "",
            "address": extracted.get("address") or "",
            "profession": extracted.get("profession") or "",
        },
        "contact": contact
        or {
            "fullName": extracted.get("issuer") or "",
            "corporateName": extracted.get("issuer") or "",
            "taxNumber": extracted.get("taxId") or "",
            "email": extracted.get("email") or "",
            "phone": extracted.get("phone") or "",
            "address": extracted.get("address") or "",
            "profession": extracted.get("profession") or "",
        },
    }


def contact_summary_from_doc(doc: dict[str, Any]) -> dict[str, str]:
    return _contact_summary(doc)
