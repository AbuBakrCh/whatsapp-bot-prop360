"""Deprecated module name. Use services.ledger_report instead."""

from services.ledger_report import (  # noqa: F401
    build_ledger_excel,
    fetch_ledger_for_property,
    fetch_ledger_report,
    GROUP_TYPE_CONTACT,
    GROUP_TYPE_PROPERTY,
)
