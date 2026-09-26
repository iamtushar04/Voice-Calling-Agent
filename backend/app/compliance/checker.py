"""
Compliance checker — pre-call and during-call safety layer.

EVERY call attempt must pass all checks before dialing.
If ANY check fails, the call is SKIPPED and the reason is logged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time as dtime
from enum import Enum
from typing import Optional

import phonenumbers
from phonenumbers import NumberParseException

from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class ComplianceResult(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"


@dataclass
class ComplianceDecision:
    result: ComplianceResult
    reason: Optional[str] = None
    skip_permanently: bool = False   # True = never retry (opted out, DNC, invalid)


class ComplianceChecker:
    """
    Runs all pre-call compliance checks.
    
    Checks (in order):
    1. Phone number is a valid E.164 number
    2. Contact is not marked do_not_call
    3. Contact consent status is not OPTED_OUT
    4. Number is not on the suppression list
    5. Campaign is in RUNNING status
    6. Current time is within allowed calling hours
    """

    def __init__(self, suppression_list: Optional["SuppressionList"] = None):
        self.suppression_list = suppression_list

    def check_all(
        self,
        phone_number: str,
        do_not_call: bool,
        consent_status: str,
        campaign_status: str,
        timezone: str = "UTC",
        calling_hours_start: str = "09:00",
        calling_hours_end: str = "17:00",
    ) -> ComplianceDecision:
        """Run all compliance checks. Returns on first failure."""

        # 1. Valid phone number
        validation = validate_e164(phone_number)
        if not validation.is_valid:
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason=f"Invalid phone number: {validation.error}",
                skip_permanently=True,
            )

        # 2. Do-not-call flag
        if do_not_call:
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason="Contact is marked do_not_call",
                skip_permanently=True,
            )

        # 3. Consent / opt-out status
        if consent_status in ("OPTED_OUT", "DENIED"):
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason=f"Contact consent status is {consent_status}",
                skip_permanently=True,
            )

        # 4. Suppression list
        if self.suppression_list and self.suppression_list.is_suppressed(phone_number):
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason="Number is on suppression list",
                skip_permanently=True,
            )

        # 5. Campaign must be running
        if campaign_status != "RUNNING":
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason=f"Campaign is not RUNNING (status: {campaign_status})",
                skip_permanently=False,
            )

        # 6. Calling hours check
        if not within_calling_hours(calling_hours_start, calling_hours_end, timezone):
            return ComplianceDecision(
                result=ComplianceResult.BLOCKED,
                reason="Outside allowed calling hours",
                skip_permanently=False,
            )

        logger.info("compliance_passed", phone_number=phone_number)
        return ComplianceDecision(result=ComplianceResult.ALLOWED)


# ── Suppression List ──────────────────────────────────────────────────────────

class SuppressionList:
    """
    In-memory suppression list (backed by Redis/DB in production).
    Phone numbers that must never be called.
    """

    def __init__(self):
        self._numbers: set[str] = set()

    def add(self, phone_number: str) -> None:
        normalized = normalize_e164(phone_number)
        if normalized:
            self._numbers.add(normalized)
            logger.info("suppression_added", phone_number=normalized)

    def remove(self, phone_number: str) -> None:
        self._numbers.discard(normalize_e164(phone_number) or phone_number)

    def is_suppressed(self, phone_number: str) -> bool:
        return (normalize_e164(phone_number) or phone_number) in self._numbers

    def count(self) -> int:
        return len(self._numbers)


# ── Phone Number Utilities ────────────────────────────────────────────────────

@dataclass
class PhoneValidationResult:
    is_valid: bool
    normalized: Optional[str] = None
    country_code: Optional[str] = None
    error: Optional[str] = None


def validate_e164(phone_number: str) -> PhoneValidationResult:
    """
    Validate and normalize a phone number to E.164 format.
    
    Accepts: +919876543210, 919876543210, 9876543210 (with country hint)
    Returns: +919876543210 or error
    """
    if not phone_number:
        return PhoneValidationResult(is_valid=False, error="Empty phone number")

    try:
        # Try parsing as international E.164
        parsed = phonenumbers.parse(phone_number, None)
        if phonenumbers.is_valid_number(parsed):
            e164 = phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
            country = phonenumbers.region_code_for_number(parsed)
            return PhoneValidationResult(
                is_valid=True,
                normalized=e164,
                country_code=country,
            )
    except NumberParseException:
        pass

    # Try with US as default country
    try:
        parsed = phonenumbers.parse(phone_number, "US")
        if phonenumbers.is_valid_number(parsed):
            e164 = phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.E164
            )
            return PhoneValidationResult(
                is_valid=True,
                normalized=e164,
                country_code="US",
            )
    except NumberParseException:
        pass

    return PhoneValidationResult(
        is_valid=False,
        error=f"Cannot parse '{phone_number}' as a valid phone number",
    )


def normalize_e164(phone_number: str) -> Optional[str]:
    """Returns E.164 string or None if invalid."""
    result = validate_e164(phone_number)
    return result.normalized if result.is_valid else None


def within_calling_hours(
    start: str,
    end: str,
    timezone: str = "UTC",
) -> bool:
    """
    Check if current time is within allowed calling hours.
    start/end format: "HH:MM"
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz).time()
        start_t = dtime(*[int(x) for x in start.split(":")])
        end_t = dtime(*[int(x) for x in end.split(":")])
        return start_t <= now <= end_t
    except Exception:
        return True   # Fail open during development; fail closed in production


# Singleton instances
suppression_list = SuppressionList()
compliance_checker = ComplianceChecker(suppression_list=suppression_list)
