"""Source-specific indicator-panel parsing rules for the Silver layer."""

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Literal


IndicatorGeographicLevel = Literal["city", "district", "barrio"]


def classify_indicator_geography(
    district_code: str | None,
    barrio_code: str | None,
) -> IndicatorGeographicLevel:
    """Classify indicator grain from nullable district and barrio codes."""

    if district_code is None:
        return "city"
    if barrio_code is None:
        return "district"
    return "barrio"


def parse_indicator_date(value: str) -> date:
    """Parse the source's day/month/year indicator date."""

    try:
        return datetime.strptime(value.strip(), "%d/%m/%Y").date()
    except ValueError as exc:
        raise ValueError(f"Invalid indicator date: {value!r}") from exc


def parse_indicator_decimal(value: str) -> Decimal:
    """Parse a Spanish decimal while preserving up to nine decimal places."""

    normalized = value.strip().replace(".", "").replace(",", ".")
    try:
        parsed = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid indicator numeric value: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"Invalid indicator numeric value: {value!r}")
    return parsed


def normalize_indicator_district_code(value: str | None) -> str | None:
    """Return a two-digit district code, or null for city-level records."""

    if value is None:
        return None
    if not value.isdigit() or not 1 <= int(value) <= 21:
        raise ValueError(f"Invalid Madrid district code: {value!r}")
    return f"{int(value):02d}"


def normalize_indicator_barrio_code(
    value: str | None,
    geographic_level: IndicatorGeographicLevel,
) -> str | None:
    """Return a three-digit code only for barrio-grain observations."""

    if geographic_level != "barrio":
        return None
    if value is None or not value.isdigit():
        raise ValueError(f"Invalid Madrid barrio code: {value!r}")
    return value.zfill(3)
