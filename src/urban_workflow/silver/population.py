"""Source-specific population parsing rules used by the Silver layer."""

import re
from typing import Literal


CITY_AGGREGATE_CODE = "Todos"
POPULATION_DATE_PATTERN = re.compile(r"^1 de enero de ((?:19|20)\d{2})$")
GeographicLevel = Literal["city", "district", "barrio"]


def classify_population_geography(
    district_code: str,
    barrio_code: str,
) -> GeographicLevel:
    """Classify the CKAN population grain from its stable source codes."""

    if district_code == CITY_AGGREGATE_CODE:
        return "city"
    if district_code == barrio_code:
        return "district"
    return "barrio"


def parse_population_reference_year(value: str) -> int:
    """Parse the source's Spanish 1-January reference-date format."""

    match = POPULATION_DATE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"Unsupported population reference date: {value!r}")
    return int(match.group(1))


def parse_spanish_integer(value: str) -> int:
    """Parse an integer whose thousands separator is a period."""

    normalized = value.strip().replace(".", "")
    if not normalized or not normalized.isdigit():
        raise ValueError(f"Invalid Spanish integer: {value!r}")
    return int(normalized)


def normalize_district_code(value: str) -> str | None:
    """Return a two-digit district code, or null for the city aggregate."""

    if value == CITY_AGGREGATE_CODE:
        return None
    if not value.isdigit() or not 1 <= int(value) <= 21:
        raise ValueError(f"Invalid Madrid district code: {value!r}")
    return f"{int(value):02d}"


def normalize_barrio_code(
    value: str,
    geographic_level: GeographicLevel,
) -> str | None:
    """Return a three-digit barrio code only for barrio-grain records."""

    if geographic_level != "barrio":
        return None
    if not value.isdigit():
        raise ValueError(f"Invalid Madrid barrio code: {value!r}")
    return value.zfill(3)
