"""Canonical Madrid district-name rules shared by Silver transformations."""

import re
import unicodedata


EMVS_DISTRICT_ALIASES = {
    "MONCLOA": "MONCLOAARAVACA",
    "SANBLAS": "SANBLASCANILLEJAS",
    "VALLECASVILLA": "VILLADEVALLECAS",
}


def normalize_district_name(value: str) -> str:
    """Return an accent-, case-, whitespace-, and punctuation-insensitive key."""

    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]", "", without_accents.upper())


def canonicalize_emvs_district_key(value: str) -> tuple[str, str]:
    """Return the canonical join key and the rule used for one EMVS label."""

    normalized_key = normalize_district_name(value)
    canonical_key = EMVS_DISTRICT_ALIASES.get(normalized_key, normalized_key)
    mapping_method = (
        "explicit_alias"
        if canonical_key != normalized_key
        else "normalized_name"
    )
    return canonical_key, mapping_method
