import re

from urban_workflow.ingestion.schemas import CompatibilityHints


YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
JOIN_KEYWORDS = (
    "id",
    "code",
    "name",
    "key",
    "district",
    "neigh",
    "barrio",
    "ward",
    "zone",
    "sector",
    "tract",
    "municip",
    "fua",
)
SPATIAL_UNIT_KEYWORDS = (
    "district",
    "neigh",
    "barrio",
    "ward",
    "zone",
    "sector",
    "tract",
    "municip",
    "fua",
    "grid",
    "cell",
)


def extract_year_candidates(values: list[str]) -> list[int]:
    candidates: list[int] = []

    for value in values:
        if not value:
            continue

        years = YEAR_PATTERN.findall(value)
        candidates.extend(int(year) for year in years)

    return sorted(set(candidates))


def infer_reference_year(reference_texts: list[str]) -> tuple[int | None, list[str], list[str]]:
    year_candidates = extract_year_candidates(reference_texts)

    if not year_candidates:
        return None, [], [
            "No reference year could be inferred during collection. Temporal alignment will need human review."
        ]

    if len(year_candidates) > 1:
        return None, [], [
            "Multiple year candidates were detected during collection. The source year remains unresolved."
        ]

    inferred_year = year_candidates[0]
    basis = [f"Inferred from collection metadata text: {inferred_year}"]
    return inferred_year, basis, []


def find_candidate_columns(columns: list[str], keywords: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in columns
        if any(keyword in column.lower() for keyword in keywords)
    ]


def build_compatibility_hints(
    columns: list[str],
    source_type: str,
    reference_texts: list[str],
) -> tuple[CompatibilityHints, list[str]]:
    inferred_year, year_basis, warnings = infer_reference_year(reference_texts)

    candidate_join_columns = find_candidate_columns(columns, JOIN_KEYWORDS)
    candidate_spatial_unit_columns = find_candidate_columns(columns, SPATIAL_UNIT_KEYWORDS)

    hints = CompatibilityHints(
        inferred_reference_year=inferred_year,
        year_inference_basis=year_basis,
        candidate_join_columns=candidate_join_columns,
        candidate_spatial_unit_columns=candidate_spatial_unit_columns,
        geometry_present=source_type == "geospatial",
    )

    if source_type == "tabular" and not candidate_spatial_unit_columns:
        warnings.append(
            "No obvious spatial unit column was detected in the tabular source. Direct integration may require later aggregation or geographic referencing."
        )

    if not candidate_join_columns:
        warnings.append(
            "No obvious join key candidates were detected during collection. Compatibility assessment should treat linkage assumptions cautiously."
        )

    return hints, warnings