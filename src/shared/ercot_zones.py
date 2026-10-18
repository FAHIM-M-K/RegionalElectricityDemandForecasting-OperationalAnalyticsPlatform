"""Small helpers for mapping EIA zone identifiers to project zone identifiers."""


def source_to_zone_code_map(subregions: list[dict]) -> dict[str, str]:
    """Map source codes to canonical project codes, defaulting to identity."""
    mapping: dict[str, str] = {}
    for subregion in subregions:
        canonical = str(subregion["code"]).strip().upper()
        source = str(subregion.get("eia_code", canonical)).strip().upper()
        if source in mapping and mapping[source] != canonical:
            raise ValueError(f"EIA subregion code {source!r} maps to multiple canonical zones.")
        mapping[source] = canonical
    return mapping


def canonical_zone_code(source_code: str, mapping: dict[str, str]) -> str:
    """Return the configured canonical zone code for an EIA source code."""
    normalized = str(source_code).strip().upper()
    return mapping.get(normalized, normalized)
