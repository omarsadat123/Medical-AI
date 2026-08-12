"""Load and resolve biomarker reference ranges."""

from __future__ import annotations

from functools import lru_cache

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "reference_ranges.json"


@lru_cache(maxsize=1)
def load_reference_ranges() -> dict[str, Any]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_range_for_biomarker(
    biomarker_key: str,
    sex: Optional[str] = None,
) -> Optional[dict[str, float]]:
    catalog = load_reference_ranges()
    entry = catalog.get(biomarker_key)
    if not entry:
        return None

    sex_norm = (sex or "").strip().lower()
    if sex_norm in {"male", "m"} and "male" in entry:
        return dict(entry["male"])
    if sex_norm in {"female", "f"} and "female" in entry:
        return dict(entry["female"])
    return dict(entry.get("default", {}))


def build_alias_index() -> dict[str, str]:
    """Map normalized alias -> biomarker key."""
    return _build_alias_index_cached()


@lru_cache(maxsize=1)
def _build_alias_index_cached() -> dict[str, str]:
    """Map normalized alias -> biomarker key."""
    catalog = load_reference_ranges()
    index: dict[str, str] = {}
    for key, meta in catalog.items():
        for alias in meta.get("aliases", []):
            index[_normalize_alias(alias)] = key
        index[_normalize_alias(meta.get("name", key))] = key
        index[_normalize_alias(key)] = key
    # Longer aliases first helps matching
    return dict(sorted(index.items(), key=lambda kv: len(kv[0]), reverse=True))


def _normalize_alias(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("_", " ").split())
