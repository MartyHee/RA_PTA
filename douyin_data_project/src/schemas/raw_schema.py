"""
Lightweight schema tools for 11 raw tables.

This module provides schema loading and validation functions based on
raw_tables.yaml and field_sources.yaml. It is designed to be a single source
of truth for downstream parser / scheduler multi-table output.

Dependencies: yaml (PyYAML) — optional, falls back to os.path checks.

Usage:
    from src.schemas.raw_schema import (
        load_raw_tables_schema,
        load_field_sources,
        get_table_columns,
        get_all_raw_table_names,
        validate_table_columns,
        build_empty_raw_record,
    )

    schema = load_raw_tables_schema()
    cols = get_table_columns("raw_video_detail")
    record = build_empty_raw_record("raw_video_detail")
"""

import csv
import os
from pathlib import Path

_SCHEMA_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# YAML loading — try PyYAML first, then fallback CSV
# ---------------------------------------------------------------------------

def _load_yaml_with_fallback(file_stem: str):
    """Load a YAML file, falling back to a simple dict reader if PyYAML missing."""
    yaml_path = _SCHEMA_DIR / f"{file_stem}.yaml"
    csv_path = _SCHEMA_DIR / f"{file_stem}.csv"

    # Prefer YAML
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            pass  # fall through to CSV fallback

    # Fallback: try CSV if it exists
    if csv_path.exists():
        return _load_csv_fallback(csv_path)

    raise FileNotFoundError(
        f"Neither {yaml_path} nor {csv_path} found. "
        "Install PyYAML or provide a CSV fallback."
    )


def _load_csv_fallback(csv_path: Path) -> dict:
    """Minimal CSV-based schema loader (YAML fallback)."""
    import json
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Build a simple dict structure from CSV rows
    return {"tables": rows, "_source": str(csv_path)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_tables_schema() -> dict:
    """Load the full 11-table schema from raw_tables.yaml.

    Returns:
        dict with top-level key "tables" mapping table_name -> table spec.
    """
    data = _load_yaml_with_fallback("raw_tables")
    if "tables" not in data:
        raise ValueError("raw_tables.yaml must contain a 'tables' key")
    return data["tables"]


def load_field_sources() -> list[dict]:
    """Load field-level source mapping from field_sources.yaml.

    Returns:
        list of dicts, each describing one field's source/priority.
    """
    data = _load_yaml_with_fallback("field_sources")
    if "fields" not in data:
        raise ValueError("field_sources.yaml must contain a 'fields' key")
    return data["fields"]


def get_all_raw_table_names() -> list[str]:
    """Return sorted list of all 11 raw table names."""
    schema = load_raw_tables_schema()
    return sorted(schema.keys())


def get_table_columns(table_name: str) -> list[str]:
    """Get ordered column list for a table.

    Order: required_fields first, then optional_fields, then quality_fields.
    """
    schema = load_raw_tables_schema()
    table = schema.get(table_name)
    if table is None:
        raise KeyError(f"Table '{table_name}' not found in schema")
    order = []
    order.extend(table.get("required_fields", []))
    order.extend(table.get("optional_fields", []))
    order.extend(table.get("quality_fields", []))
    return order


def get_table_column_count(table_name: str) -> int:
    """Get total column count for a table."""
    return len(get_table_columns(table_name))


def validate_table_columns(table_name: str, columns: list[str]) -> dict:
    """Validate that a list of columns matches the expected schema.

    Args:
        table_name: name of the raw table.
        columns: actual columns from a DataFrame / CSV.

    Returns:
        dict with keys: valid, missing, extra, expected_count, actual_count.
    """
    expected = get_table_columns(table_name)
    expected_set = set(expected)
    actual_set = set(columns)

    result = {
        "valid": True,
        "missing": sorted(expected_set - actual_set),
        "extra": sorted(actual_set - expected_set),
        "expected_count": len(expected),
        "actual_count": len(columns),
        "expected_fields": expected,
        "actual_fields": columns,
    }
    if result["missing"] or result["extra"]:
        result["valid"] = False
    return result


def build_empty_raw_record(table_name: str) -> dict:
    """Build an empty record (all None) for the given table.

    Returns:
        dict with all expected columns as keys, values = None.
    """
    columns = get_table_columns(table_name)
    return {col: None for col in columns}


def get_table_schema(table_name: str):  # -> Optional[dict]
    """Get the full schema definition for a single table."""
    schema = load_raw_tables_schema()
    return schema.get(table_name)


def find_field(table_name: str, field_name: str):  # -> Optional[dict]
    """Look up a single field definition in field_sources.yaml.

    Returns:
        The field dict, or None if not found.
    """
    sources = load_field_sources()
    for f in sources:
        if f.get("table_name") == table_name and f.get("field_name") == field_name:
            return f
    return None


# ---------------------------------------------------------------------------
# Convenience: print summary
# ---------------------------------------------------------------------------

def print_schema_summary():
    """Print a compact summary of all tables and their field counts."""
    schema = load_raw_tables_schema()
    print(f"{'Table Name':<30} {'Required':<10} {'Optional':<10} {'Quality':<10} {'Total':<10}")
    print("-" * 70)
    for name in sorted(schema):
        t = schema[name]
        req = len(t.get("required_fields", []))
        opt = len(t.get("optional_fields", []))
        qual = len(t.get("quality_fields", []))
        total = req + opt + qual
        print(f"{name:<30} {req:<10} {opt:<10} {qual:<10} {total:<10}")


if __name__ == "__main__":
    # Quick self-test when run directly
    print("=" * 60)
    print("Raw Schema Self-Test")
    print("=" * 60)

    print_schema_summary()

    print("\n--- Per-table column order ---")
    for name in get_all_raw_table_names():
        cols = get_table_columns(name)
        print(f"  {name:30s} ({len(cols):2d} cols): {', '.join(cols[:6])}{'...' if len(cols) > 6 else ''}")

    print("\n--- build_empty_raw_record check ---")
    for name in get_all_raw_table_names():
        record = build_empty_raw_record(name)
        all_none = all(v is None for v in record.values())
        status = "OK" if all_none else "FAIL (some values not None)"
        print(f"  {name:30s} [{status}]")

    print("\nSelf-test complete.")
