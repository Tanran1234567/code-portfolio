"""Canonical event schema for the hospital pipeline.

Same 13 logical columns as ``clin_jepa/data/helpers/schema.py``, with two
changes: the stay key is ``hadm_id`` rather than ``stay_id``, and a
``bucket`` column carries the parquet partition.
"""

from __future__ import annotations

import pandas as pd

EVENT_COLUMNS = [
    "hadm_id",
    "timestamp",
    "timestamp_hours",
    "domain",
    "source_table",
    "category",
    "variable_name",
    "numeric_value",
    "text_value",
    "unit",
    "end_timestamp",
    "end_timestamp_hours",
    "is_continuous",
]

EVENT_DTYPES = {
    "hadm_id": "int64",
    "timestamp": "datetime64[us]",
    "timestamp_hours": "float64",
    "domain": "object",
    "source_table": "object",
    "category": "object",
    "variable_name": "object",
    "numeric_value": "float64",
    "text_value": "object",
    "unit": "object",
    "end_timestamp": "datetime64[us]",
    "end_timestamp_hours": "float64",
    "is_continuous": "bool",
}

# Columns retained after binning, mirroring OBS_COLS / ACT_COLS in
# clin_jepa/data/step04_discretization.py.
OBS_COLS = [
    "step_idx", "variable_name", "numeric_value",
    "text_value", "unit", "category", "source_table",
]
ACT_COLS = OBS_COLS + ["is_continuous"]

# The SELECT list every step02/step03 extraction query must project, so that
# heterogeneous sources land in one long-format table.
#
# Every column is cast explicitly. Without this, a source that projects a
# literal (diagnoses uses `0.0 AS timestamp_hours`) writes DECIMAL(2,1) while
# the rest write DOUBLE, and reading the bucket back with a single glob fails
# on the schema mismatch — DuckDB infers the schema from the first file it
# sees.
SQL_EVENT_PROJECTION = """
    hadm_id::BIGINT              AS hadm_id,
    timestamp::TIMESTAMP         AS timestamp,
    timestamp_hours::DOUBLE      AS timestamp_hours,
    domain::VARCHAR              AS domain,
    source_table::VARCHAR        AS source_table,
    category::VARCHAR            AS category,
    variable_name::VARCHAR       AS variable_name,
    numeric_value::DOUBLE        AS numeric_value,
    text_value::VARCHAR          AS text_value,
    unit::VARCHAR                AS unit,
    end_timestamp::TIMESTAMP     AS end_timestamp,
    end_timestamp_hours::DOUBLE  AS end_timestamp_hours,
    is_continuous::BOOLEAN       AS is_continuous,
    bucket::INTEGER              AS bucket
"""


def validate_events(df: pd.DataFrame) -> None:
    """Raise if a DataFrame does not carry exactly the event columns."""
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")


def empty_events_df() -> pd.DataFrame:
    """Empty frame with the event schema."""
    return pd.DataFrame(
        {c: pd.Series(dtype=EVENT_DTYPES[c]) for c in EVENT_COLUMNS}
    )
