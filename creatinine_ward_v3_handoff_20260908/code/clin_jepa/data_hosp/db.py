"""DuckDB access layer for the hospital pipeline.

The ICU pipeline reads mimic-code CSV exports through
``clin_jepa/data/helpers/io_utils.py``. No such export exists on this
machine, so the hospital pipeline queries MIMIC-IV directly out of DuckDB
and lets the engine do the joins and window clipping that the ICU pipeline
does in pandas.
"""

from __future__ import annotations

import logging
from pathlib import Path
import os
from typing import Any

import duckdb
import yaml

logger = logging.getLogger(__name__)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "data_hosp"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "${" in expanded:
            raise ValueError(f"unresolved environment variable in {expanded!r}")
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge YAML overrides without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML, resolving paths and optional ``extends`` inheritance."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = CONFIG_ROOT / p.name
        if candidate.exists():
            p = candidate
    with open(p, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent = config.pop("extends", None)
    if parent is None:
        return _expand_env(config)

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = p.parent / parent_path
    merged = _deep_merge(load_yaml(parent_path), config)
    # Resolve variables introduced by the child after inheritance.
    return _expand_env(merged)


def connect(paths_config: dict, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Open the MIMIC-IV DuckDB and attach derived concepts as ``derived``.

    Args:
        paths_config: Parsed ``hosp_paths.yaml``.
        read_only: Open the main database read-only. Always leave this True
            for pipeline steps; they never write back to MIMIC.

    Returns:
        A connection where ``mimiciv_hosp`` / ``mimiciv_icu`` resolve against
        the raw database and ``derived`` resolves against the mimic-code
        concepts database (absent if not configured).
    """
    db_cfg = paths_config["db"]
    con = duckdb.connect(db_cfg["mimic"], read_only=read_only)

    duck = paths_config.get("duckdb", {})
    if duck.get("threads"):
        con.execute(f"SET threads={int(duck['threads'])}")
    if duck.get("memory_limit"):
        con.execute(f"SET memory_limit='{duck['memory_limit']}'")
    if duck.get("temp_directory"):
        tmp = Path(duck["temp_directory"])
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp}'")

    derived = db_cfg.get("derived")
    if derived and Path(derived).exists():
        con.execute(f"ATTACH '{derived}' AS derived_db (READ_ONLY)")
        logger.info("Attached derived concepts: %s", derived)
    else:
        logger.warning("Derived concepts DB unavailable; ICU enrichment disabled")

    return con


def has_derived(con: duckdb.DuckDBPyConnection) -> bool:
    """Whether the derived-concepts database was successfully attached."""
    rows = con.execute("SELECT database_name FROM duckdb_databases()").fetchall()
    return any(r[0] == "derived_db" for r in rows)


def derived_table(name: str) -> str:
    """Fully-qualified name of a mimic-code derived concept table."""
    return f"derived_db.mimiciv_derived.{name}"


def derived_tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Names of the concept tables actually present in the derived database."""
    if not has_derived(con):
        return set()
    rows = con.execute(
        """
        SELECT table_name FROM duckdb_tables()
        WHERE database_name = 'derived_db'
        """
    ).fetchall()
    return {r[0] for r in rows}


def register_cohort(
    con: duckdb.DuckDBPyConnection,
    cohort_path: Path,
    n_buckets: int,
) -> int:
    """Register the step01 cohort as the temp table ``cohort``.

    Adds a ``bucket`` column so downstream steps can shard by admission
    without keeping one file per admission.

    Returns:
        Number of cohort rows.
    """
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE cohort AS
        SELECT *, (hadm_id % {n_buckets}) AS bucket
        FROM read_parquet('{cohort_path}')
        """
    )
    return int(con.execute("SELECT count(*) FROM cohort").fetchone()[0])


def copy_to_buckets(
    con: duckdb.DuckDBPyConnection,
    select_sql: str,
    out_dir: Path,
    overwrite: bool = False,
) -> None:
    """Write a SELECT result to ``out_dir`` partitioned by ``bucket``.

    The query must project a ``bucket`` column.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = ["FORMAT PARQUET", "PARTITION_BY (bucket)", "COMPRESSION ZSTD"]
    opts.append("OVERWRITE_OR_IGNORE" if overwrite else "APPEND")
    con.execute(f"COPY ({select_sql}) TO '{out_dir}' ({', '.join(opts)})")
