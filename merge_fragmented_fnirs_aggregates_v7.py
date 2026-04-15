#!/usr/bin/env python3
"""
Merge fragmented fNIRS aggregate folders into one authoritative aggregate directory.

This script is designed for the user's thesis reconstruction workflow where results are
split across six runs/folders. It is parquet-only and memory-safe:
- reads input tables in Arrow batches
- resolves winners at a group level
- streams only winning rows to the merged output parquet
- writes validation / provenance reports

Expected input layout
---------------------
A parent directory that contains the six unzipped folders (names below), each either:
- directly containing parquet files, or
- containing an `aggregate/` subdirectory with parquet files.

Default folder names (can be overridden by CLI args):
- 86_nondeleted_aggregate
- python_rerun_aggregate
- gaussian_aggregate
- rerun_aggregate
- repair_aggregate
- actual_additional_analyzir_aggregate

Important merge logic
---------------------
1) Build the Python base from:
   - old86 (surviving original Python aggregate)
   - python_rerun (recovered missing Python subject-file jobs)
2) Exclude globally bad / undesired pipelines:
   - Homer3_tCCA_ModGamma
   - Keep LocalSS_FIR_AUTO from the trusted Python sources (old86 and python_rerun)
   - Do not use the old flex_homer3 aggregate at all; use gaussian_aggregate only for Homer3_tCCA_Gaussian
3) Overlay corrected / later sources by exact logical keys using source priority:
   repair > rerun > additional > gaussian > python_rerun > old86
4) Use actual pipeline labels present in the tables, not folder names alone.
5) For row-heavy result tables, replacement is group-level, not individual-cell patching.

Notes
-----
- This script does not use CSVs.
- It preserves source provenance in the merged parquets when feasible.
- `config.parquet` is reconstructed as a one-row consensus summary with a companion
  conflict report.
- `pipeline_manifest.parquet` is reconstructed as a deduplicated union over kept
  pipeline labels, preferring higher-priority sources for overlapping labels.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as exc:
    raise SystemExit(
        "This script requires pyarrow. Activate an environment with pyarrow installed.\n"
        f"Import error: {exc}"
    )

# -----------------------------------------------------------------------------
# Source definitions and merge policy
# -----------------------------------------------------------------------------

SOURCE_PRIORITY = {
    # lower number = higher priority when exact logical keys overlap
    "repair": 1,
    "rerun": 2,
    "additional": 3,
    "gaussian": 4,
    "python_rerun": 5,
    "old86": 6,
}

DEFAULT_SOURCE_DIRS = {
    "old86": "86_nondeleted_aggregate",
    "python_rerun": "python_rerun_aggregate",
    "gaussian": "gaussian_aggregate",
    "rerun": "rerun_aggregate",
    "repair": "repair_aggregate",
    "additional": "actual_additional_analyzir_aggregate",
}

GLOBAL_EXCLUDE_PIPELINES = {
    "Homer3_tCCA_ModGamma",
    "LocalSS_Gamma_Derivs_ARIRLS",
    "LocalSS_SPM_Derivs_ARIRLS",
}

KEEP_FIR_ONLY_FROM = {"old86", "python_rerun"}

# LocalSS_FIR_AUTO is trusted only from the original/salvaged Python aggregates.

PYTHON_BASE_ALLOWED = {
    "NoSS_Glover_AUTO",
    "LocalSS_Glover_AUTO",
    "PooledPCA2_Glover_AUTO",
    "SSAuxPCA_Glover_AUTO",
    "MultiSSOrth3_Glover_AUTO",
    "NoSS_Glover_OLS",
    "LocalSS_Glover_OLS",
    "LocalSS_SPM_AUTO",
    "LocalSS_SPM_OLS",
    "BlockAvg_LocalSS",
    "LooseQC_LocalSS_Glover_AUTO",
    "WaveletMC_LocalSS_Glover_AUTO",
    "HighPassOnly_LocalSS_Glover_AUTO",
    "NoMotion_LocalSS_Glover_AUTO",
    "LocalSS_FIR_AUTO",
    # Valid Python pipelines retained from the Python runs.
    "LocalSS_SPM_Derivs_OLS",
    "LocalSS_Gamma_Derivs_OLS",
}

GAUSSIAN_ALLOWED = {
    "Homer3_tCCA_Gaussian",
}

RERUN_ALLOWED = {
    "NoSS_Canonical_ARIRLS",
    "LocalSSReg_Canonical_ARIRLS",
    "LocalSSReg_CanonicalDerivs_ARIRLS",
    "LocalSSFilter_Canonical_ARIRLS",
}

REPAIR_ALLOWED = set(RERUN_ALLOWED)

ADDITIONAL_ALLOWED = {
    # NoSS_Canonical_ARIRLS is included as a low-priority fallback only.
    "NoSS_Canonical_ARIRLS",
    "NoSS_Canonical_ARIRLS_NoTDDR",
    "NoSS_Canonical_OLS",
    "NoSS_Canonical_OLS_NoTDDR",
    "NoSS_FIR_OLS_NoTDDR",
}

SOURCE_ALLOWED_PIPELINES = {
    "old86": PYTHON_BASE_ALLOWED,
    "python_rerun": PYTHON_BASE_ALLOWED,
    "gaussian": GAUSSIAN_ALLOWED,
    "rerun": RERUN_ALLOWED,
    "repair": REPAIR_ALLOWED,
    "additional": ADDITIONAL_ALLOWED,
}

IDENTITY_COLUMN_CANDIDATES = [
    "subject",
    "file_label",
    "pipeline_label",
    "chromophore",
    "target_status",
    "score_type",
    "roi_name",
    "channel_idx",
    "channel_label",
    "pair_idx",
    "pair_label",
    "time_s",
    "status",
    "job_dir",
]


ROI_TIMECOURSE_KEY_COLS = ["subject", "file_label", "pipeline_label", "chromophore", "target_status", "time_s"]

SPECIAL_TABLES = {
    "config.parquet",
    "pipeline_manifest.parquet",
    "master_run_log.parquet",
    "truth_summary.parquet",
}

SMALL_ROW_DEDUP_TABLES = {
    "master_run_log.parquet",
    "truth_summary.parquet",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    root_dir: Path
    aggregate_dir: Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_pipeline_label(label: object) -> str:
    s = norm_text(label)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s


def ensure_string_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(norm_text)
    return df


def find_aggregate_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Source folder not found: {root}")
    direct = root / "pipeline_manifest.parquet"
    nested = root / "aggregate" / "pipeline_manifest.parquet"
    if direct.exists():
        return root
    if nested.exists():
        return root / "aggregate"
    # Fallback: search one level down.
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "pipeline_manifest.parquet").exists():
            return child
    raise FileNotFoundError(
        f"Could not locate aggregate parquet directory under {root}. "
        "Expected pipeline_manifest.parquet directly or in aggregate/."
    )


def discover_sources(parent_dir: Path, overrides: Dict[str, Optional[str]]) -> List[SourceSpec]:
    specs: List[SourceSpec] = []
    for name, default_dir in DEFAULT_SOURCE_DIRS.items():
        folder_name = overrides.get(name) or default_dir
        root = (parent_dir / folder_name).resolve()
        aggregate_dir = find_aggregate_dir(root)
        specs.append(SourceSpec(name=name, root_dir=root, aggregate_dir=aggregate_dir))
    return specs


def source_inventory_df(sources: Sequence[SourceSpec]) -> pd.DataFrame:
    rows = []
    for spec in sources:
        parquets = sorted([p.name for p in spec.aggregate_dir.glob("*.parquet")])
        rows.append(
            {
                "source_name": spec.name,
                "root_dir": str(spec.root_dir),
                "aggregate_dir": str(spec.aggregate_dir),
                "n_parquet_tables": len(parquets),
                "parquet_tables": ", ".join(parquets),
            }
        )
    return pd.DataFrame(rows)


def get_all_parquet_names(sources: Sequence[SourceSpec]) -> List[str]:
    names: Set[str] = set()
    for spec in sources:
        for p in spec.aggregate_dir.glob("*.parquet"):
            names.add(p.name)
    return sorted(names)


def dataset_columns(path: Path) -> List[str]:
    return list(ds.dataset(str(path), format="parquet").schema.names)


def safe_columns(dataset: ds.Dataset, wanted: Sequence[str]) -> List[str]:
    names = set(dataset.schema.names)
    return [c for c in wanted if c in names]


def apply_pipeline_filter(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty or "pipeline_label" not in df.columns:
        return df
    out = df.copy()
    out["pipeline_label"] = out["pipeline_label"].map(norm_text)
    out["pipeline_norm"] = out["pipeline_label"].map(normalize_pipeline_label)
    allowed = SOURCE_ALLOWED_PIPELINES[source_name]
    keep = out["pipeline_norm"].isin({normalize_pipeline_label(x) for x in allowed})
    keep &= ~out["pipeline_norm"].isin({normalize_pipeline_label(x) for x in GLOBAL_EXCLUDE_PIPELINES})
    fir_norm = normalize_pipeline_label("LocalSS_FIR_AUTO")
    if source_name not in KEEP_FIR_ONLY_FROM:
        keep &= out["pipeline_norm"].ne(fir_norm)
    out = out.loc[keep].copy()
    return out


def choose_identity_cols(schema_cols: Sequence[str]) -> List[str]:
    cols = [c for c in IDENTITY_COLUMN_CANDIDATES if c in schema_cols]
    if cols:
        return cols
    # Fallback: if there are no known semantic columns, use all columns for rowwise dedup.
    return list(schema_cols)


def make_key_from_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    if not cols:
        raise ValueError("Cannot make key with no columns")
    temp = []
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing key column {c}")
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            temp.append(s.astype(str).fillna(""))
        else:
            temp.append(s.map(norm_text))
    out = temp[0].astype(str)
    for s in temp[1:]:
        out = out + "||" + s.astype(str)
    return out




def collect_table_schema_info(table_name: str, sources: Sequence[SourceSpec]) -> Tuple[List[str], Dict[str, pa.DataType]]:
    ordered_cols: List[str] = []
    seen_cols: Set[str] = set()
    type_map: Dict[str, List[pa.DataType]] = {}
    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        schema = ds.dataset(str(path), format="parquet").schema
        for field in schema:
            name = field.name
            if name not in seen_cols:
                ordered_cols.append(name)
                seen_cols.add(name)
            type_map.setdefault(name, []).append(field.type)
    return ordered_cols, {c: choose_common_arrow_type(ts) for c, ts in type_map.items()}


def choose_common_arrow_type(types: Sequence[pa.DataType]) -> pa.DataType:
    if not types:
        return pa.string()
    has_string = any(pa.types.is_string(t) or pa.types.is_large_string(t) for t in types)
    if has_string:
        return pa.large_string()
    has_float = any(pa.types.is_floating(t) for t in types)
    has_int = any(pa.types.is_integer(t) for t in types)
    has_bool = any(pa.types.is_boolean(t) for t in types)
    if has_float:
        return pa.float64()
    if has_int and has_bool:
        return pa.int64()
    if has_int:
        return pa.int64()
    if has_bool:
        return pa.bool_()
    has_ts = any(pa.types.is_timestamp(t) for t in types)
    if has_ts:
        return pa.timestamp('ns')
    return pa.large_string()


def coerce_series_to_arrow_type(s: pd.Series, arrow_type: pa.DataType) -> pd.Series:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return s.astype('string')
    if pa.types.is_floating(arrow_type):
        return pd.to_numeric(s, errors='coerce').astype('float64')
    if pa.types.is_integer(arrow_type):
        return pd.to_numeric(s, errors='coerce').astype('Int64')
    if pa.types.is_boolean(arrow_type):
        if pd.api.types.is_bool_dtype(s):
            return s.astype('boolean')
        sn = pd.to_numeric(s, errors='coerce')
        out = pd.Series(pd.NA, index=s.index, dtype='boolean')
        mask = sn.isin([0, 1])
        out.loc[mask] = sn.loc[mask].astype('Int64').astype('boolean')
        return out
    if pa.types.is_timestamp(arrow_type):
        return pd.to_datetime(s, errors='coerce')
    return s


def align_df_to_schema(df: pd.DataFrame, ordered_cols: Sequence[str], col_types: Dict[str, pa.DataType]) -> pd.DataFrame:
    out = df.copy()
    for c in ordered_cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[list(ordered_cols)]
    for c in ordered_cols:
        out[c] = coerce_series_to_arrow_type(out[c], col_types.get(c, pa.large_string()))
    return out


def make_arrow_schema(ordered_cols: Sequence[str], col_types: Dict[str, pa.DataType]) -> pa.Schema:
    return pa.schema([pa.field(c, col_types.get(c, pa.large_string())) for c in ordered_cols])

def build_group_winners(
    table_name: str,
    sources: Sequence[SourceSpec],
    batch_rows: int,
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    parts: List[pd.DataFrame] = []

    present_schema_cols: List[List[str]] = []
    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        present_schema_cols.append(list(dataset.schema.names))

    if not present_schema_cols:
        raise FileNotFoundError(f"Table not found in any source: {table_name}")

    common_cols = set(present_schema_cols[0])
    for cols in present_schema_cols[1:]:
        common_cols &= set(cols)
    identity_cols = choose_identity_cols([c for c in IDENTITY_COLUMN_CANDIDATES if c in common_cols])
    if not identity_cols:
        identity_cols = sorted(common_cols)

    report_dir = output_dir / "merge_reports" / table_name.replace(".parquet", "")
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"identity_cols": identity_cols}).to_csv(report_dir / "identity_columns.csv", index=False)

    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        cols_to_read = safe_columns(dataset, identity_cols)
        scanner = dataset.scanner(columns=cols_to_read, batch_size=batch_rows)
        kept = 0

        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            df = ensure_string_columns(df, [c for c in identity_cols if c in df.columns and c != "time_s"])
            if "pipeline_label" in df.columns:
                df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue
            df = df[cols_to_read].drop_duplicates()
            for c in identity_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[identity_cols].drop_duplicates()
            df["source_name"] = spec.name
            df["source_path"] = str(path)
            df["priority_rank"] = SOURCE_PRIORITY[spec.name]
            parts.append(df)
            kept += len(df)

        eprint(f"[WINNERS] {table_name} :: {spec.name} groups kept = {kept:,}")

    if not parts:
        return pd.DataFrame(), identity_cols

    coverage = pd.concat(parts, ignore_index=True).drop_duplicates()
    coverage["logical_key"] = make_key_from_df(coverage, identity_cols)
    coverage = coverage.sort_values(["logical_key", "priority_rank"], kind="stable")
    winners = coverage.drop_duplicates(subset=["logical_key"], keep="first").copy()

    report_dir = output_dir / "merge_reports" / table_name.replace(".parquet", "")
    report_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(report_dir / "coverage_candidates.csv", index=False)
    winners.to_csv(report_dir / "coverage_winners.csv", index=False)

    conflicts = coverage.groupby("logical_key")["source_name"].nunique().reset_index(name="n_sources")
    conflicts = conflicts.loc[conflicts["n_sources"] > 1]
    conflicts.to_csv(report_dir / "coverage_conflicts.csv", index=False)

    return winners, identity_cols


def stream_group_winners(
    table_name: str,
    sources: Sequence[SourceSpec],
    winners: pd.DataFrame,
    identity_cols: Sequence[str],
    batch_rows: int,
    output_aggregate_dir: Path,
    compression: Optional[str],
) -> Path:
    out_path = output_aggregate_dir / table_name
    if winners.empty:
        return out_path

    winner_source = winners.set_index("logical_key")["source_name"].to_dict()
    winner_keys_by_source: Dict[str, Set[str]] = {}
    for src_name, subdf in winners.groupby("source_name"):
        winner_keys_by_source[src_name] = set(subdf["logical_key"].tolist())

    base_cols, base_types = collect_table_schema_info(table_name, sources)
    provenance_cols = ["pipeline_norm", "logical_key", "merged_from_source", "merged_from_path", "merge_priority_rank"]
    ordered_cols = list(base_cols)
    for c in provenance_cols:
        if c not in ordered_cols:
            ordered_cols.append(c)
    col_types = dict(base_types)
    col_types.setdefault("pipeline_norm", pa.large_string())
    col_types.setdefault("logical_key", pa.large_string())
    col_types.setdefault("merged_from_source", pa.large_string())
    col_types.setdefault("merged_from_path", pa.large_string())
    col_types.setdefault("merge_priority_rank", pa.int64())
    writer_schema = make_arrow_schema(ordered_cols, col_types)

    writer = None
    wrote_rows = 0

    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        keep_keys = winner_keys_by_source.get(spec.name, set())
        if not keep_keys:
            continue

        dataset = ds.dataset(str(path), format="parquet")
        cols = list(dataset.schema.names)
        scanner = dataset.scanner(columns=cols, batch_size=batch_rows)
        src_rows = 0

        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            df = ensure_string_columns(df, [c for c in identity_cols if c in df.columns and c != "time_s"])
            if "pipeline_label" in df.columns:
                df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue
            df["logical_key"] = make_key_from_df(df, identity_cols)
            df = df.loc[df["logical_key"].isin(keep_keys)].copy()
            if df.empty:
                continue
            df = df.loc[df["logical_key"].map(winner_source) == spec.name].copy()
            if df.empty:
                continue

            df["merged_from_source"] = spec.name
            df["merged_from_path"] = str(path)
            df["merge_priority_rank"] = SOURCE_PRIORITY[spec.name]

            df = align_df_to_schema(df, ordered_cols, col_types)
            table = pa.Table.from_pandas(df, schema=writer_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    str(out_path),
                    writer_schema,
                    compression=compression,
                    use_dictionary=True,
                )
            writer.write_table(table)
            src_rows += len(df)
            wrote_rows += len(df)

        eprint(f"[WRITE] {table_name} :: {spec.name} rows written = {src_rows:,}")

    if writer is not None:
        writer.close()
    eprint(f"[WRITE] {table_name} total rows written = {wrote_rows:,}")
    return out_path


def merge_small_row_dedup_table(
    table_name: str,
    sources: Sequence[SourceSpec],
    output_aggregate_dir: Path,
    batch_rows: int,
    compression: Optional[str],
) -> None:
    ordered_cols, col_types = collect_table_schema_info(table_name, sources)
    chunks: List[pd.DataFrame] = []
    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        cols = list(dataset.schema.names)
        scanner = dataset.scanner(columns=cols, batch_size=batch_rows)
        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            if "pipeline_label" in df.columns:
                df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue
            df["merged_from_source"] = spec.name
            chunks.append(df)
    if not chunks:
        return
    out = pd.concat(chunks, ignore_index=True).drop_duplicates()
    provenance_cols = [c for c in ["merged_from_source"] if c in out.columns and c not in ordered_cols]
    final_cols = list(ordered_cols) + provenance_cols
    final_types = dict(col_types)
    for c in provenance_cols:
        final_types.setdefault(c, pa.large_string())
    out = align_df_to_schema(out, final_cols, final_types)
    pq.write_table(pa.Table.from_pandas(out, schema=make_arrow_schema(final_cols, final_types), preserve_index=False), str(output_aggregate_dir / table_name), compression=compression)
    eprint(f"[SPECIAL] wrote {table_name} rows = {len(out):,}")


def merge_pipeline_manifest(
    sources: Sequence[SourceSpec],
    output_aggregate_dir: Path,
    output_dir: Path,
    batch_rows: int,
    compression: Optional[str],
) -> None:
    table_name = "pipeline_manifest.parquet"
    parts: List[pd.DataFrame] = []
    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(columns=list(dataset.schema.names), batch_size=batch_rows)
        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue
            df["pipeline_norm"] = df["pipeline_label"].map(normalize_pipeline_label)
            df["source_name"] = spec.name
            df["priority_rank"] = SOURCE_PRIORITY[spec.name]
            parts.append(df)
    if not parts:
        return
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["pipeline_norm", "priority_rank"], kind="stable")
    chosen = merged.drop_duplicates(subset=["pipeline_norm"], keep="first").copy()
    chosen = chosen.drop(columns=[c for c in ["source_name", "priority_rank"] if c in chosen.columns])
    if "pipeline_order" in chosen.columns:
        chosen["pipeline_order"] = range(1, len(chosen) + 1)
    ordered_cols, col_types = collect_table_schema_info(table_name, sources)
    chosen = align_df_to_schema(chosen, ordered_cols, col_types)
    pq.write_table(pa.Table.from_pandas(chosen, schema=make_arrow_schema(ordered_cols, col_types), preserve_index=False), str(output_aggregate_dir / table_name), compression=compression)
    merged.to_csv(output_dir / "merge_reports" / "pipeline_manifest_candidates.csv", index=False)
    eprint(f"[SPECIAL] wrote pipeline_manifest.parquet rows = {len(chosen):,}")


def merge_config(
    sources: Sequence[SourceSpec],
    output_aggregate_dir: Path,
    output_dir: Path,
    batch_rows: int,
    compression: Optional[str],
) -> None:
    table_name = "config.parquet"
    rows: List[pd.DataFrame] = []
    for spec in sources:
        path = spec.aggregate_dir / table_name
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(columns=list(dataset.schema.names), batch_size=batch_rows)
        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            if df.empty:
                continue
            df["source_name"] = spec.name
            rows.append(df)
    if not rows:
        return
    cfg = pd.concat(rows, ignore_index=True)

    protected = {"source_name"}
    consensus = {}
    conflict_rows = []
    for col in cfg.columns:
        if col in protected:
            continue
        vals = cfg[col]
        nonnull = vals[~vals.isna()]
        if len(nonnull) == 0:
            consensus[col] = pd.NA
            continue
        counts = nonnull.astype(str).value_counts(dropna=False)
        top_value = counts.index[0]
        consensus[col] = nonnull.astype(str).mode().iloc[0] if counts.iloc[0] > 1 else nonnull.iloc[0]
        if counts.shape[0] > 1:
            conflict_rows.append(
                {
                    "column_name": col,
                    "distinct_values": int(counts.shape[0]),
                    "top_value": str(top_value),
                    "value_counts": json.dumps(counts.to_dict()),
                }
            )

    out = pd.DataFrame([consensus])
    ordered_cols, col_types = collect_table_schema_info(table_name, sources)
    out = align_df_to_schema(out, ordered_cols, col_types)
    pq.write_table(pa.Table.from_pandas(out, schema=make_arrow_schema(ordered_cols, col_types), preserve_index=False), str(output_aggregate_dir / table_name), compression=compression)
    cfg.to_csv(output_dir / "merge_reports" / "config_all_sources.csv", index=False)
    pd.DataFrame(conflict_rows).to_csv(output_dir / "merge_reports" / "config_conflicts.csv", index=False)
    eprint(f"[SPECIAL] wrote config.parquet rows = {len(out):,}")




def build_roi_timecourses_sqlite_index(
    sources: Sequence[SourceSpec],
    batch_rows: int,
    output_dir: Path,
) -> Dict[str, pa.DataType]:
    """Disk-backed winner selection for roi_timecourses.parquet to avoid RAM blowups."""
    db_path = output_dir / "roi_timecourses_winners.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        "CREATE TABLE winners (subject TEXT, file_label TEXT, pipeline_label TEXT, chromophore TEXT, target_status TEXT, time_s REAL, source_name TEXT, priority_rank INTEGER, PRIMARY KEY(subject, file_label, pipeline_label, chromophore, target_status, time_s))"
    )

    ordered_cols, col_types = collect_table_schema_info("roi_timecourses.parquet", sources)
    stats = []

    for spec in sources:
        path = spec.aggregate_dir / "roi_timecourses.parquet"
        if not path.exists():
            continue
        dataset = ds.dataset(str(path), format="parquet")
        cols = safe_columns(dataset, ROI_TIMECOURSE_KEY_COLS)
        if set(ROI_TIMECOURSE_KEY_COLS) - set(cols):
            raise SystemExit(f"roi_timecourses missing required key columns in {spec.name}: {sorted(set(ROI_TIMECOURSE_KEY_COLS) - set(cols))}")
        scanner = dataset.scanner(columns=cols, batch_size=batch_rows)
        seen = 0
        kept = 0
        priority = SOURCE_PRIORITY[spec.name]

        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            seen += len(df)
            if df.empty:
                continue
            df = ensure_string_columns(df, [c for c in ROI_TIMECOURSE_KEY_COLS if c != "time_s"])
            df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue
            df = df[ROI_TIMECOURSE_KEY_COLS].drop_duplicates()
            rows = [
                (
                    str(r.subject),
                    str(r.file_label),
                    str(r.pipeline_label),
                    str(r.chromophore),
                    str(r.target_status),
                    float(r.time_s),
                    spec.name,
                    priority,
                )
                for r in df.itertuples(index=False)
            ]
            conn.executemany(
                "INSERT INTO winners(subject,file_label,pipeline_label,chromophore,target_status,time_s,source_name,priority_rank) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(subject,file_label,pipeline_label,chromophore,target_status,time_s) DO UPDATE SET "
                "source_name=excluded.source_name, priority_rank=excluded.priority_rank "
                "WHERE excluded.priority_rank < winners.priority_rank",
                rows,
            )
            kept += len(rows)

        conn.commit()
        stats.append({"source_name": spec.name, "rows_seen": seen, "candidate_rows_kept": kept})
        eprint(f"[ROI-SQLITE INDEX] {spec.name}: rows_seen={seen:,} candidate_rows_kept={kept:,}")

    conn.execute("CREATE INDEX idx_winners_source ON winners(source_name)")
    conn.commit()
    conn.close()

    report_dir = output_dir / "merge_reports" / "roi_timecourses"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(report_dir / "sqlite_build_stats.csv", index=False)
    return col_types


def write_roi_timecourses_from_sqlite(
    sources: Sequence[SourceSpec],
    output_aggregate_dir: Path,
    output_dir: Path,
    batch_rows: int,
    compression: Optional[str],
    col_types: Dict[str, pa.DataType],
) -> Path:
    db_path = output_dir / "roi_timecourses_winners.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing ROI SQLite index: {db_path}")
    out_path = output_aggregate_dir / "roi_timecourses.parquet"
    conn = sqlite3.connect(str(db_path))

    ordered_cols = list(col_types.keys())
    provenance_cols = ["pipeline_norm", "logical_key", "merged_from_source", "merged_from_path", "merge_priority_rank"]
    for c in provenance_cols:
        if c not in ordered_cols:
            ordered_cols.append(c)
    writer_types = dict(col_types)
    writer_types.setdefault("pipeline_norm", pa.large_string())
    writer_types.setdefault("logical_key", pa.large_string())
    writer_types.setdefault("merged_from_source", pa.large_string())
    writer_types.setdefault("merged_from_path", pa.large_string())
    writer_types.setdefault("merge_priority_rank", pa.int64())
    writer_schema = make_arrow_schema(ordered_cols, writer_types)

    writer = None
    summary = []

    for spec in sources:
        path = spec.aggregate_dir / "roi_timecourses.parquet"
        if not path.exists():
            continue
        winner_count = conn.execute("SELECT COUNT(*) FROM winners WHERE source_name=?", (spec.name,)).fetchone()[0]
        if winner_count == 0:
            continue
        dataset = ds.dataset(str(path), format="parquet")
        scanner = dataset.scanner(columns=list(dataset.schema.names), batch_size=batch_rows)
        src_rows = 0
        priority = SOURCE_PRIORITY[spec.name]

        for rb in scanner.to_batches():
            df = rb.to_pandas(types_mapper=pd.ArrowDtype)
            if df.empty:
                continue
            df = ensure_string_columns(df, [c for c in ROI_TIMECOURSE_KEY_COLS if c in df.columns and c != "time_s"])
            df = apply_pipeline_filter(df, spec.name)
            if df.empty:
                continue

            keys = [
                (
                    str(r.subject),
                    str(r.file_label),
                    str(r.pipeline_label),
                    str(r.chromophore),
                    str(r.target_status),
                    float(r.time_s),
                )
                for r in df[ROI_TIMECOURSE_KEY_COLS].itertuples(index=False)
            ]
            temp_name = f"batch_keys_{spec.name}"
            conn.execute(f"DROP TABLE IF EXISTS {temp_name}")
            conn.execute(f"CREATE TEMP TABLE {temp_name} (subject TEXT, file_label TEXT, pipeline_label TEXT, chromophore TEXT, target_status TEXT, time_s REAL)")
            conn.executemany(f"INSERT INTO {temp_name} VALUES (?,?,?,?,?,?)", keys)
            keep_df = pd.read_sql_query(
                f"SELECT b.rowid - 1 as batch_idx FROM {temp_name} b JOIN winners w ON "
                "b.subject=w.subject AND b.file_label=w.file_label AND b.pipeline_label=w.pipeline_label AND b.chromophore=w.chromophore AND b.target_status=w.target_status AND b.time_s=w.time_s "
                "WHERE w.source_name=?",
                conn,
                params=(spec.name,),
            )
            conn.execute(f"DROP TABLE {temp_name}")
            if keep_df.empty:
                continue
            df = df.iloc[keep_df["batch_idx"].to_numpy(dtype=int)].copy()
            if df.empty:
                continue

            df["pipeline_norm"] = df["pipeline_label"].map(normalize_pipeline_label)
            df["logical_key"] = make_key_from_df(df, ROI_TIMECOURSE_KEY_COLS)
            df["merged_from_source"] = spec.name
            df["merged_from_path"] = str(path)
            df["merge_priority_rank"] = priority

            df = align_df_to_schema(df, ordered_cols, writer_types)
            table = pa.Table.from_pandas(df, schema=writer_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), writer_schema, compression=compression, use_dictionary=True)
            writer.write_table(table)
            src_rows += len(df)

        summary.append({"source_name": spec.name, "rows_written": src_rows})
        eprint(f"[ROI-SQLITE WRITE] {spec.name}: rows_written={src_rows:,}")

    if writer is None:
        raise SystemExit("No rows written for roi_timecourses.parquet")
    writer.close()
    conn.close()

    report_dir = output_dir / "merge_reports" / "roi_timecourses"
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(report_dir / "sqlite_write_summary.csv", index=False)
    return out_path

def validate_coverage(output_aggregate_dir: Path, output_dir: Path, batch_rows: int) -> None:
    table_name = "roi_timecourses.parquet"
    path = output_aggregate_dir / table_name
    if not path.exists():
        return
    dataset = ds.dataset(str(path), format="parquet")
    cols = safe_columns(dataset, ["subject", "file_label", "pipeline_label", "merged_from_source", "chromophore", "target_status", "time_s"])
    scanner = dataset.scanner(columns=cols, batch_size=batch_rows)
    cov_parts = []
    dup_parts = []
    for rb in scanner.to_batches():
        df = rb.to_pandas(types_mapper=pd.ArrowDtype)
        if "pipeline_label" in df.columns:
            df["pipeline_label"] = df["pipeline_label"].map(norm_text)
        grp_cols = [c for c in ["subject", "file_label", "pipeline_label", "chromophore", "target_status", "time_s"] if c in df.columns]
        if grp_cols:
            dup = df.groupby(grp_cols).size().reset_index(name="n")
            dup = dup.loc[dup["n"] > 1]
            if not dup.empty:
                dup_parts.append(dup)
        use = [c for c in ["subject", "file_label", "pipeline_label", "merged_from_source"] if c in df.columns]
        if use:
            cov_parts.append(df[use].drop_duplicates())

    report_dir = output_dir / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    if cov_parts:
        cov = pd.concat(cov_parts, ignore_index=True).drop_duplicates()
        subj_file = cov.groupby(["subject", "file_label"])["pipeline_label"].nunique().reset_index(name="n_pipelines_present")
        subj_file.to_csv(report_dir / "roi_timecourses_subject_file_pipeline_counts.csv", index=False)
        by_pipeline = cov.groupby("pipeline_label")["subject"].count().reset_index(name="n_subject_file_groups")
        by_pipeline.to_csv(report_dir / "roi_timecourses_pipeline_counts.csv", index=False)
        by_source = cov.groupby(["pipeline_label", "merged_from_source"])["subject"].count().reset_index(name="n_subject_file_groups")
        by_source.to_csv(report_dir / "roi_timecourses_pipeline_source_breakdown.csv", index=False)
    if dup_parts:
        dup_df = pd.concat(dup_parts, ignore_index=True)
    else:
        dup_df = pd.DataFrame(columns=["subject", "file_label", "pipeline_label", "chromophore", "target_status", "time_s", "n"])
    dup_df.to_csv(report_dir / "roi_timecourses_duplicates.csv", index=False)
    eprint("[VALIDATE] wrote validation reports")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge fragmented fNIRS aggregate folders into one parquet-only aggregate.")
    p.add_argument("--parent-dir", type=str, required=True, help="Directory containing the six unzipped source folders.")
    p.add_argument("--output-dir", type=str, required=True, help="Directory to write the merged aggregate folder and reports.")
    p.add_argument("--batch-rows", type=int, default=250000)
    p.add_argument("--compression", type=str, default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"])
    p.add_argument("--old86-dir", type=str, default=None)
    p.add_argument("--python-rerun-dir", type=str, default=None)
    p.add_argument("--gaussian-dir", type=str, default=None)
    p.add_argument("--rerun-dir", type=str, default=None)
    p.add_argument("--repair-dir", type=str, default=None)
    p.add_argument("--additional-dir", type=str, default=None)
    p.add_argument("--tables", nargs="*", default=None, help="Optional explicit parquet table names to merge.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_aggregate_dir = output_dir / "aggregate"
    output_aggregate_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "merge_reports").mkdir(parents=True, exist_ok=True)

    overrides = {
        "old86": args.old86_dir,
        "python_rerun": args.python_rerun_dir,
        "gaussian": args.gaussian_dir,
        "rerun": args.rerun_dir,
        "repair": args.repair_dir,
        "additional": args.additional_dir,
    }
    sources = discover_sources(parent_dir, overrides)

    inv = source_inventory_df(sources)
    inv.to_csv(output_dir / "source_inventory.csv", index=False)
    with open(output_dir / "merge_policy.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_priority": SOURCE_PRIORITY,
                "global_exclude_pipelines": sorted(GLOBAL_EXCLUDE_PIPELINES),
                "keep_fir_only_from": sorted(KEEP_FIR_ONLY_FROM),
                "note": "LocalSS_FIR_AUTO is retained only from the trusted Python sources (old86 and python_rerun); the mixed Python+AnalyzIR derivative ARIRLS pipelines are excluded globally; trusted native AnalyzIR pipelines come only from rerun/repair; NoSS_FIR_OLS_NoTDDR remains available from the additional AnalyzIR run as a secondary comparison; and Homer3_tCCA_Gaussian comes only from gaussian_aggregate.",
                "source_allowed_pipelines": {k: sorted(v) for k, v in SOURCE_ALLOWED_PIPELINES.items()},
                "special_tables": sorted(SPECIAL_TABLES),
                "identity_column_candidates": IDENTITY_COLUMN_CANDIDATES,
            },
            f,
            indent=2,
        )

    compression = None if args.compression == "none" else args.compression
    table_names = args.tables or get_all_parquet_names(sources)

    table_inventory_rows = []
    for table_name in table_names:
        present_in = [spec.name for spec in sources if (spec.aggregate_dir / table_name).exists()]
        table_inventory_rows.append({
            "table_name": table_name,
            "present_in_sources": ", ".join(present_in),
            "n_sources_present": len(present_in),
        })
    pd.DataFrame(table_inventory_rows).sort_values("table_name").to_csv(output_dir / "table_inventory.csv", index=False)

    for table_name in table_names:
        eprint(f"\n=== MERGING {table_name} ===")
        if table_name == "pipeline_manifest.parquet":
            merge_pipeline_manifest(sources, output_aggregate_dir, output_dir, args.batch_rows, compression)
            continue
        if table_name == "config.parquet":
            merge_config(sources, output_aggregate_dir, output_dir, args.batch_rows, compression)
            continue
        if table_name in SMALL_ROW_DEDUP_TABLES:
            merge_small_row_dedup_table(table_name, sources, output_aggregate_dir, args.batch_rows, compression)
            continue
        if table_name == "roi_timecourses.parquet":
            eprint("[SPECIAL] using SQLite-backed merge path for roi_timecourses.parquet")
            roi_col_types = build_roi_timecourses_sqlite_index(sources, args.batch_rows, output_dir)
            write_roi_timecourses_from_sqlite(
                sources=sources,
                output_aggregate_dir=output_aggregate_dir,
                output_dir=output_dir,
                batch_rows=args.batch_rows,
                compression=compression,
                col_types=roi_col_types,
            )
            continue

        winners, identity_cols = build_group_winners(table_name, sources, args.batch_rows, output_dir)
        if winners.empty:
            eprint(f"[SKIP] {table_name}: no rows survived pipeline filters")
            continue
        stream_group_winners(
            table_name=table_name,
            sources=sources,
            winners=winners,
            identity_cols=identity_cols,
            batch_rows=args.batch_rows,
            output_aggregate_dir=output_aggregate_dir,
            compression=compression,
        )


    # Structural completeness check for downstream analysis notebook.
    required_tables = [
        "shape_fidelity.parquet",
        "shape_fidelity_summary.parquet",
        "target_vs_nontarget_summary.parquet",
        "truth_summary.parquet",
        "variability_summary.parquet",
        "roi_timecourses.parquet",
        "roi_scores.parquet",
        "canonical_channel_metrics.parquet",
        "channel_availability.parquet",
        "pipeline_manifest.parquet",
        "config.parquet",
    ]
    missing_required = [t for t in required_tables if not (output_aggregate_dir / t).exists()]
    pd.DataFrame({"missing_required_table": missing_required}).to_csv(
        output_dir / "required_table_check.csv", index=False
    )
    if missing_required:
        eprint(f"[WARN] missing required output tables: {missing_required}")

    validate_coverage(output_aggregate_dir, output_dir, args.batch_rows)
    eprint(f"\n[DONE] merged aggregate written to: {output_aggregate_dir}")


if __name__ == "__main__":
    main()
