#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io

# Keep BLAS/OpenMP from oversubscribing worker processes.
for env_key in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(env_key, "1")


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FileSpec:
    label: str
    filename: str
    amplitude_value: int
    is_null: bool = False
    annotation_source_filename: Optional[str] = None


@dataclass(frozen=True)
class PipelineSpec:
    label: str
    backend: str
    nuisance_method: str
    hrf_model: str
    solver: str
    pruning_style: str
    do_tddr: bool = True
    use_block_average: bool = False
    include_in_empirical_null: bool = False
    include_in_primary_variability: bool = False
    secondary_pipeline: bool = False
    description: str = ""


@dataclass
class BenchmarkConfig:
    # Paths
    root: str = str(Path.home() / "fnirs-representation-learning")
    dataset_dirname: str = "snirf_dataset_2"
    output_dirname: str = "outputs_benchmark_v6"
    output_prefix: str = "benchmark_v6"

    # Dataset
    subject_names: list[str] = field(default_factory=lambda: [
        "Subj86", "Subj91", "Subj92", "Subj94", "Subj95", "Subj96",
        "Subj97", "Subj98", "Subj99", "Subj100", "Subj101",
        "Subj102", "Subj103", "Subj104",
    ])
    file_specs: list[FileSpec] = field(default_factory=list)
    pipeline_specs: list[PipelineSpec] = field(default_factory=list)

    # Channel geometry / QC
    short_separation_threshold_m: float = 0.015
    long_separation_threshold_m: float = 0.025
    local_ss_max_distance_m: float = 0.015
    multi_ss_k: int = 3
    pooled_ss_n_components: int = 2
    ss_aux_n_components: int = 3

    strict_sci_threshold: float = 0.50
    loose_sci_threshold: float = 0.35
    strict_snr_threshold: float = 2.0
    strict_negative_fraction_threshold: float = 0.001

    # Signal processing
    filter_low_hz: float = 0.01
    filter_high_hz: float = 0.20
    ppf_value: float = 0.1
    stim_duration_s: float = 1.0
    drift_high_pass_hz: float = 0.01

    # Epoch / waveform summary
    epoch_tmin: float = -5.0
    epoch_tmax: float = 30.0
    baseline_window: tuple[float, float] = (-5.0, 0.0)
    response_window: tuple[float, float] = (4.0, 8.0)

    # FIR
    fir_resample_sfreq_hz: float = 1.0
    fir_delays_scans: list[int] = field(default_factory=lambda: list(range(0, 26)))

    # Null
    empirical_null_shift_count: int = 50
    empirical_null_min_shift_s: float = 20.0

    # Execution
    n_workers: int = max(1, min(os.cpu_count() or 1, 8))
    overwrite: bool = False
    write_csv: bool = True
    write_parquet: bool = True

    # MATLAB / AnalyzIR
    use_matlab: bool = False
    matlab_cmd: str = "matlab"
    matlab_timeout_s: int = 7200
    analyzir_path: Optional[str] = None

    # Truth templates
    truth_template_dir: Optional[str] = None
    truth_template_map: dict[str, str] = field(default_factory=lambda: {
        "hrf_20": "hrf_20.mat",
        "hrf_50": "hrf_50.mat",
        "hrf_100": "hrf_100.mat",
    })

    def root_path(self) -> Path:
        return Path(self.root).expanduser().resolve()

    def dataset_path(self) -> Path:
        return self.root_path() / self.dataset_dirname

    def output_path(self) -> Path:
        return self.root_path() / self.output_dirname

    def jobs_path(self) -> Path:
        return self.output_path() / "job_results"

    def aggregate_path(self) -> Path:
        return self.output_path() / "aggregate"

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["file_specs"] = [asdict(item) for item in self.file_specs]
        payload["pipeline_specs"] = [asdict(item) for item in self.pipeline_specs]
        return payload


# -----------------------------------------------------------------------------
# Simple helpers
# -----------------------------------------------------------------------------


class BenchmarkError(RuntimeError):
    pass


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_write_table(df: pd.DataFrame, path_base: Path, write_csv: bool = True, write_parquet: bool = True) -> None:
    if df is None or len(df) == 0:
        return
    if write_csv:
        df.to_csv(path_base.with_suffix(".csv"), index=False)
    if write_parquet:
        try:
            df.to_parquet(path_base.with_suffix(".parquet"), index=False)
        except Exception as exc:
            path_base.with_name(path_base.name + "__parquet_failed.txt").write_text(str(exc), encoding="utf-8")


def read_any_table(path_base: Path) -> pd.DataFrame:
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(len(p_values), np.nan)
    valid_mask = np.isfinite(p_values)
    if valid_mask.sum() == 0:
        return q_values
    valid = p_values[valid_mask]
    order = np.argsort(valid)
    ordered = valid[order]
    n = len(ordered)
    ordered_q = ordered * n / np.arange(1, n + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.clip(ordered_q, 0.0, 1.0)
    unsorted = np.empty_like(ordered_q)
    unsorted[order] = ordered_q
    q_values[valid_mask] = unsorted
    return q_values


def safe_standardize_rows(data_matrix: np.ndarray) -> np.ndarray:
    data_matrix = np.asarray(data_matrix, dtype=float)
    if data_matrix.ndim == 1:
        data_matrix = data_matrix[None, :]
    centered = data_matrix - data_matrix.mean(axis=1, keepdims=True)
    stds = data_matrix.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return centered / stds


def principal_components_rows(data_matrix: np.ndarray, n_components: int) -> Optional[np.ndarray]:
    data_matrix = safe_standardize_rows(data_matrix)
    if data_matrix.shape[0] == 0:
        return None
    if data_matrix.shape[0] == 1:
        return data_matrix.T
    u, s, _ = np.linalg.svd(data_matrix.T, full_matrices=False)
    k = min(n_components, u.shape[1], data_matrix.shape[0])
    if k <= 0:
        return None
    return u[:, :k] * s[:k]


def qr_orth_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] == 0:
        return matrix
    q, r = np.linalg.qr(matrix)
    keep = np.abs(np.diag(r)) > 1e-10
    if keep.ndim == 0:
        keep = np.asarray([bool(keep)])
    return q[:, keep]


def make_error_row(subject: str, file_label: Optional[str], pipeline_label: Optional[str], stage: str,
                   error_message: str, channel_name: Optional[str] = None) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc_iso(),
        "subject": subject,
        "file_label": file_label,
        "pipeline_label": pipeline_label,
        "stage": stage,
        "channel_name": channel_name,
        "error_message": str(error_message),
    }


def default_file_specs() -> list[FileSpec]:
    return [
        FileSpec(label="no_hrf", filename="resting_clean.snirf", amplitude_value=0, is_null=True,
                 annotation_source_filename="resting_hrf_20.snirf"),
        FileSpec(label="hrf_20", filename="resting_hrf_20.snirf", amplitude_value=20),
        FileSpec(label="hrf_50", filename="resting_hrf_50.snirf", amplitude_value=50),
        FileSpec(label="hrf_100", filename="resting_hrf_100.snirf", amplitude_value=100),
    ]


def default_pipeline_specs() -> list[PipelineSpec]:
    # Maxed curated set: serious nuisance family + HRF family + solver family + secondary contrasts.
    return [
        # Nuisance-model core family
        PipelineSpec("NoSS_Glover_AUTO", "python", "none", "glover", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("LocalSS_Glover_AUTO", "python", "local_nearest", "glover", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("PooledPCA2_Glover_AUTO", "python", "pooled_pca2", "glover", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("SSAuxPCA_Glover_AUTO", "python", "ss_aux_pca", "glover", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("MultiSSOrth3_Glover_AUTO", "python", "multi_ss_orth3", "glover", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),

        # Solver/statistics contrasts
        PipelineSpec("NoSS_Glover_OLS", "python", "none", "glover", "ols", "strict_combined"),
        PipelineSpec("LocalSS_Glover_OLS", "python", "local_nearest", "glover", "ols", "strict_combined",
                     include_in_primary_variability=True),

        # HRF-family contrasts
        PipelineSpec("LocalSS_SPM_AUTO", "python", "local_nearest", "spm", "auto", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("LocalSS_SPM_OLS", "python", "local_nearest", "spm", "ols", "strict_combined"),
        PipelineSpec("LocalSS_FIR_AUTO", "python", "local_nearest", "fir", "auto", "strict_combined",
                     include_in_primary_variability=True),

        # Secondary contrasts
        PipelineSpec("BlockAvg_LocalSS", "python", "local_nearest", "block_average", "none", "strict_combined",
                     use_block_average=True, secondary_pipeline=True),
        PipelineSpec("LooseQC_LocalSS_Glover_AUTO", "python", "local_nearest", "glover", "auto", "loose_sci",
                     secondary_pipeline=True),

        # MATLAB / AnalyzIR AR-IRLS family
        PipelineSpec("LocalSS_Glover_ARIRLS", "matlab_arirls", "local_nearest", "glover", "arirls", "strict_combined",
                     include_in_empirical_null=True, include_in_primary_variability=True),
        PipelineSpec("NoSS_Glover_ARIRLS", "matlab_arirls", "none", "glover", "arirls", "strict_combined",
                     include_in_empirical_null=True),
        PipelineSpec("LocalSS_Gamma_ARIRLS", "matlab_arirls", "local_nearest", "gamma", "arirls", "strict_combined",
                     include_in_primary_variability=True),
    ]


# -----------------------------------------------------------------------------
# Dynamic imports for MNE-NIRS / Nilearn
# -----------------------------------------------------------------------------


def import_mne_nirs_modules():
    try:
        from mne_nirs.experimental_design import make_first_level_design_matrix
        from mne_nirs.statistics import run_glm
        from mne_nirs.io import read_snirf_aux_data
    except Exception as exc:
        raise BenchmarkError(
            "mne_nirs is required for this benchmark. Install mne-nirs (and nilearn) in the target environment."
        ) from exc
    return make_first_level_design_matrix, run_glm, read_snirf_aux_data


def import_nilearn_functions():
    try:
        from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix
    except Exception as exc:
        raise BenchmarkError(
            "nilearn is required for custom HRF basis functions and design-matrix helpers."
        ) from exc
    return compute_regressor, make_first_level_design_matrix


# -----------------------------------------------------------------------------
# Truth-template loading
# -----------------------------------------------------------------------------


def load_truth_templates(config: BenchmarkConfig) -> dict[str, dict[str, np.ndarray]]:
    templates: dict[str, dict[str, np.ndarray]] = {}
    if config.truth_template_dir is None:
        return templates
    template_dir = Path(config.truth_template_dir).expanduser().resolve()
    for file_label, template_name in config.truth_template_map.items():
        template_path = template_dir / template_name
        if not template_path.exists():
            continue
        mat = scipy.io.loadmat(template_path, squeeze_me=True, struct_as_record=False)
        if "hrf" not in mat:
            continue
        hrf = mat["hrf"]
        hrf_conc = np.asarray(hrf.hrf_conc, dtype=float)
        templates[file_label] = {
            "time_s": np.asarray(hrf.t_hrf, dtype=float).reshape(-1),
            "hbo": hrf_conc[:, 0].reshape(-1),
            "hbr": hrf_conc[:, 1].reshape(-1),
            "hbt": hrf_conc[:, 2].reshape(-1),
        }
    return templates


# -----------------------------------------------------------------------------
# SNIRF truth labels and annotation sanitization
# -----------------------------------------------------------------------------


def read_measurement_data_type_labels(snirf_file_path: Path) -> Optional[np.ndarray]:
    if not snirf_file_path.exists():
        return None
    with h5py.File(snirf_file_path, "r") as h5_file:
        if "nirs" not in h5_file:
            return None
        nirs_group = h5_file["nirs"]
        data_group_name = "data1" if "data1" in nirs_group else next((k for k in nirs_group.keys() if k.startswith("data")), None)
        if data_group_name is None:
            return None
        data_group = nirs_group[data_group_name]
        entries: list[tuple[int, int]] = []
        for key in data_group.keys():
            if not key.startswith("measurementList"):
                continue
            suffix = key.replace("measurementList", "")
            try:
                idx = int(suffix)
            except ValueError:
                continue
            measurement_group = data_group[key]
            label_value = measurement_group["dataTypeLabel"][()] if "dataTypeLabel" in measurement_group else 0
            if isinstance(label_value, np.ndarray):
                label_value = 0 if label_value.size == 0 else label_value.reshape(-1)[0]
            try:
                label_value = int(label_value)
            except Exception:
                label_value = 0
            entries.append((idx, label_value))
    if not entries:
        return None
    entries.sort(key=lambda item: item[0])
    return np.asarray([label for _, label in entries], dtype=int)


def get_cw_channel_indices(raw: mne.io.BaseRaw) -> np.ndarray:
    picks_fnirs = mne.pick_types(raw.info, fnirs=True)
    channel_types = np.asarray(raw.get_channel_types())
    return picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]


def sanitize_annotations_to_single_task(raw: mne.io.BaseRaw, default_description: str = "task") -> mne.io.BaseRaw:
    raw = raw.copy()
    kept_onsets, kept_durations, kept_descriptions = [], [], []
    for onset_s, duration_s, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if str(description).lower().startswith("bad"):
            continue
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(duration_s) if float(duration_s) > 0 else 1.0)
        kept_descriptions.append(default_description)
    raw.set_annotations(mne.Annotations(onset=kept_onsets, duration=kept_durations, description=kept_descriptions))
    return raw


def copy_valid_annotations(raw_target_cw: mne.io.BaseRaw, annotation_source_file_path: Path) -> mne.io.BaseRaw:
    raw_source = mne.io.read_raw_snirf(annotation_source_file_path, preload=False, verbose=False)
    source_annotations = raw_source.annotations
    target_duration_s = float(raw_target_cw.times[-1])

    kept_onsets, kept_durations, kept_descriptions = [], [], []
    for onset_s, duration_s, description in zip(source_annotations.onset, source_annotations.duration, source_annotations.description):
        if str(description).lower().startswith("bad"):
            continue
        if float(onset_s) >= target_duration_s:
            continue
        clipped_duration = min(float(duration_s) if float(duration_s) > 0 else 1.0, max(0.0, target_duration_s - float(onset_s)))
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(clipped_duration))
        kept_descriptions.append("task")
    raw_target_cw = raw_target_cw.copy()
    raw_target_cw.set_annotations(mne.Annotations(onset=kept_onsets, duration=kept_durations, description=kept_descriptions))
    return raw_target_cw


# -----------------------------------------------------------------------------
# Channel tables and QC
# -----------------------------------------------------------------------------


def build_cw_channel_table(raw_cw: mne.io.BaseRaw, subject: str, file_label: str, config: BenchmarkConfig) -> pd.DataFrame:
    from mne.preprocessing.nirs import source_detector_distances, short_channels

    picks_cw = get_cw_channel_indices(raw_cw)
    cw_names = np.asarray(raw_cw.ch_names)[picks_cw]
    distances_m = source_detector_distances(raw_cw.info, picks=picks_cw)
    short_mask_all = short_channels(raw_cw.info, threshold=config.short_separation_threshold_m)
    short_mask = short_mask_all[picks_cw]
    long_mask = distances_m >= config.long_separation_threshold_m
    pair_names = np.asarray([name.split(" ")[0] for name in cw_names])

    df = pd.DataFrame({
        "subject": subject,
        "file_label": file_label,
        "channel_name": cw_names,
        "pair_name": pair_names,
        "distance_m": distances_m,
        "is_ss": short_mask,
        "is_ls": long_mask,
    })
    df["group"] = np.select([df["is_ss"], df["is_ls"]], ["SS", "LS"], default="MID")
    return df


def build_hb_channel_table(raw_hb: mne.io.BaseRaw, subject: str, file_label: str, config: BenchmarkConfig) -> pd.DataFrame:
    from mne.preprocessing.nirs import source_detector_distances, short_channels

    picks_hbo = mne.pick_types(raw_hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(raw_hb.info, fnirs="hbr")
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))

    hb_names = np.asarray(raw_hb.ch_names)[picks_hb]
    hb_types = np.asarray(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.asarray([name.split(" ")[0] for name in hb_names])
    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]
    short_mask_all = short_channels(raw_hb.info, threshold=config.short_separation_threshold_m)
    short_mask = short_mask_all[picks_hb]
    long_mask = distances_hb >= config.long_separation_threshold_m

    df = pd.DataFrame({
        "subject": subject,
        "file_label": file_label,
        "channel_name": hb_names,
        "pair_name": pair_names,
        "chromophore": hb_types,
        "distance_m": distances_hb,
        "is_ss": short_mask,
        "is_ls": long_mask,
    })
    df["group"] = np.select([df["is_ss"], df["is_ls"]], ["SS", "LS"], default="MID")
    return df


def build_quality_tables(raw_cw: mne.io.BaseRaw, subject: str, file_label: str, config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    from mne.preprocessing.nirs import optical_density, scalp_coupling_index

    raw_od = optical_density(raw_cw.copy())
    picks_cw = get_cw_channel_indices(raw_cw)
    cw_channel_table = build_cw_channel_table(raw_cw, subject, file_label, config)
    cw_names = np.asarray(raw_cw.ch_names)[picks_cw]
    cw_data = raw_cw.get_data(picks=picks_cw)

    sci_values = scalp_coupling_index(raw_od)
    mean_abs_signal = np.mean(np.abs(cw_data), axis=1)
    signal_std = np.std(cw_data, axis=1)
    snr_values = mean_abs_signal / np.maximum(signal_std, 1e-12)
    negative_fraction_values = np.mean(cw_data <= 0, axis=1)

    channel_quality = pd.DataFrame({
        "subject": subject,
        "file_label": file_label,
        "channel_name": cw_names,
        "pair_name": cw_channel_table["pair_name"].to_numpy(),
        "distance_m": cw_channel_table["distance_m"].to_numpy(),
        "group": cw_channel_table["group"].to_numpy(),
        "sci": sci_values,
        "snr": snr_values,
        "negative_fraction": negative_fraction_values,
    })

    pair_quality = (
        channel_quality
        .groupby(["subject", "file_label", "pair_name"], as_index=False)
        .agg(
            distance_m=("distance_m", "first"),
            group=("group", "first"),
            sci_min=("sci", "min"),
            sci_mean=("sci", "mean"),
            snr_min=("snr", "min"),
            snr_mean=("snr", "mean"),
            negative_fraction_max=("negative_fraction", "max"),
        )
    )
    return channel_quality, pair_quality


def get_bad_pair_names(pair_quality: pd.DataFrame, pruning_style: str, config: BenchmarkConfig) -> list[str]:
    if pruning_style == "strict_combined":
        bad_mask = (
            (pair_quality["sci_min"] < config.strict_sci_threshold)
            | (pair_quality["snr_min"] < config.strict_snr_threshold)
            | (pair_quality["negative_fraction_max"] > config.strict_negative_fraction_threshold)
        )
    elif pruning_style == "loose_sci":
        bad_mask = pair_quality["sci_min"] < config.loose_sci_threshold
    else:
        raise ValueError(f"Unknown pruning_style {pruning_style!r}")
    return pair_quality.loc[bad_mask, "pair_name"].astype(str).tolist()


def apply_bad_pairs_to_hb(raw_hb: mne.io.BaseRaw, bad_pair_names: Iterable[str]) -> mne.io.BaseRaw:
    raw_hb = raw_hb.copy()
    bad_pairs = set(bad_pair_names)
    raw_hb.info["bads"] = [name for name in raw_hb.ch_names if name.split(" ")[0] in bad_pairs]
    return raw_hb


def get_channel_position(raw: mne.io.BaseRaw, channel_name: str) -> np.ndarray:
    idx = raw.ch_names.index(channel_name)
    return np.asarray(raw.info["chs"][idx]["loc"][:3], dtype=float)


def get_available_long_channel_names(raw_hb: mne.io.BaseRaw, config: BenchmarkConfig) -> list[str]:
    hb_table = build_hb_channel_table(raw_hb, "", "", config)
    long_names = hb_table.loc[hb_table["group"] == "LS", "channel_name"].astype(str).tolist()
    return sorted([name for name in long_names if name not in raw_hb.info["bads"]])


def get_available_short_channel_names(raw_hb: mne.io.BaseRaw, chromophore: str, config: BenchmarkConfig) -> list[str]:
    hb_table = build_hb_channel_table(raw_hb, "", "", config)
    short_names = hb_table.loc[
        (hb_table["group"] == "SS") & (hb_table["chromophore"] == chromophore),
        "channel_name"
    ].astype(str).tolist()
    return [name for name in short_names if name not in raw_hb.info["bads"]]


# -----------------------------------------------------------------------------
# Auxiliary-signal handling
# -----------------------------------------------------------------------------


def read_auxiliary_dataframe(snirf_file_path: Path, raw_cw: mne.io.BaseRaw) -> pd.DataFrame:
    _, _, read_snirf_aux_data = import_mne_nirs_modules()
    try:
        aux_df = read_snirf_aux_data(str(snirf_file_path), raw_cw)
    except Exception:
        return pd.DataFrame(index=raw_cw.times)
    if aux_df is None or len(aux_df.columns) == 0:
        return pd.DataFrame(index=raw_cw.times)
    aux_df = aux_df.copy()
    # Drop columns that are all-NaN or constant.
    kept_cols = []
    for col in aux_df.columns:
        values = np.asarray(aux_df[col], dtype=float)
        if not np.isfinite(values).any():
            continue
        if np.nanstd(values) < 1e-12:
            continue
        kept_cols.append(col)
    return aux_df[kept_cols] if kept_cols else pd.DataFrame(index=raw_cw.times)


# -----------------------------------------------------------------------------
# Truth / basis functions
# -----------------------------------------------------------------------------


def gamma_hrf_function(t_r: float, oversampling: int = 50) -> np.ndarray:
    peak_time = 6.0
    peak_disp = 1.0
    duration = 32.0
    dt = float(t_r) / float(oversampling)
    t = np.arange(0.0, duration + dt, dt)
    h = (peak_disp ** peak_time) * (t ** (peak_time - 1.0)) * np.exp(-peak_disp * t) / math.gamma(peak_time)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    total = h.sum()
    return h / total if total > 0 else h


def analyzir_canonical_hrf(times_s: np.ndarray) -> np.ndarray:
    a1, a2, b1, b2, c = 4.0, 16.0, 1.0, 1.0, 1.0 / 6.0
    t = np.clip(times_s, 0.0, None)
    h = (b1 ** a1) * (t ** (a1 - 1.0)) * np.exp(-b1 * t) / math.gamma(a1)
    h -= c * (b2 ** a2) * (t ** (a2 - 1.0)) * np.exp(-b2 * t) / math.gamma(a2)
    total = np.sum(h)
    return h / total if abs(total) > 0 else h


def get_truth_curve(templates: dict[str, dict[str, np.ndarray]], file_label: str, epoch_times_s: np.ndarray, chromophore: str) -> np.ndarray:
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    if file_label == "no_hrf":
        return np.zeros_like(epoch_times_s, dtype=float)
    if file_label not in templates:
        return np.full_like(epoch_times_s, np.nan, dtype=float)
    tmpl = templates[file_label]
    return np.interp(epoch_times_s, tmpl["time_s"], tmpl[chromophore], left=0.0, right=0.0)


def build_basis_curve(epoch_times_s: np.ndarray, hrf_model: str,
                     fir_delays: Optional[list[int]] = None,
                     fir_betas: Optional[np.ndarray] = None) -> np.ndarray:
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    compute_regressor, _ = import_nilearn_functions()

    if hrf_model in {"glover", "spm"}:
        signal, _ = compute_regressor(
            np.array([[0.0], [1.0], [1.0]]),
            hrf_model,
            epoch_times_s,
            con_id="task",
            oversampling=50,
        )
        return signal[:, 0]
    if hrf_model == "gamma":
        dt = np.median(np.diff(epoch_times_s))
        signal, _ = compute_regressor(
            np.array([[0.0], [1.0], [1.0]]),
            [gamma_hrf_function],
            epoch_times_s,
            con_id="task",
            oversampling=50,
        )
        return signal[:, 0]
    if hrf_model == "canonical":
        return analyzir_canonical_hrf(epoch_times_s)
    if hrf_model == "fir":
        if fir_delays is None or fir_betas is None:
            raise ValueError("FIR basis requires delays and betas.")
        return np.interp(epoch_times_s, np.asarray(fir_delays, dtype=float), np.asarray(fir_betas, dtype=float), left=0.0, right=0.0)
    raise ValueError(f"Unknown hrf_model {hrf_model!r}")


def peak_value_and_time(signal: np.ndarray, time_s: np.ndarray, chromophore: str) -> tuple[float, float]:
    if chromophore == "hbo":
        idx = int(np.nanargmax(signal))
    else:
        idx = int(np.nanargmin(signal))
    return float(signal[idx]), float(time_s[idx])


def compute_shape_metrics(recovered_curve: np.ndarray, truth_curve: np.ndarray,
                          epoch_times_s: np.ndarray, chromophore: str) -> dict[str, float]:
    recovered_curve = np.asarray(recovered_curve, dtype=float)
    truth_curve = np.asarray(truth_curve, dtype=float)
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    mask = np.isfinite(recovered_curve) & np.isfinite(truth_curve)
    if mask.sum() < 3:
        return {k: np.nan for k in [
            "curve_corr", "curve_rmse", "curve_nrmse",
            "peak_latency_error_s", "peak_amplitude_bias",
            "peak_amplitude_ratio", "auc_bias"
        ]}
    rec = recovered_curve[mask]
    tru = truth_curve[mask]
    t = epoch_times_s[mask]
    corr = np.nan if (np.std(rec) < 1e-12 or np.std(tru) < 1e-12) else float(np.corrcoef(rec, tru)[0, 1])
    rmse = float(np.sqrt(np.mean((rec - tru) ** 2)))
    scale = float(np.max(tru) - np.min(tru))
    nrmse = float(rmse / scale) if scale > 1e-12 else np.nan
    rec_peak, rec_t = peak_value_and_time(rec, t, chromophore)
    tru_peak, tru_t = peak_value_and_time(tru, t, chromophore)
    peak_ratio = float(rec_peak / tru_peak) if abs(tru_peak) > 1e-12 else np.nan
    auc_bias = float(np.trapz(rec, t) - np.trapz(tru, t))
    return {
        "curve_corr": corr,
        "curve_rmse": rmse,
        "curve_nrmse": nrmse,
        "peak_latency_error_s": float(rec_t - tru_t),
        "peak_amplitude_bias": float(rec_peak - tru_peak),
        "peak_amplitude_ratio": peak_ratio,
        "auc_bias": auc_bias,
    }


# -----------------------------------------------------------------------------
# Nuisance-regressor factories
# -----------------------------------------------------------------------------


@dataclass
class NuisanceBundle:
    design_df: Optional[pd.DataFrame]
    metadata: dict[str, Any]


@dataclass
class GlobalNuisanceBundle:
    by_chromophore: dict[str, NuisanceBundle]


@dataclass
class ChannelSpecificNuisanceFactory:
    channel_to_bundle: dict[str, NuisanceBundle]


def build_global_nuisance_bundle(raw_hb: mne.io.BaseRaw, snirf_file_path: Path, raw_cw: mne.io.BaseRaw,
                                 nuisance_method: str, config: BenchmarkConfig) -> GlobalNuisanceBundle:
    by_chrom: dict[str, NuisanceBundle] = {}
    aux_df = read_auxiliary_dataframe(snirf_file_path, raw_cw)
    aux_data = aux_df.to_numpy().T if len(aux_df.columns) > 0 else None
    for chrom in ["hbo", "hbr"]:
        if nuisance_method == "none":
            by_chrom[chrom] = NuisanceBundle(None, {"nuisance_method_used": "none"})
            continue

        if nuisance_method == "pooled_pca2":
            short_names = get_available_short_channel_names(raw_hb, chrom, config)
            if not short_names:
                by_chrom[chrom] = NuisanceBundle(None, {"nuisance_method_used": "none_no_short_channels"})
                continue
            short_data = raw_hb.copy().pick(short_names).get_data()
            pcs = principal_components_rows(short_data, config.pooled_ss_n_components)
            if pcs is None:
                by_chrom[chrom] = NuisanceBundle(None, {"nuisance_method_used": "none_failed_pca"})
                continue
            df = pd.DataFrame(pcs, columns=[f"ss_pc{i+1}_{chrom}" for i in range(pcs.shape[1])])
            by_chrom[chrom] = NuisanceBundle(df, {
                "nuisance_method_used": "pooled_pca2",
                "n_short_channels_used": len(short_names),
                "n_nuisance_components": pcs.shape[1],
            })
            continue

        if nuisance_method == "ss_aux_pca":
            blocks = []
            short_names = get_available_short_channel_names(raw_hb, chrom, config)
            if short_names:
                blocks.append(raw_hb.copy().pick(short_names).get_data())
            if aux_data is not None:
                blocks.append(aux_data)
            if not blocks:
                by_chrom[chrom] = NuisanceBundle(None, {"nuisance_method_used": "none_no_short_or_aux"})
                continue
            pcs = principal_components_rows(np.vstack(blocks), config.ss_aux_n_components)
            if pcs is None:
                by_chrom[chrom] = NuisanceBundle(None, {"nuisance_method_used": "none_failed_pca"})
                continue
            df = pd.DataFrame(pcs, columns=[f"aux_pc{i+1}_{chrom}" for i in range(pcs.shape[1])])
            by_chrom[chrom] = NuisanceBundle(df, {
                "nuisance_method_used": "ss_aux_pca",
                "n_short_channels_used": 0 if not short_names else len(short_names),
                "n_aux_channels_used": 0 if aux_data is None else aux_data.shape[0],
                "aux_channel_names": "|".join(aux_df.columns.astype(str).tolist()) if len(aux_df.columns) > 0 else "",
                "n_nuisance_components": pcs.shape[1],
            })
            continue

        raise ValueError(f"Unsupported global nuisance_method {nuisance_method!r}")
    return GlobalNuisanceBundle(by_chrom)


def build_channel_specific_nuisance_factory(raw_hb: mne.io.BaseRaw, nuisance_method: str,
                                            config: BenchmarkConfig) -> ChannelSpecificNuisanceFactory:
    bundles: dict[str, NuisanceBundle] = {}
    for channel_name in get_available_long_channel_names(raw_hb, config):
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"

        if nuisance_method == "none":
            bundles[channel_name] = NuisanceBundle(None, {"nuisance_method_used": "none"})
            continue

        short_names = get_available_short_channel_names(raw_hb, chrom, config)
        if not short_names:
            bundles[channel_name] = NuisanceBundle(None, {"nuisance_method_used": "none_no_short_channels"})
            continue

        long_pos = get_channel_position(raw_hb, channel_name)
        distance_rows = []
        for short_name in short_names:
            short_pos = get_channel_position(raw_hb, short_name)
            distance_rows.append((short_name, float(np.linalg.norm(long_pos - short_pos))))
        distance_rows.sort(key=lambda item: item[1])

        if nuisance_method == "local_nearest":
            nearest_name, nearest_dist = distance_rows[0]
            if nearest_dist <= config.local_ss_max_distance_m:
                reg = raw_hb.copy().pick([nearest_name]).get_data()[0]
                df = pd.DataFrame({f"ss_local_{chrom}": reg})
                bundles[channel_name] = NuisanceBundle(df, {
                    "nuisance_method_used": "nearest_short_channel",
                    "nuisance_regressor_label": nearest_name,
                    "nearest_short_distance_m": nearest_dist,
                    "n_short_channels_used": 1,
                })
            else:
                reg = raw_hb.copy().pick(short_names).get_data().mean(axis=0)
                df = pd.DataFrame({f"ss_pooled_fallback_{chrom}": reg})
                bundles[channel_name] = NuisanceBundle(df, {
                    "nuisance_method_used": "pooled_fallback_average",
                    "nuisance_regressor_label": "pooled_fallback_average",
                    "nearest_short_distance_m": nearest_dist,
                    "n_short_channels_used": len(short_names),
                })
            continue

        if nuisance_method == "multi_ss_orth3":
            selected = distance_rows[: max(1, min(config.multi_ss_k, len(distance_rows)))]
            selected_names = [name for name, _ in selected]
            selected_data = raw_hb.copy().pick(selected_names).get_data().T
            selected_data = (selected_data - selected_data.mean(axis=0, keepdims=True)) / np.maximum(selected_data.std(axis=0, keepdims=True), 1e-12)
            q = qr_orth_columns(selected_data)
            if q.size == 0:
                bundles[channel_name] = NuisanceBundle(None, {"nuisance_method_used": "none_failed_qr"})
            else:
                df = pd.DataFrame(q, columns=[f"ss_qr{i+1}_{chrom}" for i in range(q.shape[1])])
                bundles[channel_name] = NuisanceBundle(df, {
                    "nuisance_method_used": "multi_ss_orth3",
                    "n_short_channels_used": len(selected_names),
                    "n_nuisance_components": q.shape[1],
                    "nearest_short_distance_m": selected[0][1],
                    "short_regressor_labels": "|".join(selected_names),
                })
            continue

        raise ValueError(f"Unsupported channel-specific nuisance_method {nuisance_method!r}")

    return ChannelSpecificNuisanceFactory(bundles)


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------


def preprocess_raw_to_hb(raw_cw: mne.io.BaseRaw, pair_quality: pd.DataFrame,
                         pipeline: PipelineSpec, config: BenchmarkConfig) -> tuple[mne.io.BaseRaw, list[str]]:
    from mne.preprocessing.nirs import optical_density, beer_lambert_law
    from mne.preprocessing.nirs import temporal_derivative_distribution_repair as tddr

    bad_pair_names = get_bad_pair_names(pair_quality, pipeline.pruning_style, config)
    raw_od = optical_density(raw_cw.copy())
    processed_od = tddr(raw_od.copy()) if pipeline.do_tddr else raw_od.copy()
    processed_od = processed_od.copy().filter(config.filter_low_hz, config.filter_high_hz, verbose=False)
    raw_hb = beer_lambert_law(processed_od, ppf=config.ppf_value)
    raw_hb = apply_bad_pairs_to_hb(raw_hb, bad_pair_names)
    return raw_hb, bad_pair_names


# -----------------------------------------------------------------------------
# GLM utilities
# -----------------------------------------------------------------------------


def standardize_glm_dataframe(glm_df: pd.DataFrame) -> pd.DataFrame:
    glm_df = glm_df.copy()
    glm_df.columns = [str(col).strip().lower() for col in glm_df.columns]
    return glm_df


def find_first_matching_column(column_names: Iterable[str], candidates: list[str]) -> Optional[str]:
    names = [str(c) for c in column_names]
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def get_task_regressor_names(design_matrix: pd.DataFrame) -> list[str]:
    names = []
    for col in design_matrix.columns:
        s = str(col)
        if s == "constant" or s.startswith("drift") or s.startswith("ss_") or s.startswith("aux_") or s.startswith("reg_"):
            continue
        names.append(s)
    return names


def build_design_matrix(raw_hb: mne.io.BaseRaw, hrf_model: str, nuisance_df: Optional[pd.DataFrame],
                        config: BenchmarkConfig) -> pd.DataFrame:
    make_first_level_design_matrix, _, _ = import_mne_nirs_modules()
    add_regs = None if nuisance_df is None else nuisance_df
    add_names = None
    if hrf_model in {"glover", "spm", "fir"}:
        fir_delays = config.fir_delays_scans if hrf_model == "fir" else None
        return make_first_level_design_matrix(
            raw_hb,
            stim_dur=config.stim_duration_s,
            hrf_model=hrf_model,
            drift_model="cosine",
            high_pass=config.drift_high_pass_hz,
            fir_delays=fir_delays,
            add_regs=add_regs,
            add_reg_names=add_names,
            oversampling=1 if hrf_model == "fir" else 50,
        )

    if hrf_model == "gamma":
        # Use Nilearn directly so that custom gamma HRF stays aligned with the design-matrix logic.
        _, nilearn_make_dm = import_nilearn_functions()
        events = pd.DataFrame({
            "trial_type": list(raw_hb.annotations.description),
            "onset": raw_hb.annotations.onset - raw_hb.first_time,
            "duration": config.stim_duration_s * np.ones(len(raw_hb.annotations)),
        })
        return nilearn_make_dm(
            frame_times=raw_hb.times,
            events=events,
            drift_model="cosine",
            high_pass=config.drift_high_pass_hz,
            hrf_model=[gamma_hrf_function],
            add_regs=add_regs,
            add_reg_names=None if nuisance_df is None else list(nuisance_df.columns),
            oversampling=50,
        )

    raise ValueError(f"Unsupported hrf_model {hrf_model!r}")


def parse_mne_glm_to_channel_rows(glm_df: pd.DataFrame, design_matrix: pd.DataFrame, channel_names: list[str],
                                  subject: str, file_label: str, amplitude_value: int,
                                  pipeline: PipelineSpec, target_pair_names: set[str],
                                  nuisance_metadata_by_channel: dict[str, dict[str, Any]]) -> pd.DataFrame:
    glm_df = standardize_glm_dataframe(glm_df)
    ch_col = find_first_matching_column(glm_df.columns, ["ch_name", "channel", "name"])
    cond_col = find_first_matching_column(glm_df.columns, ["condition", "cond", "regressor", "variable", "name"])
    beta_col = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])
    t_col = find_first_matching_column(glm_df.columns, ["t", "t_value", "tstat", "t_stat"])
    p_col = find_first_matching_column(glm_df.columns, ["p_value", "pvalue", "p"])
    if beta_col is None or cond_col is None:
        raise BenchmarkError("Could not parse GLM dataframe columns.")
    task_names = set(get_task_regressor_names(design_matrix))
    rows = []
    for _, row in glm_df.iterrows():
        condition = str(row[cond_col])
        if condition not in task_names and not any(condition.startswith(name) for name in task_names):
            continue
        channel_name = str(row[ch_col]) if ch_col is not None else None
        if channel_name is None or channel_name not in channel_names:
            continue
        pair_name = channel_name.split(" ")[0]
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        out = {
            "subject": subject,
            "file_label": file_label,
            "amplitude_value": amplitude_value,
            "pipeline_label": pipeline.label,
            "backend": pipeline.backend,
            "hrf_model": pipeline.hrf_model,
            "solver": pipeline.solver,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chrom,
            "target_status": "true_target" if pair_name in target_pair_names else "true_non_target",
            "task_regressor": condition,
            "beta": float(row[beta_col]),
            **nuisance_metadata_by_channel.get(channel_name, {}),
        }
        if t_col is not None:
            out["t_value"] = float(row[t_col])
        if p_col is not None:
            out["p_value"] = float(row[p_col])
        rows.append(out)
    return pd.DataFrame(rows)


def resample_for_fir(raw_hb: mne.io.BaseRaw, config: BenchmarkConfig) -> mne.io.BaseRaw:
    raw_fir = raw_hb.copy().load_data()
    raw_fir.resample(config.fir_resample_sfreq_hz, npad="auto")
    raw_fir = sanitize_annotations_to_single_task(raw_fir)
    return raw_fir


def make_epochs(raw_hb: mne.io.BaseRaw, config: BenchmarkConfig) -> mne.Epochs:
    events, event_id = mne.events_from_annotations(raw_hb, verbose=False)
    if len(events) == 0:
        raise BenchmarkError("No events available for epoching.")
    return mne.Epochs(
        raw_hb,
        events=events,
        event_id=event_id,
        tmin=config.epoch_tmin,
        tmax=config.epoch_tmax,
        baseline=config.baseline_window,
        preload=True,
        detrend=None,
        reject_by_annotation=False,
        verbose=False,
    )


# -----------------------------------------------------------------------------
# Block averaging
# -----------------------------------------------------------------------------


def apply_channelwise_nuisance_regression(raw_hb: mne.io.BaseRaw, nuisance_factory: ChannelSpecificNuisanceFactory) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    denoised = raw_hb.copy().load_data()
    meta_rows = []
    for channel_name, bundle in nuisance_factory.channel_to_bundle.items():
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        meta_rows.append({
            "channel_name": channel_name,
            "chromophore": chrom,
            **bundle.metadata,
        })
        if bundle.design_df is None or bundle.design_df.shape[1] == 0:
            continue
        X = np.column_stack([np.ones(len(denoised.times)), bundle.design_df.to_numpy()])
        y = denoised.copy().pick([channel_name]).get_data()[0]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        fitted = X @ beta
        cleaned = y - fitted + beta[0]
        denoised._data[denoised.ch_names.index(channel_name), :] = cleaned
    return denoised, pd.DataFrame(meta_rows)


def compute_block_average_channel_metrics(epochs_hb: mne.Epochs, channel_names: list[str], subject: str,
                                          file_label: str, amplitude_value: int, pipeline: PipelineSpec,
                                          target_pair_names: set[str], config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_mask = (epochs_hb.times >= config.baseline_window[0]) & (epochs_hb.times <= config.baseline_window[1])
    resp_mask = (epochs_hb.times >= config.response_window[0]) & (epochs_hb.times <= config.response_window[1])
    metric_rows, shape_rows, roi_rows = [], [], []

    for channel_name in channel_names:
        if channel_name not in epochs_hb.ch_names:
            continue
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        pair_name = channel_name.split(" ")[0]
        channel_data = epochs_hb.copy().pick([channel_name]).get_data()[:, 0, :]
        score = channel_data[:, resp_mask].mean(axis=1) - channel_data[:, base_mask].mean(axis=1)
        mean_curve = channel_data.mean(axis=0)
        metric_rows.append({
            "subject": subject,
            "file_label": file_label,
            "amplitude_value": amplitude_value,
            "pipeline_label": pipeline.label,
            "backend": pipeline.backend,
            "hrf_model": "block_average",
            "solver": "none",
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chrom,
            "target_status": "true_target" if pair_name in target_pair_names else "true_non_target",
            "score": float(score.mean()),
            "score_std": float(score.std()),
        })
        shape_rows.append({
            "subject": subject,
            "file_label": file_label,
            "amplitude_value": amplitude_value,
            "pipeline_label": pipeline.label,
            "backend": pipeline.backend,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chrom,
            "target_status": "true_target" if pair_name in target_pair_names else "true_non_target",
            "shape_source": "block_average_mean_epoch",
            **compute_shape_metrics(mean_curve, get_truth_curve({}, file_label, epochs_hb.times, chrom), epochs_hb.times, chrom),
        })

    metric_df = pd.DataFrame(metric_rows)
    if len(metric_df) > 0:
        for (chrom, target_status), group in metric_df.groupby(["chromophore", "target_status"]):
            names = group["channel_name"].tolist()
            mean_curve = epochs_hb.copy().pick(names).get_data().mean(axis=(0, 1))
            for t_s, y in zip(epochs_hb.times, mean_curve):
                roi_rows.append({
                    "subject": subject,
                    "file_label": file_label,
                    "amplitude_value": amplitude_value,
                    "pipeline_label": pipeline.label,
                    "backend": pipeline.backend,
                    "chromophore": chrom,
                    "target_status": target_status,
                    "curve_source": "block_average_mean_epoch",
                    "time_s": float(t_s),
                    "signal": float(y),
                })

    return metric_df, pd.DataFrame(shape_rows), pd.DataFrame(roi_rows)


# -----------------------------------------------------------------------------
# Python-pipeline execution
# -----------------------------------------------------------------------------


def execute_python_pipeline(subject: str, file_spec: FileSpec, pipeline: PipelineSpec, raw_cw: mne.io.BaseRaw,
                            raw_hb: mne.io.BaseRaw, target_pair_names: set[str],
                            truth_templates: dict[str, dict[str, np.ndarray]],
                            snirf_file_path: Path, config: BenchmarkConfig) -> dict[str, Any]:
    _, run_glm, _ = import_mne_nirs_modules()
    amplitude_value = file_spec.amplitude_value
    nuisance_rows, canonical_frames, fir_frames, shape_rows, roi_rows = [], [], [], [], []

    long_channel_names = get_available_long_channel_names(raw_hb, config)
    if not long_channel_names:
        return {
            "canonical_channel_metrics": pd.DataFrame(),
            "block_average_channel_metrics": pd.DataFrame(),
            "fir_channel_metrics": pd.DataFrame(),
            "shape_metrics": pd.DataFrame(),
            "roi_timecourses": pd.DataFrame(),
            "nuisance_detail": pd.DataFrame(),
            "matlab_input_specs_list": [],
            "matlab_shift_specs_list": [],
        }

    # Block-average branch: nuisance regression in time domain, then epoching.
    if pipeline.use_block_average:
        nuisance_factory = build_channel_specific_nuisance_factory(raw_hb, pipeline.nuisance_method, config)
        denoised_hb, nuisance_meta_df = apply_channelwise_nuisance_regression(raw_hb, nuisance_factory)
        epochs = make_epochs(denoised_hb, config)
        metric_df, shape_df, roi_df = compute_block_average_channel_metrics(
            epochs, long_channel_names, subject, file_spec.label, amplitude_value, pipeline, target_pair_names, config
        )
        if len(shape_df) > 0:
            # Replace placeholder truth with real template now that we have the full template map.
            corrected_shape_rows = []
            for _, row in shape_df.iterrows():
                chrom = row["chromophore"]
                pair_name = row["pair_name"]
                channel_name = row["channel_name"]
                mean_curve = epochs.copy().pick([channel_name]).get_data()[:, 0, :].mean(axis=0)
                corrected_shape_rows.append({
                    **row.drop(labels=["curve_corr","curve_rmse","curve_nrmse","peak_latency_error_s",
                                      "peak_amplitude_bias","peak_amplitude_ratio","auc_bias"]).to_dict(),
                    **compute_shape_metrics(
                        mean_curve,
                        get_truth_curve(truth_templates, file_spec.label, epochs.times, chrom),
                        epochs.times,
                        chrom,
                    ),
                })
            shape_df = pd.DataFrame(corrected_shape_rows)
        if len(nuisance_meta_df) > 0:
            nuisance_meta_df = nuisance_meta_df.copy()
            nuisance_meta_df["subject"] = subject
            nuisance_meta_df["file_label"] = file_spec.label
            nuisance_meta_df["pipeline_label"] = pipeline.label
        return {
            "canonical_channel_metrics": pd.DataFrame(),
            "block_average_channel_metrics": metric_df,
            "fir_channel_metrics": pd.DataFrame(),
            "shape_metrics": shape_df,
            "roi_timecourses": roi_df,
            "nuisance_detail": nuisance_meta_df,
            "matlab_input_specs_list": [],
            "matlab_shift_specs_list": [],
        }

    # MATLAB sidecar branch: prepare design matrices in Python, run AR-IRLS in MATLAB.
    if pipeline.backend == "matlab_arirls":
        nuisance_factory = build_channel_specific_nuisance_factory(raw_hb, pipeline.nuisance_method, config)
        observed_spec = build_matlab_input_spec(
            subject=subject,
            file_label=file_spec.label,
            amplitude_value=amplitude_value,
            pipeline=pipeline,
            raw_hb=raw_hb,
            nuisance_factory=nuisance_factory,
            target_pair_names=target_pair_names,
            shift_index=0,
            shift_s=0.0,
            config=config,
        )
        shift_specs = []
        if file_spec.is_null and pipeline.include_in_empirical_null:
            total_duration_s = float(raw_hb.times[-1])
            for shift_index, shift_s in enumerate(get_shift_values(total_duration_s, config.empirical_null_shift_count, config.empirical_null_min_shift_s), start=1):
                shifted = raw_hb.copy()
                shifted.set_annotations(make_shifted_annotations(raw_hb.annotations, total_duration_s, shift_s))
                shifted_factory = build_channel_specific_nuisance_factory(shifted, pipeline.nuisance_method, config)
                shift_spec = build_matlab_input_spec(
                    subject=subject,
                    file_label=file_spec.label,
                    amplitude_value=amplitude_value,
                    pipeline=pipeline,
                    raw_hb=shifted,
                    nuisance_factory=shifted_factory,
                    target_pair_names=target_pair_names,
                    shift_index=shift_index,
                    shift_s=shift_s,
                    config=config,
                )
                if shift_spec is not None:
                    shift_specs.append(shift_spec)
        nuisance_detail_df = pd.DataFrame()
        if observed_spec is not None and observed_spec.get("nuisance_detail_rows"):
            nuisance_detail_df = pd.DataFrame(observed_spec["nuisance_detail_rows"])
        return {
            "canonical_channel_metrics": pd.DataFrame(),
            "block_average_channel_metrics": pd.DataFrame(),
            "fir_channel_metrics": pd.DataFrame(),
            "shape_metrics": pd.DataFrame(),
            "roi_timecourses": pd.DataFrame(),
            "nuisance_detail": nuisance_detail_df,
            "matlab_input_specs_list": [] if observed_spec is None else [observed_spec],
            "matlab_shift_specs_list": shift_specs,
        }

    # Python GLM branches.
    if pipeline.nuisance_method in {"none", "pooled_pca2", "ss_aux_pca"}:
        global_bundle = build_global_nuisance_bundle(raw_hb, snirf_file_path, raw_cw, pipeline.nuisance_method, config)
        if pipeline.hrf_model == "fir":
            raw_fir = resample_for_fir(raw_hb, config)
            fir_long_names = get_available_long_channel_names(raw_fir, config)
            for chrom in ["hbo", "hbr"]:
                bundle = global_bundle.by_chromophore[chrom]
                channel_subset = [name for name in fir_long_names if name.endswith(chrom)]
                for channel_name in channel_subset:
                    nuisance_rows.append({
                        "subject": subject, "file_label": file_spec.label, "pipeline_label": pipeline.label,
                        "channel_name": channel_name, "chromophore": chrom, **bundle.metadata,
                    })
                if not channel_subset:
                    continue
                nuisance_df = None if bundle.design_df is None else pd.DataFrame(
                    np.vstack([np.interp(raw_fir.times, raw_hb.times, bundle.design_df[col].to_numpy()) for col in bundle.design_df.columns]).T,
                    columns=bundle.design_df.columns,
                )
                design = build_design_matrix(raw_fir, "fir", nuisance_df, config)
                glm = run_glm(raw_fir.copy().pick(channel_subset), design, noise_model=pipeline.solver)
                fir_df = parse_fir_dataframe(
                    glm.to_dataframe(), design, channel_subset, subject, file_spec.label, amplitude_value,
                    pipeline, target_pair_names, {name: bundle.metadata for name in channel_subset}
                )
                if len(fir_df) > 0:
                    fir_frames.append(fir_df)
        else:
            epoch_times = np.arange(config.epoch_tmin, config.epoch_tmax + 1e-9, np.median(np.diff(raw_hb.times)))
            for chrom in ["hbo", "hbr"]:
                bundle = global_bundle.by_chromophore[chrom]
                channel_subset = [name for name in long_channel_names if name.endswith(chrom)]
                for channel_name in channel_subset:
                    nuisance_rows.append({
                        "subject": subject, "file_label": file_spec.label, "pipeline_label": pipeline.label,
                        "channel_name": channel_name, "chromophore": chrom, **bundle.metadata,
                    })
                if not channel_subset:
                    continue
                design = build_design_matrix(raw_hb, pipeline.hrf_model, bundle.design_df, config)
                glm = run_glm(raw_hb.copy().pick(channel_subset), design, noise_model=pipeline.solver)
                channel_df = parse_mne_glm_to_channel_rows(
                    glm.to_dataframe(), design, channel_subset, subject, file_spec.label, amplitude_value,
                    pipeline, target_pair_names, {name: bundle.metadata for name in channel_subset}
                )
                if len(channel_df) == 0:
                    continue
                canonical_frames.append(channel_df)
                basis = build_basis_curve(epoch_times, pipeline.hrf_model)
                truth_curve = get_truth_curve(truth_templates, file_spec.label, epoch_times, chrom)
                for _, row in channel_df.iterrows():
                    shape_rows.append({
                        "subject": subject,
                        "file_label": file_spec.label,
                        "amplitude_value": amplitude_value,
                        "pipeline_label": pipeline.label,
                        "backend": pipeline.backend,
                        "channel_name": row["channel_name"],
                        "pair_name": row["pair_name"],
                        "chromophore": chrom,
                        "target_status": row["target_status"],
                        "shape_source": f"beta_scaled_{pipeline.hrf_model}",
                        **compute_shape_metrics(float(row["beta"]) * basis, truth_curve, epoch_times, chrom),
                    })
                for target_status, group in channel_df.groupby("target_status"):
                    roi_curve = float(group["beta"].mean()) * basis
                    for t_s, y in zip(epoch_times, roi_curve):
                        roi_rows.append({
                            "subject": subject,
                            "file_label": file_spec.label,
                            "amplitude_value": amplitude_value,
                            "pipeline_label": pipeline.label,
                            "backend": pipeline.backend,
                            "chromophore": chrom,
                            "target_status": target_status,
                            "curve_source": f"roi_mean_beta_scaled_{pipeline.hrf_model}",
                            "time_s": float(t_s),
                            "signal": float(y),
                        })
        fir_df = pd.concat(fir_frames, ignore_index=True) if fir_frames else pd.DataFrame()
        if len(fir_df) > 0:
            fir_shapes, fir_roi_rows = summarize_fir_shapes(fir_df, file_spec.label, pipeline, truth_templates, config)
            shape_rows.extend(fir_shapes)
            roi_rows.extend(fir_roi_rows)
        return {
            "canonical_channel_metrics": pd.concat(canonical_frames, ignore_index=True) if canonical_frames else pd.DataFrame(),
            "block_average_channel_metrics": pd.DataFrame(),
            "fir_channel_metrics": fir_df,
            "shape_metrics": pd.DataFrame(shape_rows),
            "roi_timecourses": pd.DataFrame(roi_rows),
            "nuisance_detail": pd.DataFrame(nuisance_rows),
            "matlab_input_specs_list": [],
            "matlab_shift_specs_list": [],
        }

    nuisance_factory = build_channel_specific_nuisance_factory(raw_hb, pipeline.nuisance_method, config)
    if pipeline.hrf_model == "fir":
        raw_fir = resample_for_fir(raw_hb, config)
        fir_long_names = get_available_long_channel_names(raw_fir, config)
        fir_factory = build_channel_specific_nuisance_factory(raw_fir, pipeline.nuisance_method, config)
        for channel_name in fir_long_names:
            chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
            bundle = fir_factory.channel_to_bundle[channel_name]
            nuisance_rows.append({
                "subject": subject, "file_label": file_spec.label, "pipeline_label": pipeline.label,
                "channel_name": channel_name, "chromophore": chrom, **bundle.metadata,
            })
            design = build_design_matrix(raw_fir, "fir", bundle.design_df, config)
            fir_df = parse_fir_dataframe(
                run_glm(raw_fir.copy().pick([channel_name]), design, noise_model=pipeline.solver).to_dataframe(),
                design, [channel_name], subject, file_spec.label, amplitude_value,
                pipeline, target_pair_names, {channel_name: bundle.metadata}
            )
            if len(fir_df) > 0:
                fir_frames.append(fir_df)
        fir_df = pd.concat(fir_frames, ignore_index=True) if fir_frames else pd.DataFrame()
        if len(fir_df) > 0:
            fir_shapes, fir_roi_rows = summarize_fir_shapes(fir_df, file_spec.label, pipeline, truth_templates, config)
            shape_rows.extend(fir_shapes)
            roi_rows.extend(fir_roi_rows)
        return {
            "canonical_channel_metrics": pd.DataFrame(),
            "block_average_channel_metrics": pd.DataFrame(),
            "fir_channel_metrics": fir_df,
            "shape_metrics": pd.DataFrame(shape_rows),
            "roi_timecourses": pd.DataFrame(roi_rows),
            "nuisance_detail": pd.DataFrame(nuisance_rows),
            "matlab_input_specs_list": [],
            "matlab_shift_specs_list": [],
        }

    epoch_times = np.arange(config.epoch_tmin, config.epoch_tmax + 1e-9, np.median(np.diff(raw_hb.times)))
    for channel_name in long_channel_names:
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        bundle = nuisance_factory.channel_to_bundle[channel_name]
        nuisance_rows.append({
            "subject": subject, "file_label": file_spec.label, "pipeline_label": pipeline.label,
            "channel_name": channel_name, "chromophore": chrom, **bundle.metadata,
        })
        design = build_design_matrix(raw_hb, pipeline.hrf_model, bundle.design_df, config)
        channel_df = parse_mne_glm_to_channel_rows(
            run_glm(raw_hb.copy().pick([channel_name]), design, noise_model=pipeline.solver).to_dataframe(),
            design, [channel_name], subject, file_spec.label, amplitude_value,
            pipeline, target_pair_names, {channel_name: bundle.metadata}
        )
        if len(channel_df) > 0:
            canonical_frames.append(channel_df)

    canonical_df = pd.concat(canonical_frames, ignore_index=True) if canonical_frames else pd.DataFrame()
    if len(canonical_df) > 0:
        basis = build_basis_curve(epoch_times, pipeline.hrf_model)
        for _, row in canonical_df.iterrows():
            chrom = row["chromophore"]
            truth_curve = get_truth_curve(truth_templates, file_spec.label, epoch_times, chrom)
            shape_rows.append({
                "subject": subject,
                "file_label": file_spec.label,
                "amplitude_value": amplitude_value,
                "pipeline_label": pipeline.label,
                "backend": pipeline.backend,
                "channel_name": row["channel_name"],
                "pair_name": row["pair_name"],
                "chromophore": chrom,
                "target_status": row["target_status"],
                "shape_source": f"beta_scaled_{pipeline.hrf_model}",
                **compute_shape_metrics(float(row["beta"]) * basis, truth_curve, epoch_times, chrom),
            })
        for (chrom, target_status), group in canonical_df.groupby(["chromophore", "target_status"]):
            roi_curve = float(group["beta"].mean()) * basis
            for t_s, y in zip(epoch_times, roi_curve):
                roi_rows.append({
                    "subject": subject,
                    "file_label": file_spec.label,
                    "amplitude_value": amplitude_value,
                    "pipeline_label": pipeline.label,
                    "backend": pipeline.backend,
                    "chromophore": chrom,
                    "target_status": target_status,
                    "curve_source": f"roi_mean_beta_scaled_{pipeline.hrf_model}",
                    "time_s": float(t_s),
                    "signal": float(y),
                })

    return {
        "canonical_channel_metrics": canonical_df,
        "block_average_channel_metrics": pd.DataFrame(),
        "fir_channel_metrics": pd.DataFrame(),
        "shape_metrics": pd.DataFrame(shape_rows),
        "roi_timecourses": pd.DataFrame(roi_rows),
        "nuisance_detail": pd.DataFrame(nuisance_rows),
        "matlab_input_specs_list": [],
        "matlab_shift_specs_list": [],
    }


def parse_fir_dataframe(glm_df: pd.DataFrame, design_matrix: pd.DataFrame, channel_names: list[str],
                        subject: str, file_label: str, amplitude_value: int,
                        pipeline: PipelineSpec, target_pair_names: set[str],
                        nuisance_metadata_by_channel: dict[str, dict[str, Any]]) -> pd.DataFrame:
    glm_df = standardize_glm_dataframe(glm_df)
    ch_col = find_first_matching_column(glm_df.columns, ["ch_name", "channel", "name"])
    cond_col = find_first_matching_column(glm_df.columns, ["condition", "cond", "regressor", "variable", "name"])
    beta_col = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])
    if beta_col is None or cond_col is None:
        raise BenchmarkError("Could not parse FIR GLM dataframe columns.")
    task_names = get_task_regressor_names(design_matrix)
    rows = []
    for _, row in glm_df.iterrows():
        condition = str(row[cond_col])
        if condition not in task_names and not any(condition.startswith(name) for name in task_names):
            continue
        try:
            delay_scan = float(condition.split("_")[-1])
        except Exception:
            continue
        channel_name = str(row[ch_col]) if ch_col is not None else None
        if channel_name is None or channel_name not in channel_names:
            continue
        pair_name = channel_name.split(" ")[0]
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        rows.append({
            "subject": subject,
            "file_label": file_label,
            "amplitude_value": amplitude_value,
            "pipeline_label": pipeline.label,
            "backend": pipeline.backend,
            "hrf_model": pipeline.hrf_model,
            "solver": pipeline.solver,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chrom,
            "target_status": "true_target" if pair_name in target_pair_names else "true_non_target",
            "fir_regressor": condition,
            "delay_scan": delay_scan,
            "delay_s": delay_scan / 1.0,  # fir pipelines are resampled to 1 Hz
            "beta": float(row[beta_col]),
            **nuisance_metadata_by_channel.get(channel_name, {}),
        })
    return pd.DataFrame(rows)


def summarize_fir_shapes(fir_df: pd.DataFrame, file_label: str, pipeline: PipelineSpec,
                         truth_templates: dict[str, dict[str, np.ndarray]],
                         config: BenchmarkConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    epoch_times = np.arange(config.epoch_tmin, config.epoch_tmax + 1e-9, 1.0 / config.fir_resample_sfreq_hz)
    shape_rows, roi_rows = [], []
    for channel_name, group in fir_df.groupby("channel_name"):
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        pair_name = channel_name.split(" ")[0]
        group = group.sort_values("delay_s")
        recovered = build_basis_curve(epoch_times, "fir", fir_delays=group["delay_s"].tolist(), fir_betas=group["beta"].to_numpy())
        truth_curve = get_truth_curve(truth_templates, file_label, epoch_times, chrom)
        shape_rows.append({
            "subject": group["subject"].iloc[0],
            "file_label": file_label,
            "amplitude_value": int(group["amplitude_value"].iloc[0]),
            "pipeline_label": pipeline.label,
            "backend": pipeline.backend,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chrom,
            "target_status": group["target_status"].iloc[0],
            "shape_source": "fir_beta_curve",
            **compute_shape_metrics(recovered, truth_curve, epoch_times, chrom),
        })
    for (chrom, target_status), group in fir_df.groupby(["chromophore", "target_status"]):
        curve = group.groupby("delay_s", as_index=False)["beta"].mean().sort_values("delay_s")
        roi_curve = build_basis_curve(epoch_times, "fir", fir_delays=curve["delay_s"].tolist(), fir_betas=curve["beta"].to_numpy())
        for t_s, y in zip(epoch_times, roi_curve):
            roi_rows.append({
                "subject": group["subject"].iloc[0],
                "file_label": file_label,
                "amplitude_value": int(group["amplitude_value"].iloc[0]),
                "pipeline_label": pipeline.label,
                "backend": pipeline.backend,
                "chromophore": chrom,
                "target_status": target_status,
                "curve_source": "fir_roi_mean_curve",
                "time_s": float(t_s),
                "signal": float(y),
            })
    return shape_rows, roi_rows


# -----------------------------------------------------------------------------
# MATLAB / AnalyzIR helpers
# -----------------------------------------------------------------------------


def build_matlab_input_spec(subject: str, file_label: str, amplitude_value: int, pipeline: PipelineSpec,
                            raw_hb: mne.io.BaseRaw, nuisance_factory: ChannelSpecificNuisanceFactory,
                            target_pair_names: set[str], shift_index: int, shift_s: float,
                            config: BenchmarkConfig) -> Optional[dict[str, Any]]:
    long_channel_names = get_available_long_channel_names(raw_hb, config)
    if not long_channel_names:
        return None
    channel_names, pair_names, chroms, target_statuses = [], [], [], []
    y_list, x_list, n_reg_list, task_reg_idx_list = [], [], [], []
    nuisance_detail_rows = []

    for channel_name in long_channel_names:
        chrom = "hbo" if channel_name.endswith("hbo") else "hbr"
        bundle = nuisance_factory.channel_to_bundle[channel_name]
        nuisance_detail_rows.append({
            "subject": subject,
            "file_label": file_label,
            "pipeline_label": pipeline.label,
            "channel_name": channel_name,
            "chromophore": chrom,
            **bundle.metadata,
        })
        design = build_design_matrix(raw_hb, pipeline.hrf_model, bundle.design_df, config)
        task_names = get_task_regressor_names(design)
        if not task_names:
            continue
        task_reg_index = list(design.columns).index(task_names[0]) + 1  # MATLAB 1-based indexing
        y = raw_hb.copy().pick([channel_name]).get_data()[0]
        channel_names.append(channel_name)
        pair_names.append(channel_name.split(" ")[0])
        chroms.append(chrom)
        target_statuses.append("true_target" if channel_name.split(" ")[0] in target_pair_names else "true_non_target")
        y_list.append(y)
        x_list.append(design.to_numpy())
        n_reg_list.append(design.shape[1])
        task_reg_idx_list.append(task_reg_index)

    if not channel_names:
        return None

    max_reg = max(n_reg_list)
    n_time = len(raw_hb.times)
    n_chan = len(channel_names)
    X_tensor = np.zeros((n_time, max_reg, n_chan), dtype=float)
    Y = np.zeros((n_time, n_chan), dtype=float)
    for idx, (X, y) in enumerate(zip(x_list, y_list)):
        X_tensor[:, :X.shape[1], idx] = X
        Y[:, idx] = y

    return {
        "subject": subject,
        "file_label": file_label,
        "amplitude_value": amplitude_value,
        "pipeline_label": pipeline.label,
        "backend": pipeline.backend,
        "hrf_model": pipeline.hrf_model,
        "solver": pipeline.solver,
        "times_s": raw_hb.times.astype(float),
        "Y": Y,
        "X": X_tensor,
        "n_reg": np.asarray(n_reg_list, dtype=np.int32),
        "task_reg_index": np.asarray(task_reg_idx_list, dtype=np.int32),
        "channel_names": np.asarray(channel_names, dtype=object),
        "pair_names": np.asarray(pair_names, dtype=object),
        "chromophores": np.asarray(chroms, dtype=object),
        "target_status": np.asarray(target_statuses, dtype=object),
        "shift_index": int(shift_index),
        "shift_s": float(shift_s),
        "nuisance_detail_rows": nuisance_detail_rows,
    }


def write_matlab_bundle(observed_specs: list[dict[str, Any]], shift_specs: list[dict[str, Any]], job_dir: Path) -> Optional[Path]:
    specs = [spec for spec in observed_specs + shift_specs if spec is not None]
    if not specs:
        return None
    in_dir = ensure_dir(job_dir / "matlab_inputs")
    bundle = {"input_mat_files": [], "output_csv_files": []}
    for idx, spec in enumerate(specs, start=1):
        pipeline_label = spec["pipeline_label"]
        shift_index = int(spec.get("shift_index", 0))
        suffix = f"{pipeline_label}__shift{shift_index:03d}" if shift_index > 0 else f"{pipeline_label}__observed"
        in_path = in_dir / f"{idx:04d}__{suffix}__input.mat"
        out_path = in_dir / f"{idx:04d}__{suffix}__output.csv"
        scipy.io.savemat(in_path, {
            "times_s": np.asarray(spec["times_s"], dtype=float),
            "Y": np.asarray(spec["Y"], dtype=float),
            "X": np.asarray(spec["X"], dtype=float),
            "n_reg": np.asarray(spec["n_reg"], dtype=np.int32),
            "task_reg_index": np.asarray(spec["task_reg_index"], dtype=np.int32),
            "channel_names": np.asarray(spec["channel_names"], dtype=object),
            "pair_names": np.asarray(spec["pair_names"], dtype=object),
            "chromophores": np.asarray(spec["chromophores"], dtype=object),
            "target_status": np.asarray(spec["target_status"], dtype=object),
            "subject": np.asarray([spec["subject"]], dtype=object),
            "file_label": np.asarray([spec["file_label"]], dtype=object),
            "pipeline_label": np.asarray([spec["pipeline_label"]], dtype=object),
            "backend": np.asarray([spec["backend"]], dtype=object),
            "hrf_model": np.asarray([spec["hrf_model"]], dtype=object),
            "solver": np.asarray([spec["solver"]], dtype=object),
            "amplitude_value": np.asarray([[spec["amplitude_value"]]], dtype=np.int32),
            "shift_index": np.asarray([[spec["shift_index"]]], dtype=np.int32),
            "shift_s": np.asarray([[spec["shift_s"]]], dtype=float),
        }, do_compression=True)
        bundle["input_mat_files"].append(str(in_path))
        bundle["output_csv_files"].append(str(out_path))
    bundle_path = in_dir / "matlab_bundle.json"
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle_path


def run_matlab_sidecar(config: BenchmarkConfig, bundle_json_path: Path, job_dir: Path) -> pd.DataFrame:
    if not config.use_matlab:
        return pd.DataFrame()
    helper_m = Path(__file__).with_name("analyzir_arirls_batch.m")
    if not helper_m.exists():
        raise BenchmarkError(f"Missing MATLAB helper: {helper_m}")
    env = os.environ.copy()
    env["FNIRS_BUNDLE_JSON"] = str(bundle_json_path)
    if config.analyzir_path:
        env["FNIRS_ANALYZIR_PATH"] = str(config.analyzir_path)
    helper_dir = helper_m.parent.as_posix()
    helper_name = helper_m.stem
    command = [
        config.matlab_cmd,
        "-batch",
        f"try, addpath('{helper_dir}'); {helper_name}; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=config.matlab_timeout_s)
    (job_dir / "matlab_batch.log").write_text(
        f"STDOUT\n{'='*80}\n{result.stdout}\n\nSTDERR\n{'='*80}\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise BenchmarkError(f"MATLAB sidecar failed; see {job_dir / 'matlab_batch.log'}")
    bundle = json.loads(bundle_json_path.read_text(encoding="utf-8"))
    tables = []
    for csv_path_str in bundle["output_csv_files"]:
        csv_path = Path(csv_path_str)
        if csv_path.exists():
            tables.append(pd.read_csv(csv_path))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# -----------------------------------------------------------------------------
# Null helpers
# -----------------------------------------------------------------------------


def make_shifted_annotations(source_annotations: mne.Annotations, total_duration_s: float, shift_s: float) -> mne.Annotations:
    shifted_onsets, shifted_durations, shifted_descriptions = [], [], []
    for onset_s, duration_s, description in zip(source_annotations.onset, source_annotations.duration, source_annotations.description):
        shifted_onset = float(onset_s) + float(shift_s)
        while shifted_onset >= total_duration_s:
            shifted_onset -= total_duration_s
        shifted_onsets.append(shifted_onset)
        shifted_durations.append(float(duration_s))
        shifted_descriptions.append(str(description))
    order = np.argsort(shifted_onsets)
    return mne.Annotations(
        onset=np.asarray(shifted_onsets)[order],
        duration=np.asarray(shifted_durations)[order],
        description=np.asarray(shifted_descriptions)[order].tolist(),
    )


def get_shift_values(total_duration_s: float, shift_count: int, min_shift_s: float) -> list[float]:
    if shift_count <= 0:
        return []
    max_shift_s = max(min_shift_s + 1.0, total_duration_s - min_shift_s)
    values = np.linspace(min_shift_s, max_shift_s, shift_count + 2)[1:-1]
    return [float(v) for v in values]


def summarize_target_minus_non_target(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    grouped = (
        df.groupby(["chromophore", "target_status"], as_index=False)
        .agg(mean_score=(value_column, "mean"))
    )
    wide = (
        grouped.pivot_table(index=["chromophore"], columns="target_status", values="mean_score")
        .reset_index()
        .rename(columns={"true_target": "mean_target_score", "true_non_target": "mean_non_target_score"})
    )
    if "mean_target_score" not in wide.columns:
        wide["mean_target_score"] = np.nan
    if "mean_non_target_score" not in wide.columns:
        wide["mean_non_target_score"] = np.nan
    wide["target_minus_non_target_score"] = wide["mean_target_score"] - wide["mean_non_target_score"]
    return wide


# -----------------------------------------------------------------------------
# Per-job processing
# -----------------------------------------------------------------------------


def process_subject_file_job(subject: str, file_spec_dict: dict[str, Any], config_dict: dict[str, Any]) -> dict[str, Any]:
    config = BenchmarkConfig(**{k: v for k, v in config_dict.items() if k not in {"file_specs", "pipeline_specs"}})
    config.file_specs = [FileSpec(**item) for item in config_dict["file_specs"]]
    config.pipeline_specs = [PipelineSpec(**item) for item in config_dict["pipeline_specs"]]
    file_spec = FileSpec(**file_spec_dict)

    jobs_dir = ensure_dir(config.jobs_path())
    job_dir = ensure_dir(jobs_dir / subject / file_spec.label)
    success_marker = job_dir / "SUCCESS.json"

    if success_marker.exists() and not config.overwrite:
        return {"subject": subject, "file_label": file_spec.label, "status": "skipped_existing", "job_dir": str(job_dir)}

    if config.overwrite and job_dir.exists():
        for child in job_dir.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child, ignore_errors=True)
        ensure_dir(job_dir)

    (job_dir / "STARTED.json").write_text(json.dumps({
        "subject": subject, "file_label": file_spec.label, "started_utc": now_utc_iso()
    }, indent=2), encoding="utf-8")

    dataset_dir = config.dataset_path()
    subject_dir = dataset_dir / subject
    snirf_file_path = subject_dir / file_spec.filename
    annotation_source_path = subject_dir / file_spec.annotation_source_filename if file_spec.annotation_source_filename else None
    truth_templates = load_truth_templates(config)
    errors: list[dict[str, Any]] = []

    try:
        if not snirf_file_path.exists():
            raise BenchmarkError(f"Missing input file: {snirf_file_path}")

        raw_cw = mne.io.read_raw_snirf(snirf_file_path, preload=True, verbose=False)
        if file_spec.is_null:
            if annotation_source_path is None or not annotation_source_path.exists():
                raise BenchmarkError("Null file needs a valid annotation source file.")
            raw_cw = copy_valid_annotations(raw_cw, annotation_source_path)
        raw_cw = sanitize_annotations_to_single_task(raw_cw)

        reference_path = subject_dir / "resting_hrf_20.snirf"
        if not reference_path.exists():
            raise BenchmarkError(f"Reference truth file missing: {reference_path}")
        reference_raw = mne.io.read_raw_snirf(reference_path, preload=True, verbose=False)
        data_type_labels = read_measurement_data_type_labels(reference_path)
        if data_type_labels is None:
            raise BenchmarkError("Could not read truth labels from reference file.")
        picks_cw = get_cw_channel_indices(reference_raw)
        cw_names = np.asarray(reference_raw.ch_names)[picks_cw]
        if len(data_type_labels) == len(cw_names):
            aligned_names = cw_names
        elif len(data_type_labels) == len(reference_raw.ch_names):
            aligned_names = np.asarray(reference_raw.ch_names)
        else:
            raise BenchmarkError("Truth-label alignment failed.")
        target_pair_names = sorted(set(name.split(" ")[0] for name in aligned_names[data_type_labels == 1].tolist()))
        target_pair_set = set(target_pair_names)

        cw_channel_table = build_cw_channel_table(reference_raw, subject, file_spec.label, config)
        long_pair_names = sorted(cw_channel_table.loc[cw_channel_table["group"] == "LS", "pair_name"].astype(str).unique())
        non_target_pair_names = [pair for pair in long_pair_names if pair not in target_pair_set]

        truth_summary = pd.DataFrame([{
            "subject": subject,
            "file_label": file_spec.label,
            "amplitude_value": file_spec.amplitude_value,
            "n_true_target_pairs": len(target_pair_names),
            "n_true_non_target_pairs": len(non_target_pair_names),
            "target_pair_names": "|".join(target_pair_names),
        }])
        maybe_write_table(truth_summary, job_dir / "truth_summary", config.write_csv, config.write_parquet)

        channel_quality, pair_quality = build_quality_tables(raw_cw, subject, file_spec.label, config)
        maybe_write_table(channel_quality, job_dir / "channel_quality", config.write_csv, config.write_parquet)
        maybe_write_table(pair_quality, job_dir / "pair_quality", config.write_csv, config.write_parquet)

        preproc_cache: dict[tuple[str, bool], tuple[mne.io.BaseRaw, list[str]]] = {}

        def get_preprocessed(pipeline: PipelineSpec) -> tuple[mne.io.BaseRaw, list[str]]:
            key = (pipeline.pruning_style, pipeline.do_tddr)
            if key not in preproc_cache:
                preproc_cache[key] = preprocess_raw_to_hb(raw_cw, pair_quality, pipeline, config)
            return preproc_cache[key]

        all_canonical, all_block, all_fir = [], [], []
        all_shape, all_roi, all_nuisance, all_empirical = [], [], [], []
        availability_rows = []

        matlab_observed_specs, matlab_shift_specs = [], []

        for pipeline in config.pipeline_specs:
            if pipeline.backend == "matlab_arirls" and not config.use_matlab:
                continue

            raw_hb, bad_pairs = get_preprocessed(pipeline)
            try:
                result = execute_python_pipeline(
                    subject=subject,
                    file_spec=file_spec,
                    pipeline=pipeline,
                    raw_cw=raw_cw,
                    raw_hb=raw_hb,
                    target_pair_names=target_pair_set,
                    truth_templates=truth_templates,
                    snirf_file_path=snirf_file_path,
                    config=config,
                )
            except Exception as exc:
                errors.append(make_error_row(subject, file_spec.label, pipeline.label, "pipeline_execute", f"{exc}\n\n{traceback.format_exc()}"))
                continue

            if len(result["canonical_channel_metrics"]) > 0:
                all_canonical.append(result["canonical_channel_metrics"])
            if len(result["block_average_channel_metrics"]) > 0:
                all_block.append(result["block_average_channel_metrics"])
            if len(result["fir_channel_metrics"]) > 0:
                all_fir.append(result["fir_channel_metrics"])
            if len(result["shape_metrics"]) > 0:
                all_shape.append(result["shape_metrics"])
            if len(result["roi_timecourses"]) > 0:
                all_roi.append(result["roi_timecourses"])
            if len(result["nuisance_detail"]) > 0:
                all_nuisance.append(result["nuisance_detail"])
            if result["matlab_input_specs_list"]:
                matlab_observed_specs.extend(result["matlab_input_specs_list"])
            if result["matlab_shift_specs_list"]:
                matlab_shift_specs.extend(result["matlab_shift_specs_list"])

            long_channel_names = get_available_long_channel_names(raw_hb, config)
            available_target_pairs = sorted(set(name.split(" ")[0] for name in long_channel_names if name.split(" ")[0] in target_pair_set))
            available_non_target_pairs = sorted(set(name.split(" ")[0] for name in long_channel_names if name.split(" ")[0] in non_target_pair_names))
            availability_rows.append({
                "subject": subject,
                "file_label": file_spec.label,
                "amplitude_value": file_spec.amplitude_value,
                "pipeline_label": pipeline.label,
                "backend": pipeline.backend,
                "n_total_true_target_pairs": len(target_pair_names),
                "n_available_true_target_pairs": len(available_target_pairs),
                "n_total_true_non_target_pairs": len(non_target_pair_names),
                "n_available_true_non_target_pairs": len(available_non_target_pairs),
                "n_available_long_channels": len(long_channel_names),
                "target_pair_retention_fraction": len(available_target_pairs) / len(target_pair_names) if target_pair_names else np.nan,
                "non_target_pair_retention_fraction": len(available_non_target_pairs) / len(non_target_pair_names) if non_target_pair_names else np.nan,
                "n_bad_pairs": len(bad_pairs),
            })

            # Empirical null for python pipelines.
            if file_spec.is_null and pipeline.include_in_empirical_null and pipeline.backend == "python":
                total_duration_s = float(raw_hb.times[-1])
                shift_values = get_shift_values(total_duration_s, config.empirical_null_shift_count, config.empirical_null_min_shift_s)
                for shift_index, shift_s in enumerate(shift_values, start=1):
                    shifted_raw_hb = raw_hb.copy()
                    shifted_raw_hb.set_annotations(make_shifted_annotations(raw_hb.annotations, total_duration_s, shift_s))
                    try:
                        shifted_result = execute_python_pipeline(
                            subject=subject,
                            file_spec=file_spec,
                            pipeline=pipeline,
                            raw_cw=raw_cw,
                            raw_hb=shifted_raw_hb,
                            target_pair_names=target_pair_set,
                            truth_templates=truth_templates,
                            snirf_file_path=snirf_file_path,
                            config=config,
                        )
                    except Exception as exc:
                        errors.append(make_error_row(subject, file_spec.label, pipeline.label, "empirical_null_shift_python", str(exc)))
                        continue
                    shifted_ch_df = shifted_result["canonical_channel_metrics"]
                    if len(shifted_ch_df) == 0:
                        continue
                    sep_df = summarize_target_minus_non_target(shifted_ch_df, "beta")
                    for _, row in sep_df.iterrows():
                        all_empirical.append(pd.DataFrame([{
                            "subject": subject,
                            "file_label": file_spec.label,
                            "pipeline_label": pipeline.label,
                            "backend": pipeline.backend,
                            "chromophore": row["chromophore"],
                            "shift_index": shift_index,
                            "shift_s": shift_s,
                            "mean_target_score": row["mean_target_score"],
                            "mean_non_target_score": row["mean_non_target_score"],
                            "target_minus_non_target_score": row["target_minus_non_target_score"],
                        }]))

        if config.use_matlab and (matlab_observed_specs or matlab_shift_specs):
            bundle_path = write_matlab_bundle(matlab_observed_specs, matlab_shift_specs, job_dir)
            if bundle_path is not None:
                try:
                    matlab_df = run_matlab_sidecar(config, bundle_path, job_dir)
                except Exception as exc:
                    errors.append(make_error_row(subject, file_spec.label, "MATLAB_ARIRLS", "matlab_sidecar", f"{exc}\n\n{traceback.format_exc()}"))
                    matlab_df = pd.DataFrame()

                if len(matlab_df) > 0:
                    observed_matlab = matlab_df.loc[matlab_df["shift_index"] == 0].copy() if "shift_index" in matlab_df.columns else matlab_df.copy()
                    if len(observed_matlab) > 0:
                        all_canonical.append(observed_matlab)
                        epoch_times = np.arange(config.epoch_tmin, config.epoch_tmax + 1e-9, np.median(np.diff(raw_hb.times)))
                        for _, row in observed_matlab.iterrows():
                            model = "gamma" if str(row.get("hrf_model", "")).lower() == "gamma" else "canonical"
                            chrom = row["chromophore"]
                            recovered = float(row["beta"]) * build_basis_curve(epoch_times, model)
                            truth_curve = get_truth_curve(truth_templates, file_spec.label, epoch_times, chrom)
                            all_shape.append(pd.DataFrame([{
                                "subject": row["subject"],
                                "file_label": row["file_label"],
                                "amplitude_value": row["amplitude_value"],
                                "pipeline_label": row["pipeline_label"],
                                "backend": row["backend"],
                                "channel_name": row["channel_name"],
                                "pair_name": row["pair_name"],
                                "chromophore": chrom,
                                "target_status": row["target_status"],
                                "shape_source": f"beta_scaled_{model}",
                                **compute_shape_metrics(recovered, truth_curve, epoch_times, chrom),
                            }]))
                        for (pipeline_label, chrom, target_status), group in observed_matlab.groupby(["pipeline_label", "chromophore", "target_status"]):
                            model = "gamma" if "Gamma" in pipeline_label or (("hrf_model" in group.columns) and str(group["hrf_model"].iloc[0]).lower() == "gamma") else "canonical"
                            roi_curve = float(group["beta"].mean()) * build_basis_curve(epoch_times, model)
                            all_roi.append(pd.DataFrame({
                                "subject": subject,
                                "file_label": file_spec.label,
                                "amplitude_value": file_spec.amplitude_value,
                                "pipeline_label": pipeline_label,
                                "backend": "matlab_arirls",
                                "chromophore": chrom,
                                "target_status": target_status,
                                "curve_source": f"roi_mean_beta_scaled_{model}",
                                "time_s": epoch_times,
                                "signal": roi_curve,
                            }))
                    shifted_matlab = matlab_df.loc[matlab_df["shift_index"] > 0].copy() if "shift_index" in matlab_df.columns else pd.DataFrame()
                    if len(shifted_matlab) > 0:
                        sep = (
                            shifted_matlab
                            .groupby(["subject", "file_label", "pipeline_label", "backend", "shift_index", "shift_s", "chromophore", "target_status"], as_index=False)
                            .agg(mean_score=("beta", "mean"))
                        )
                        sep_wide = (
                            sep.pivot_table(
                                index=["subject", "file_label", "pipeline_label", "backend", "shift_index", "shift_s", "chromophore"],
                                columns="target_status",
                                values="mean_score",
                            )
                            .reset_index()
                            .rename(columns={"true_target": "mean_target_score", "true_non_target": "mean_non_target_score"})
                        )
                        if "mean_target_score" not in sep_wide.columns:
                            sep_wide["mean_target_score"] = np.nan
                        if "mean_non_target_score" not in sep_wide.columns:
                            sep_wide["mean_non_target_score"] = np.nan
                        sep_wide["target_minus_non_target_score"] = sep_wide["mean_target_score"] - sep_wide["mean_non_target_score"]
                        all_empirical.append(sep_wide)

        maybe_write_table(pd.DataFrame(availability_rows), job_dir / "channel_availability", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_canonical, ignore_index=True) if all_canonical else pd.DataFrame(), job_dir / "canonical_channel_metrics", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_block, ignore_index=True) if all_block else pd.DataFrame(), job_dir / "block_average_channel_metrics", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_fir, ignore_index=True) if all_fir else pd.DataFrame(), job_dir / "fir_channel_metrics", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_shape, ignore_index=True) if all_shape else pd.DataFrame(), job_dir / "shape_fidelity", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_roi, ignore_index=True) if all_roi else pd.DataFrame(), job_dir / "roi_timecourses", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_nuisance, ignore_index=True) if all_nuisance else pd.DataFrame(), job_dir / "nuisance_detail", config.write_csv, config.write_parquet)
        maybe_write_table(pd.concat(all_empirical, ignore_index=True) if all_empirical else pd.DataFrame(), job_dir / "empirical_null_shift", config.write_csv, config.write_parquet)
        if errors:
            maybe_write_table(pd.DataFrame(errors), job_dir / "error_log", True, config.write_parquet)

        success_marker.write_text(json.dumps({
            "subject": subject,
            "file_label": file_spec.label,
            "finished_utc": now_utc_iso(),
            "job_dir": str(job_dir),
        }, indent=2), encoding="utf-8")
        return {"subject": subject, "file_label": file_spec.label, "status": "finished", "job_dir": str(job_dir)}

    except Exception as exc:
        errors.append(make_error_row(subject, file_spec.label, None, "job_crash", f"{exc}\n\n{traceback.format_exc()}"))
        maybe_write_table(pd.DataFrame(errors), job_dir / "error_log", True, config.write_parquet)
        return {"subject": subject, "file_label": file_spec.label, "status": "crashed", "job_dir": str(job_dir), "error": str(exc)}


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def aggregate_outputs(config: BenchmarkConfig, run_results: list[dict[str, Any]]) -> None:
    aggregate_dir = ensure_dir(config.aggregate_path())
    table_names = [
        "truth_summary", "channel_quality", "pair_quality", "channel_availability",
        "canonical_channel_metrics", "block_average_channel_metrics", "fir_channel_metrics",
        "shape_fidelity", "roi_timecourses", "nuisance_detail", "empirical_null_shift", "error_log",
    ]
    aggregated = {name: [] for name in table_names}
    for job_dir in sorted([p for p in config.jobs_path().glob("*/*") if p.is_dir()]):
        for name in table_names:
            df = read_any_table(job_dir / name)
            if len(df) > 0:
                aggregated[name].append(df)
    combined = {name: (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()) for name, frames in aggregated.items()}

    canonical_df = combined["canonical_channel_metrics"]
    if len(canonical_df) > 0 and "p_value" in canonical_df.columns:
        canonical_df = canonical_df.copy()
        canonical_df["q_value_bh"] = np.nan
        for _, group in canonical_df.groupby(["subject", "file_label", "pipeline_label", "chromophore"]):
            idx = group.index.to_list()
            canonical_df.loc[idx, "q_value_bh"] = benjamini_hochberg(group["p_value"].to_numpy())
        combined["canonical_channel_metrics"] = canonical_df

    # ROI scores
    roi_tables = []
    if len(canonical_df) > 0:
        c_roi = canonical_df.groupby(
            ["subject", "file_label", "amplitude_value", "pipeline_label", "backend", "chromophore", "target_status"],
            as_index=False,
        ).agg(
            roi_mean_score=("beta", "mean"),
            roi_std_score=("beta", "std"),
            n_channels=("channel_name", "count"),
        )
        c_roi["score_type"] = "canonical_beta"
        roi_tables.append(c_roi)
    block_df = combined["block_average_channel_metrics"]
    if len(block_df) > 0:
        b_roi = block_df.groupby(
            ["subject", "file_label", "amplitude_value", "pipeline_label", "backend", "chromophore", "target_status"],
            as_index=False,
        ).agg(
            roi_mean_score=("score", "mean"),
            roi_std_score=("score", "std"),
            n_channels=("channel_name", "count"),
        )
        b_roi["score_type"] = "block_average_score"
        roi_tables.append(b_roi)
    roi_scores = pd.concat(roi_tables, ignore_index=True) if roi_tables else pd.DataFrame()

    # Target-vs-nontarget summary
    sep_tables = []
    if len(canonical_df) > 0:
        c_sep = (
            canonical_df
            .groupby(["subject", "file_label", "pipeline_label", "backend", "chromophore", "target_status"], as_index=False)
            .agg(mean_score=("beta", "mean"))
            .pivot_table(
                index=["subject", "file_label", "pipeline_label", "backend", "chromophore"],
                columns="target_status",
                values="mean_score",
            )
            .reset_index()
            .rename(columns={"true_target": "mean_target_score", "true_non_target": "mean_non_target_score"})
        )
        if "mean_target_score" not in c_sep.columns:
            c_sep["mean_target_score"] = np.nan
        if "mean_non_target_score" not in c_sep.columns:
            c_sep["mean_non_target_score"] = np.nan
        c_sep["target_minus_non_target_score"] = c_sep["mean_target_score"] - c_sep["mean_non_target_score"]
        c_sep["score_type"] = "canonical_beta"
        sep_tables.append(c_sep)
    if len(block_df) > 0:
        b_sep = (
            block_df
            .groupby(["subject", "file_label", "pipeline_label", "backend", "chromophore", "target_status"], as_index=False)
            .agg(mean_score=("score", "mean"))
            .pivot_table(
                index=["subject", "file_label", "pipeline_label", "backend", "chromophore"],
                columns="target_status",
                values="mean_score",
            )
            .reset_index()
            .rename(columns={"true_target": "mean_target_score", "true_non_target": "mean_non_target_score"})
        )
        if "mean_target_score" not in b_sep.columns:
            b_sep["mean_target_score"] = np.nan
        if "mean_non_target_score" not in b_sep.columns:
            b_sep["mean_non_target_score"] = np.nan
        b_sep["target_minus_non_target_score"] = b_sep["mean_target_score"] - b_sep["mean_non_target_score"]
        b_sep["score_type"] = "block_average_score"
        sep_tables.append(b_sep)
    target_vs_nontarget = pd.concat(sep_tables, ignore_index=True) if sep_tables else pd.DataFrame()

    # Parametric null summary
    parametric_null_summary = pd.DataFrame()
    if len(canonical_df) > 0 and "p_value" in canonical_df.columns:
        null_df = canonical_df.loc[canonical_df["file_label"] == "no_hrf"].copy()
        if len(null_df) > 0:
            null_df["is_false_positive_p_lt_0_05"] = null_df["p_value"] < 0.05
            null_df["is_false_positive_q_lt_0_05"] = null_df["q_value_bh"] < 0.05 if "q_value_bh" in null_df.columns else np.nan
            parametric_null_summary = null_df.groupby(
                ["subject", "pipeline_label", "backend", "chromophore", "target_status"], as_index=False
            ).agg(
                false_positive_rate_p_lt_0_05=("is_false_positive_p_lt_0_05", "mean"),
                false_positive_rate_q_lt_0_05=("is_false_positive_q_lt_0_05", "mean"),
                mean_abs_beta=("beta", lambda x: float(np.mean(np.abs(x)))),
                n_channels=("channel_name", "count"),
            )

    # Empirical null p-values
    empirical_null_shift_df = combined["empirical_null_shift"]
    empirical_null_pvalues_rows = []
    if len(target_vs_nontarget) > 0 and len(empirical_null_shift_df) > 0:
        observed_df = target_vs_nontarget.loc[target_vs_nontarget["score_type"] == "canonical_beta"].copy()
        for _, row in observed_df.iterrows():
            null_rows = empirical_null_shift_df.loc[
                (empirical_null_shift_df["subject"] == row["subject"])
                & (empirical_null_shift_df["pipeline_label"] == row["pipeline_label"])
                & (empirical_null_shift_df["chromophore"] == row["chromophore"])
            ]
            if len(null_rows) == 0:
                continue
            observed_value = float(row["target_minus_non_target_score"])
            null_values = null_rows["target_minus_non_target_score"].to_numpy(dtype=float)
            if row["chromophore"] == "hbo":
                empirical_p = (np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1)
            else:
                empirical_p = (np.sum(null_values <= observed_value) + 1) / (len(null_values) + 1)
            empirical_null_pvalues_rows.append({
                "subject": row["subject"],
                "file_label": row["file_label"],
                "pipeline_label": row["pipeline_label"],
                "backend": row["backend"],
                "chromophore": row["chromophore"],
                "observed_target_minus_non_target_score": observed_value,
                "null_shift_mean": float(np.mean(null_values)),
                "null_shift_std": float(np.std(null_values)),
                "null_shift_min": float(np.min(null_values)),
                "null_shift_max": float(np.max(null_values)),
                "empirical_p_value": empirical_p,
                "n_null_shifts": len(null_values),
            })
    empirical_null_pvalues = pd.DataFrame(empirical_null_pvalues_rows)

    # Variability summaries (core primary family)
    primary_pipelines = [p.label for p in config.pipeline_specs if p.include_in_primary_variability]
    variability_summary = pd.DataFrame()
    pairwise_deltas = pd.DataFrame()
    if len(target_vs_nontarget) > 0:
        variability_df = target_vs_nontarget.loc[
            (target_vs_nontarget["score_type"] == "canonical_beta")
            & (target_vs_nontarget["pipeline_label"].isin(primary_pipelines))
            & (target_vs_nontarget["file_label"] == "hrf_20")
        ].copy()
        if len(variability_df) > 0:
            variability_summary = variability_df.groupby(["subject", "chromophore"], as_index=False).agg(
                mean_target_minus_non_target=("target_minus_non_target_score", "mean"),
                std_across_pipelines=("target_minus_non_target_score", "std"),
                min_across_pipelines=("target_minus_non_target_score", "min"),
                max_across_pipelines=("target_minus_non_target_score", "max"),
                n_pipelines=("pipeline_label", "count"),
            )
            variability_summary["range_across_pipelines"] = variability_summary["max_across_pipelines"] - variability_summary["min_across_pipelines"]

            pivot = variability_df.pivot_table(index=["subject", "chromophore"], columns="pipeline_label", values="target_minus_non_target_score").reset_index()
            pairwise_rows = []
            for _, row in pivot.iterrows():
                for i, left in enumerate(primary_pipelines):
                    for right in primary_pipelines[i + 1:]:
                        if left not in row.index or right not in row.index:
                            continue
                        pairwise_rows.append({
                            "subject": row["subject"],
                            "chromophore": row["chromophore"],
                            "left_pipeline": left,
                            "right_pipeline": right,
                            "left_minus_right": row[left] - row[right],
                            "abs_left_minus_right": abs(row[left] - row[right]),
                        })
            pairwise_deltas = pd.DataFrame(pairwise_rows)

    all_tables = {
        **combined,
        "master_run_log": pd.DataFrame(run_results),
        "roi_scores": roi_scores,
        "target_vs_nontarget_summary": target_vs_nontarget,
        "parametric_null_summary": parametric_null_summary,
        "empirical_null_pvalues": empirical_null_pvalues,
        "variability_summary": variability_summary,
        "pairwise_pipeline_deltas": pairwise_deltas,
        "config": pd.DataFrame([config.to_jsonable()]),
    }
    for name, df in all_tables.items():
        maybe_write_table(df, aggregate_dir / name, config.write_csv, config.write_parquet)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    config = BenchmarkConfig(
        root=args.root,
        dataset_dirname=args.dataset_dirname,
        output_dirname=args.output_dirname,
        output_prefix=args.output_prefix,
        n_workers=args.n_workers,
        overwrite=args.overwrite,
        write_csv=not args.no_csv,
        write_parquet=not args.no_parquet,
        use_matlab=args.use_matlab,
        matlab_cmd=args.matlab_cmd,
        matlab_timeout_s=args.matlab_timeout_s,
        analyzir_path=args.analyzir_path,
        truth_template_dir=args.truth_template_dir,
        empirical_null_shift_count=args.empirical_null_shift_count,
    )
    config.file_specs = default_file_specs()
    config.pipeline_specs = default_pipeline_specs()
    if not config.use_matlab:
        config.pipeline_specs = [p for p in config.pipeline_specs if p.backend != "matlab_arirls"]
    return config


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel semisynthetic fNIRS benchmark with MNE + optional MATLAB AR-IRLS.")
    parser.add_argument("--root", default=str(Path.home() / "fnirs-representation-learning"))
    parser.add_argument("--dataset-dirname", default="snirf_dataset_2")
    parser.add_argument("--output-dirname", default="outputs_benchmark_v6")
    parser.add_argument("--output-prefix", default="benchmark_v6")
    parser.add_argument("--n-workers", type=int, default=max(1, min(os.cpu_count() or 1, 8)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--use-matlab", action="store_true")
    parser.add_argument("--matlab-cmd", default="matlab")
    parser.add_argument("--matlab-timeout-s", type=int, default=7200)
    parser.add_argument("--analyzir-path", default=None)
    parser.add_argument("--truth-template-dir", default="/mnt/data")
    parser.add_argument("--empirical-null-shift-count", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config_from_args(args)

    ensure_dir(config.output_path())
    ensure_dir(config.jobs_path())
    ensure_dir(config.aggregate_path())

    jobs = [(subject, asdict(file_spec), config.to_jsonable()) for subject in config.subject_names for file_spec in config.file_specs]
    print(f"Running {len(jobs)} subject-file jobs with {config.n_workers} workers")
    run_results = []

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        future_to_job = {
            executor.submit(process_subject_file_job, subject, file_spec_dict, config_dict): (subject, file_spec_dict["label"])
            for subject, file_spec_dict, config_dict in jobs
        }
        for future in as_completed(future_to_job):
            subject, file_label = future_to_job[future]
            try:
                result = future.result()
                run_results.append(result)
                print(f"[{result['status']}] {subject} {file_label}")
            except Exception as exc:
                run_results.append({"subject": subject, "file_label": file_label, "status": "crashed-submit", "error": str(exc)})
                print(f"[crashed-submit] {subject} {file_label}: {exc}")

    aggregate_outputs(config, run_results)
    print(f"Done. Aggregate outputs written to: {config.aggregate_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
