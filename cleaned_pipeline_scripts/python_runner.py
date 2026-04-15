#!/usr/bin/env python3
"""Semisynthetic fNIRS benchmark runner.

The code below keeps the existing benchmark behavior but presents the setup
and pipeline registry in a simpler, more reviewable format.
"""
import argparse
import re
import json
import math
import os
import itertools
import importlib
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io
import scipy.integrate


# Keep BLAS/OpenMP from oversubscribing.
for env_key in [
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(env_key, "1")



# Per-process runtime caches. The raw/data caches are cleared at the start of
# each subject-file job so repeated work is avoided across pipelines without
# leaking job-local state forever.
_JOB_RUNTIME_CACHE = {}
_TRUTH_TEMPLATE_CACHE = {}


def clear_job_runtime_caches():
    _JOB_RUNTIME_CACHE.clear()


def runtime_cache_bucket(name):
    return _JOB_RUNTIME_CACHE.setdefault(name, {})


def _cache_key_with_thresholds(raw_obj, config):
    return (
        id(raw_obj),
        float(config['short_separation_threshold_m']),
        float(config['long_separation_threshold_m']),
    )


# -----------------------------------------------------------------------------
# Edit these settings first
# -----------------------------------------------------------------------------

ROOT = str(Path.home() / "fnirs-representation-learning")
DATASET_DIRNAME = "snirf_dataset_2"
OUTPUT_DIRNAME = "benchmark_outputs_full_v10"
OUTPUT_PREFIX = "benchmark_v10"

# Leave empty to run every Subj* folder in the dataset directory.
SUBJECT_NAMES = []

SHORT_SEPARATION_THRESHOLD_M = 0.015
LONG_SEPARATION_THRESHOLD_M = 0.025
LOCAL_SS_MAX_DISTANCE_M = 0.015
MULTI_SS_K = 3
POOLED_SS_N_COMPONENTS = 2
SS_AUX_N_COMPONENTS = 3

STRICT_SCI_THRESHOLD = 0.50
LOOSE_SCI_THRESHOLD = 0.35
STRICT_SNR_THRESHOLD = 2.0
STRICT_NEGATIVE_FRACTION_THRESHOLD = 0.001

FILTER_LOW_HZ = 0.01
FILTER_HIGH_HZ = 0.20
FILTER_HIGHPASS_ONLY_HZ = 0.01
PPF_VALUE = 6.0
STIM_DURATION_S = None
DRIFT_HIGH_PASS_HZ = 0.01
WAVELET_IQR_MULTIPLIER = 1.5
WAVELET_NAME = "db2"
WAVELET_PADDING_MODE = "periodization"
DLPFC_FRONTAL_QUANTILE = 0.65
DLPFC_LATERAL_QUANTILE = 0.50

EPOCH_TMIN = -5.0
EPOCH_TMAX = 30.0
BASELINE_WINDOW = (-5.0, 0.0)
RESPONSE_WINDOW = (4.0, 8.0)

FIR_RESAMPLE_SFREQ_HZ = 1.0
FIR_DELAYS_SCANS = list(range(0, 26))

MATLAB_RESAMPLE_FS_HZ = 0.0
ANALYZIR_RESAMPLE_FS_HZ = 1.0
ANALYZIR_HRF_EXPORT_FS_HZ = ANALYZIR_RESAMPLE_FS_HZ
HOMER3_TCCA_PARAMS = [3.0, 0.08, 10.0]
HOMER3_TCCA_REST_WINDOW_S = [30.0, 210.0]
HOMER3_GLM_TRANGE_S = [-2.0, 17.0]
HOMER3_LOWPASS_HZ = 0.5
HOMER3_AUX_LABEL_ALLOWLIST = ["acc", "ppg", "bp", "resp"]
HOMER3_SS_CHANNEL_SELECTION = [0]

EMPIRICAL_NULL_SHIFT_COUNT = 1
EMPIRICAL_NULL_MIN_SHIFT_S = 20.0

OVERWRITE = False
WRITE_CSV = False
WRITE_PARQUET = True

USE_MATLAB = True
USE_MATLAB_ENGINE = False
PREFER_MATLAB_ENGINE = False
MATLAB_CMD = "matlab"
MATLAB_TIMEOUT_S = 43200
MATLAB_STARTUP_OPTIONS = ""
ANALYZIR_PATH = "/home/asunkari/nirs-toolbox"
HOMER3_PATH = "/home/asunkari/Homer3"

TRUTH_TEMPLATE_DIR = "/home/asunkari/fnirs-representation-learning/truth_templates"

FILE_SPECS = []


def parse_hrf_template_key_from_filename(filename: str) -> str | None:
    m = re.search(r"(hrf_\d+(?:_[A-Za-z0-9]+)*)\.snirf$", filename)
    return m.group(1) if m else None


def parse_amplitude_value_from_template_key(template_key: str | None) -> int:
    if template_key is None:
        return 0
    m = re.match(r"hrf_(\d+)", template_key)
    return int(m.group(1)) if m else 0


def classify_snirf_filename(filename: str) -> dict | None:
    stem = Path(filename).stem

    if filename == "resting_clean.snirf":
        return {
            "label": stem,
            "filename": filename,
            "amplitude_value": 0,
            "is_null": True,
            "annotation_source_filename": "resting_hrf_20.snirf",
            "truth_label_source_filename": "resting_hrf_20.snirf",
            "truth_template_key": None,
        }

    if re.fullmatch(r"resting_clean_surrogate_\d+\.snirf", filename):
        return {
            "label": stem,
            "filename": filename,
            "amplitude_value": 0,
            "is_null": True,
            "annotation_source_filename": "resting_hrf_20.snirf",
            "truth_label_source_filename": "resting_hrf_20.snirf",
            "truth_template_key": None,
        }

    if re.fullmatch(r"resting_hrf_(20|50|100)\.snirf", filename):
        template_key = stem.replace("resting_", "")
        return {
            "label": stem,
            "filename": filename,
            "amplitude_value": parse_amplitude_value_from_template_key(template_key),
            "is_null": False,
            "annotation_source_filename": None,
            "truth_label_source_filename": filename,
            "truth_template_key": template_key,
        }

    if re.fullmatch(r"(orig|sur\d+)_hrf_\d+(?:_[A-Za-z0-9]+)+\.snirf", filename):
        template_key = parse_hrf_template_key_from_filename(filename)
        return {
            "label": stem,
            "filename": filename,
            "amplitude_value": parse_amplitude_value_from_template_key(template_key),
            "is_null": False,
            "annotation_source_filename": None,
            "truth_label_source_filename": filename,
            "truth_template_key": template_key,
        }

    return None


def discover_file_specs_by_subject(dataset_dir: Path, subject_names: list[str]) -> dict[str, list[dict]]:
    discovered = {}
    for subject in subject_names:
        subject_dir = dataset_dir / subject
        specs = []
        for snirf_path in sorted(subject_dir.glob("*.snirf")):
            spec = classify_snirf_filename(snirf_path.name)
            if spec is not None:
                specs.append(spec)
        if not specs:
            raise BenchmarkError(f"No recognized SNIRF files found in {subject_dir}")
        discovered[subject] = specs
    return discovered


def pipeline_spec(
    label,
    *,
    backend,
    nuisance_method,
    hrf_model,
    solver,
    pruning_style="strict_combined",
    motion_method="tddr",
    filter_mode="bandpass",
    use_block_average=False,
    include_in_empirical_null=False,
    include_in_primary_variability=True,
    secondary_pipeline=False,
    comparison_group,
    description,
):
    return {
        "label": label,
        "backend": backend,
        "nuisance_method": nuisance_method,
        "hrf_model": hrf_model,
        "solver": solver,
        "pruning_style": pruning_style,
        "motion_method": motion_method,
        "filter_mode": filter_mode,
        "use_block_average": use_block_average,
        "include_in_empirical_null": include_in_empirical_null,
        "include_in_primary_variability": include_in_primary_variability,
        "secondary_pipeline": secondary_pipeline,
        "comparison_group": comparison_group,
        "description": description,
    }


PYTHON_PIPELINES = [
    pipeline_spec(
        "BlockAvg_LocalSS",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="block_average",
        solver="none",
        use_block_average=True,
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance regression -> block average",
    ),
    pipeline_spec(
        "HighPassOnly_LocalSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="auto",
        filter_mode="highpass_only",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> high-pass only -> local nearest short-separation nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "LocalSS_FIR_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="fir",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> FIR GLM (AUTO)",
    ),
    pipeline_spec(
        "LocalSS_Gamma_Derivs_OLS",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="gamma_derivs",
        solver="ols",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> gamma+derivatives GLM (OLS)",
    ),
    pipeline_spec(
        "LocalSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "LocalSS_Glover_OLS",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="ols",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> Glover GLM (OLS)",
    ),
    pipeline_spec(
        "LocalSS_SPM_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="spm",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> SPM GLM (AUTO)",
    ),
    pipeline_spec(
        "LocalSS_SPM_Derivs_OLS",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="spm_derivs",
        solver="ols",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> SPM+derivatives GLM (OLS)",
    ),
    pipeline_spec(
        "LocalSS_SPM_OLS",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="spm",
        solver="ols",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> local nearest short-separation nuisance -> SPM GLM (OLS)",
    ),
    pipeline_spec(
        "LooseQC_LocalSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="auto",
        pruning_style="loose_sci",
        comparison_group="python_rerun_optimized",
        description="LooseSCI QC -> TDDR -> band-pass -> local nearest short-separation nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "MultiSSOrth3_Glover_AUTO",
        backend="python_mne",
        nuisance_method="multi_ss_orth3",
        hrf_model="glover",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> orthogonalized 3-short-channel nuisance set -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "NoMotion_LocalSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="auto",
        motion_method="none",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> no motion correction -> band-pass -> local nearest short-separation nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "NoSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="none",
        hrf_model="glover",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> no short-separation nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "NoSS_Glover_OLS",
        backend="python_mne",
        nuisance_method="none",
        hrf_model="glover",
        solver="ols",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> no short-separation nuisance -> Glover GLM (OLS)",
    ),
    pipeline_spec(
        "PooledPCA2_Glover_AUTO",
        backend="python_mne",
        nuisance_method="pooled_pca2",
        hrf_model="glover",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> pooled short-separation PCA(2) nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "SSAuxPCA_Glover_AUTO",
        backend="python_mne",
        nuisance_method="ss_aux_pca",
        hrf_model="glover",
        solver="auto",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> TDDR -> band-pass -> joint short-separation plus auxiliary PCA nuisance -> Glover GLM (AUTO)",
    ),
    pipeline_spec(
        "WaveletMC_LocalSS_Glover_AUTO",
        backend="python_mne",
        nuisance_method="local_nearest",
        hrf_model="glover",
        solver="auto",
        motion_method="wavelet",
        comparison_group="python_rerun_optimized",
        description="StrictQC -> wavelet motion correction -> band-pass -> local nearest short-separation nuisance -> Glover GLM (AUTO)",
    ),
]

MATLAB_ANALYZIR_PIPELINES = [
    pipeline_spec(
        "LocalSSFilter_Canonical_ARIRLS",
        backend="matlab_arirls_native",
        nuisance_method="native_ss_filter",
        hrf_model="canonical",
        solver="arirls",
        motion_method="tddr",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline with short-separation filtering, canonical HRF, and AR-IRLS",
    ),
    pipeline_spec(
        "LocalSSReg_CanonicalDerivs_ARIRLS",
        backend="matlab_arirls_native",
        nuisance_method="native_ss_regressors",
        hrf_model="canonical_derivs",
        solver="arirls",
        motion_method="tddr",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline with short-separation regressors, canonical+derivatives HRF, and AR-IRLS",
    ),
    pipeline_spec(
        "LocalSSReg_Canonical_ARIRLS",
        backend="matlab_arirls_native",
        nuisance_method="native_ss_regressors",
        hrf_model="canonical",
        solver="arirls",
        motion_method="tddr",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline with short-separation regressors, canonical HRF, and AR-IRLS",
    ),
    pipeline_spec(
        "NoSS_Canonical_ARIRLS",
        backend="matlab_arirls_native",
        nuisance_method="none",
        hrf_model="canonical",
        solver="arirls",
        motion_method="tddr",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline without short-separation regressors, canonical HRF, and AR-IRLS",
    ),
    pipeline_spec(
        "NoSS_Canonical_ARIRLS_NoTDDR",
        backend="matlab_arirls_native",
        nuisance_method="none",
        hrf_model="canonical",
        solver="arirls",
        motion_method="none",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline without short-separation regressors or TDDR, canonical HRF, and AR-IRLS",
    ),
    pipeline_spec(
        "NoSS_Canonical_OLS",
        backend="matlab_arirls_native",
        nuisance_method="none",
        hrf_model="canonical",
        solver="ols",
        motion_method="tddr",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline without short-separation regressors, canonical HRF, and OLS",
    ),
    pipeline_spec(
        "NoSS_Canonical_OLS_NoTDDR",
        backend="matlab_arirls_native",
        nuisance_method="none",
        hrf_model="canonical",
        solver="ols",
        motion_method="none",
        filter_mode="native_default",
        comparison_group="analyzir_repair",
        description="AnalyzIR native pipeline without short-separation regressors or TDDR, canonical HRF, and OLS",
    ),
]

HOMER3_PIPELINES = [
    pipeline_spec(
        "Homer3_tCCA_Gaussian",
        backend="matlab_homer3",
        nuisance_method="homer3_tcca",
        hrf_model="homer3_gaussian",
        solver="homer3_glm",
        motion_method="homer3_tcca",
        filter_mode="homer3_native",
        comparison_group="homer3_gaussian_rerun",
        description="Homer3 tCCA followed by native Homer3 GLM with Gaussian basis",
    ),
]

PIPELINE_SPECS = PYTHON_PIPELINES + MATLAB_ANALYZIR_PIPELINES + HOMER3_PIPELINES

PIPELINE_LABEL_FILTER = None
if PIPELINE_LABEL_FILTER:
    keep = set(PIPELINE_LABEL_FILTER)
    PIPELINE_SPECS = [spec for spec in PIPELINE_SPECS if spec['label'] in keep]


def root_path(config=None):
    root = ROOT if config is None else config["root"]
    return Path(root).expanduser().resolve()


def dataset_path(config=None):
    if config is None:
        return root_path() / DATASET_DIRNAME
    return root_path(config) / config["dataset_dirname"]


def output_path(config=None):
    if config is None:
        return root_path() / OUTPUT_DIRNAME
    return root_path(config) / config["output_dirname"]


def jobs_path(config=None):
    return output_path(config) / "job_results"


def aggregate_path(config=None):
    return output_path(config) / "aggregate"


def config_snapshot():
    return {
        "root": ROOT,
        "dataset_dirname": DATASET_DIRNAME,
        "output_dirname": OUTPUT_DIRNAME,
        "output_prefix": OUTPUT_PREFIX,
        "subject_names": list(SUBJECT_NAMES),
        "file_specs": [dict(x) for x in FILE_SPECS],
        "pipeline_specs": [dict(x) for x in PIPELINE_SPECS],
        "short_separation_threshold_m": SHORT_SEPARATION_THRESHOLD_M,
        "long_separation_threshold_m": LONG_SEPARATION_THRESHOLD_M,
        "local_ss_max_distance_m": LOCAL_SS_MAX_DISTANCE_M,
        "multi_ss_k": MULTI_SS_K,
        "pooled_ss_n_components": POOLED_SS_N_COMPONENTS,
        "ss_aux_n_components": SS_AUX_N_COMPONENTS,
        "strict_sci_threshold": STRICT_SCI_THRESHOLD,
        "loose_sci_threshold": LOOSE_SCI_THRESHOLD,
        "strict_snr_threshold": STRICT_SNR_THRESHOLD,
        "strict_negative_fraction_threshold": STRICT_NEGATIVE_FRACTION_THRESHOLD,
        "filter_low_hz": FILTER_LOW_HZ,
        "filter_high_hz": FILTER_HIGH_HZ,
        "filter_highpass_only_hz": FILTER_HIGHPASS_ONLY_HZ,
        "ppf_value": PPF_VALUE,
        "stim_duration_s": STIM_DURATION_S,
        "drift_high_pass_hz": DRIFT_HIGH_PASS_HZ,
        "wavelet_iqr_multiplier": WAVELET_IQR_MULTIPLIER,
        "wavelet_name": WAVELET_NAME,
        "wavelet_padding_mode": WAVELET_PADDING_MODE,
        "dlpfc_frontal_quantile": DLPFC_FRONTAL_QUANTILE,
        "dlpfc_lateral_quantile": DLPFC_LATERAL_QUANTILE,
        "epoch_tmin": EPOCH_TMIN,
        "epoch_tmax": EPOCH_TMAX,
        "baseline_window": BASELINE_WINDOW,
        "response_window": RESPONSE_WINDOW,
        "fir_resample_sfreq_hz": FIR_RESAMPLE_SFREQ_HZ,
        "fir_delays_scans": list(FIR_DELAYS_SCANS),
        "matlab_resample_fs_hz": MATLAB_RESAMPLE_FS_HZ,
        "analyzir_resample_fs_hz": ANALYZIR_RESAMPLE_FS_HZ,
        "analyzir_hrf_export_fs_hz": ANALYZIR_HRF_EXPORT_FS_HZ,
        "homer3_tcca_params": list(HOMER3_TCCA_PARAMS),
        "homer3_tcca_rest_window_s": list(HOMER3_TCCA_REST_WINDOW_S),
        "homer3_glm_trange_s": list(HOMER3_GLM_TRANGE_S),
        "homer3_lowpass_hz": HOMER3_LOWPASS_HZ,
        "homer3_aux_label_allowlist": list(HOMER3_AUX_LABEL_ALLOWLIST),
        "homer3_ss_channel_selection": list(HOMER3_SS_CHANNEL_SELECTION),
        "empirical_null_shift_count": EMPIRICAL_NULL_SHIFT_COUNT,
        "empirical_null_min_shift_s": EMPIRICAL_NULL_MIN_SHIFT_S,
        "overwrite": OVERWRITE,
        "write_csv": WRITE_CSV,
        "write_parquet": WRITE_PARQUET,
        "use_matlab": USE_MATLAB,
        "use_matlab_engine": USE_MATLAB_ENGINE,
        "prefer_matlab_engine": PREFER_MATLAB_ENGINE,
        "matlab_cmd": MATLAB_CMD,
        "matlab_timeout_s": MATLAB_TIMEOUT_S,
        "matlab_startup_options": MATLAB_STARTUP_OPTIONS,
        "analyzir_path": ANALYZIR_PATH,
        "homer3_path": HOMER3_PATH,
        "truth_template_dir": TRUTH_TEMPLATE_DIR,
    }


class BenchmarkError(RuntimeError):
    pass

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def maybe_write_table(df, path_base, write_csv=True, write_parquet=True):
    if df is None or len(df) == 0:
        return
    if write_csv:
        df.to_csv(path_base.with_suffix('.csv'), index=False)
    if write_parquet:
        try:
            df.to_parquet(path_base.with_suffix('.parquet'), index=False)
        except Exception as exc:
            path_base.with_name(path_base.name + '__parquet_failed.txt').write_text(str(exc), encoding='utf-8')

def read_any_table(path_base):
    parquet_path = path_base.with_suffix('.parquet')
    csv_path = path_base.with_suffix('.csv')
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()

def pipeline_manifest_df(config):
    rows = []
    for order_index, pipeline in enumerate(config['pipeline_specs'], start=1):
        rows.append({'pipeline_order': order_index, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'nuisance_method': pipeline['nuisance_method'], 'hrf_model': pipeline['hrf_model'], 'solver': pipeline['solver'], 'pruning_style': pipeline['pruning_style'], 'motion_method': pipeline['motion_method'], 'filter_mode': pipeline['filter_mode'], 'use_block_average': pipeline['use_block_average'], 'include_in_empirical_null': pipeline['include_in_empirical_null'], 'include_in_primary_variability': pipeline['include_in_primary_variability'], 'secondary_pipeline': pipeline['secondary_pipeline'], 'comparison_group': pipeline['comparison_group'], 'description': pipeline['description']})
    return pd.DataFrame(rows)

def attach_pipeline_metadata(df, config):
    if df is None or len(df) == 0 or 'pipeline_label' not in df.columns:
        return df
    manifest = pipeline_manifest_df(config)
    join_cols = [c for c in manifest.columns if c != 'pipeline_label' and c not in df.columns]
    if not join_cols:
        return df
    return df.merge(manifest[['pipeline_label', *join_cols]], on='pipeline_label', how='left')

def optional_import_matlab_engine():
    try:
        matlab = importlib.import_module('matlab')
        matlab_engine = importlib.import_module('matlab.engine')
    except Exception:
        return (None, None)
    return (matlab, matlab_engine)

def benjamini_hochberg(p_values):
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

def safe_standardize_rows(data_matrix):
    data_matrix = np.asarray(data_matrix, dtype=float)
    if data_matrix.ndim == 1:
        data_matrix = data_matrix[None, :]
    centered = data_matrix - data_matrix.mean(axis=1, keepdims=True)
    stds = data_matrix.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return centered / stds

def principal_components_rows(data_matrix, n_components):
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

def qr_orth_columns(matrix):
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

def make_error_row(subject, file_label, pipeline_label, stage, error_message, channel_name=None):
    return {'timestamp_utc': now_utc_iso(), 'subject': subject, 'file_label': file_label, 'pipeline_label': pipeline_label, 'stage': stage, 'channel_name': channel_name, 'error_message': str(error_message)}

def import_mne_nirs_modules():
    try:
        from mne_nirs.experimental_design import make_first_level_design_matrix
        from mne_nirs.statistics import run_glm
        from mne_nirs.io import read_snirf_aux_data
    except Exception as exc:
        raise BenchmarkError('mne_nirs is required for this benchmark. Install mne-nirs (and nilearn) in the target environment.') from exc
    return (make_first_level_design_matrix, run_glm, read_snirf_aux_data)

def import_nilearn_functions():
    try:
        from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix
    except Exception as exc:
        raise BenchmarkError('nilearn is required for custom HRF basis functions and design-matrix helpers.') from exc
    return (compute_regressor, make_first_level_design_matrix)

def load_truth_templates(config):
    if config["truth_template_dir"] is None:
        return {}

    template_dir = str(Path(config["truth_template_dir"]).expanduser().resolve())
    file_specs_source = config.get("file_specs", [])
    if isinstance(file_specs_source, dict):
        all_specs = [spec for specs in file_specs_source.values() for spec in specs]
    else:
        all_specs = list(file_specs_source)
    needed_keys = tuple(sorted({spec.get("truth_template_key") for spec in all_specs if spec.get("truth_template_key")}))
    cache_key = (template_dir, needed_keys)
    if cache_key in _TRUTH_TEMPLATE_CACHE:
        return _TRUTH_TEMPLATE_CACHE[cache_key]

    templates = {}
    template_dir_path = Path(template_dir)
    for template_key in needed_keys:
        template_path = template_dir_path / f"{template_key}.mat"
        if not template_path.exists():
            continue
        mat = scipy.io.loadmat(template_path, squeeze_me=True, struct_as_record=False)
        if "hrf" not in mat:
            continue
        hrf = mat["hrf"]
        hrf_conc = np.asarray(hrf.hrf_conc, dtype=float)
        templates[template_key] = {
            "time_s": np.asarray(hrf.t_hrf, dtype=float).reshape(-1),
            "hbo": hrf_conc[:, 0].reshape(-1),
            "hbr": hrf_conc[:, 1].reshape(-1),
            "hbt": hrf_conc[:, 2].reshape(-1),
        }
    _TRUTH_TEMPLATE_CACHE[cache_key] = templates
    return templates

def read_measurement_data_type_labels(snirf_file_path):
    if not snirf_file_path.exists():
        return None
    with h5py.File(snirf_file_path, 'r') as h5_file:
        if 'nirs' not in h5_file:
            return None
        nirs_group = h5_file['nirs']
        data_group_name = 'data1' if 'data1' in nirs_group else next((k for k in nirs_group.keys() if k.startswith('data')), None)
        if data_group_name is None:
            return None
        data_group = nirs_group[data_group_name]
        entries = []
        for key in data_group.keys():
            if not key.startswith('measurementList'):
                continue
            suffix = key.replace('measurementList', '')
            try:
                idx = int(suffix)
            except ValueError:
                continue
            measurement_group = data_group[key]
            label_value = measurement_group['dataTypeLabel'][()] if 'dataTypeLabel' in measurement_group else 0
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

def get_cw_channel_indices(raw):
    picks_fnirs = mne.pick_types(raw.info, fnirs=True)
    channel_types = np.asarray(raw.get_channel_types())
    return picks_fnirs[channel_types[picks_fnirs] == 'fnirs_cw_amplitude']

def sanitize_annotations_to_single_task(raw, default_description='task'):
    raw = raw.copy()
    kept_onsets, kept_durations, kept_descriptions = ([], [], [])
    for onset_s, duration_s, description in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if str(description).lower().startswith('bad'):
            continue
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(duration_s) if float(duration_s) > 0 else 1.0)
        kept_descriptions.append(default_description)
    raw.set_annotations(mne.Annotations(onset=kept_onsets, duration=kept_durations, description=kept_descriptions))
    return raw

def copy_valid_annotations(raw_target_cw, annotation_source_file_path):
    raw_source = mne.io.read_raw_snirf(annotation_source_file_path, preload=False, verbose=False)
    source_annotations = raw_source.annotations
    target_duration_s = float(raw_target_cw.times[-1])
    kept_onsets, kept_durations, kept_descriptions = ([], [], [])
    for onset_s, duration_s, description in zip(source_annotations.onset, source_annotations.duration, source_annotations.description):
        if str(description).lower().startswith('bad'):
            continue
        if float(onset_s) >= target_duration_s:
            continue
        clipped_duration = min(float(duration_s) if float(duration_s) > 0 else 1.0, max(0.0, target_duration_s - float(onset_s)))
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(clipped_duration))
        kept_descriptions.append('task')
    raw_target_cw = raw_target_cw.copy()
    raw_target_cw.set_annotations(mne.Annotations(onset=kept_onsets, duration=kept_durations, description=kept_descriptions))
    return raw_target_cw

def build_cw_channel_table(raw_cw, subject, file_label, config):
    from mne.preprocessing.nirs import source_detector_distances, short_channels
    picks_cw = get_cw_channel_indices(raw_cw)
    cw_names = np.asarray(raw_cw.ch_names)[picks_cw]
    distances_m = source_detector_distances(raw_cw.info, picks=picks_cw)
    short_mask_all = short_channels(raw_cw.info, threshold=config['short_separation_threshold_m'])
    short_mask = short_mask_all[picks_cw]
    long_mask = distances_m >= config['long_separation_threshold_m']
    pair_names = np.asarray([name.split(' ')[0] for name in cw_names])
    positions = np.asarray([raw_cw.info['chs'][int(idx)]['loc'][:3] for idx in picks_cw], dtype=float)
    df = pd.DataFrame({'subject': subject, 'file_label': file_label, 'channel_name': cw_names, 'pair_name': pair_names, 'distance_m': distances_m, 'midpoint_x': positions[:, 0], 'midpoint_y': positions[:, 1], 'midpoint_z': positions[:, 2], 'is_ss': short_mask, 'is_ls': long_mask})
    df['hemisphere'] = np.where(df['midpoint_x'] < 0, 'left', np.where(df['midpoint_x'] > 0, 'right', 'midline'))
    df['group'] = np.select([df['is_ss'], df['is_ls']], ['SS', 'LS'], default='MID')
    return df

def _build_hb_channel_table_base(raw_hb, config):
    from mne.preprocessing.nirs import source_detector_distances, short_channels
    picks_hbo = mne.pick_types(raw_hb.info, fnirs='hbo')
    picks_hbr = mne.pick_types(raw_hb.info, fnirs='hbr')
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))
    hb_names = np.asarray(raw_hb.ch_names)[picks_hb]
    hb_types = np.asarray(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.asarray([name.split(' ')[0] for name in hb_names])
    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]
    short_mask_all = short_channels(raw_hb.info, threshold=config['short_separation_threshold_m'])
    short_mask = short_mask_all[picks_hb]
    long_mask = distances_hb >= config['long_separation_threshold_m']
    positions = np.asarray([raw_hb.info['chs'][int(idx)]['loc'][:3] for idx in picks_hb], dtype=float)
    df = pd.DataFrame({
        'channel_name': hb_names,
        'pair_name': pair_names,
        'chromophore': hb_types,
        'distance_m': distances_hb,
        'midpoint_x': positions[:, 0],
        'midpoint_y': positions[:, 1],
        'midpoint_z': positions[:, 2],
        'is_ss': short_mask,
        'is_ls': long_mask,
    })
    df['hemisphere'] = np.where(df['midpoint_x'] < 0, 'left', np.where(df['midpoint_x'] > 0, 'right', 'midline'))
    df['group'] = np.select([df['is_ss'], df['is_ls']], ['SS', 'LS'], default='MID')
    return df


def _get_hb_channel_table_base(raw_hb, config):
    cache = runtime_cache_bucket('hb_channel_table')
    cache_key = _cache_key_with_thresholds(raw_hb, config)
    if cache_key not in cache:
        cache[cache_key] = _build_hb_channel_table_base(raw_hb, config)
    return cache[cache_key]


def _get_raw_array(raw_obj):
    cache = runtime_cache_bucket('raw_data')
    cache_key = id(raw_obj)
    if cache_key not in cache:
        cache[cache_key] = raw_obj.get_data()
    return cache[cache_key]


def _get_channel_index_map(raw_obj):
    cache = runtime_cache_bucket('channel_index_map')
    cache_key = id(raw_obj)
    if cache_key not in cache:
        cache[cache_key] = {name: idx for idx, name in enumerate(raw_obj.ch_names)}
    return cache[cache_key]


def _get_channel_position_map(raw_obj):
    cache = runtime_cache_bucket('channel_position_map')
    cache_key = id(raw_obj)
    if cache_key not in cache:
        cache[cache_key] = {name: np.asarray(ch['loc'][:3], dtype=float) for name, ch in zip(raw_obj.ch_names, raw_obj.info['chs'])}
    return cache[cache_key]


def _get_nuisance_signature(nuisance_df, nuisance_metadata):
    metadata_key = json.dumps(nuisance_metadata or {}, sort_keys=True, default=str)
    if nuisance_df is None or nuisance_df.shape[1] == 0:
        return ('none', metadata_key)
    return (tuple(str(c) for c in nuisance_df.columns), metadata_key)


def _build_design_matrix_cached(raw_hb, hrf_model, nuisance_df, nuisance_signature, config):
    cache = runtime_cache_bucket('design_matrix')
    cache_key = (id(raw_hb), str(hrf_model), nuisance_signature)
    if cache_key not in cache:
        cache[cache_key] = build_design_matrix(raw_hb, hrf_model, nuisance_df, config)
    return cache[cache_key]


def _resample_for_fir_cached(raw_hb, config):
    cache = runtime_cache_bucket('fir_raw')
    cache_key = (id(raw_hb), float(config['fir_resample_sfreq_hz']))
    if cache_key not in cache:
        cache[cache_key] = resample_for_fir(raw_hb, config)
    return cache[cache_key]

def build_hb_channel_table(raw_hb, subject, file_label, config):
    df = _get_hb_channel_table_base(raw_hb, config).copy()
    df.insert(0, 'file_label', file_label)
    df.insert(0, 'subject', subject)
    return df

def build_quality_tables(raw_cw, subject, file_label, config):
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
    channel_quality = pd.DataFrame({'subject': subject, 'file_label': file_label, 'channel_name': cw_names, 'pair_name': cw_channel_table['pair_name'].to_numpy(), 'distance_m': cw_channel_table['distance_m'].to_numpy(), 'group': cw_channel_table['group'].to_numpy(), 'sci': sci_values, 'snr': snr_values, 'negative_fraction': negative_fraction_values})
    pair_geometry = cw_channel_table.groupby(['subject', 'file_label', 'pair_name'], as_index=False).agg(distance_m=('distance_m', 'first'), group=('group', 'first'), midpoint_x=('midpoint_x', 'mean'), midpoint_y=('midpoint_y', 'mean'), midpoint_z=('midpoint_z', 'mean'), hemisphere=('hemisphere', 'first'))
    pair_quality = channel_quality.groupby(['subject', 'file_label', 'pair_name'], as_index=False).agg(sci_min=('sci', 'min'), sci_mean=('sci', 'mean'), snr_min=('snr', 'min'), snr_mean=('snr', 'mean'), negative_fraction_max=('negative_fraction', 'max')).merge(pair_geometry, on=['subject', 'file_label', 'pair_name'], how='left')
    return (channel_quality, pair_quality)

def get_bad_pair_names(pair_quality, pruning_style, config):
    if pruning_style == 'strict_combined':
        bad_mask = (pair_quality['sci_min'] < config['strict_sci_threshold']) | (pair_quality['snr_min'] < config['strict_snr_threshold']) | (pair_quality['negative_fraction_max'] > config['strict_negative_fraction_threshold'])
    elif pruning_style == 'loose_sci':
        bad_mask = pair_quality['sci_min'] < config['loose_sci_threshold']
    else:
        raise ValueError(f'Unknown pruning_style {pruning_style!r}')
    return pair_quality.loc[bad_mask, 'pair_name'].astype(str).tolist()

def apply_bad_pairs_to_hb(raw_hb, bad_pair_names):
    raw_hb = raw_hb.copy()
    bad_pairs = set(bad_pair_names)
    raw_hb.info['bads'] = [name for name in raw_hb.ch_names if name.split(' ')[0] in bad_pairs]
    return raw_hb

def get_channel_position(raw, channel_name):
    position_map = _get_channel_position_map(raw)
    return position_map[channel_name]

def get_available_long_channel_names(raw_hb, config):
    hb_table = _get_hb_channel_table_base(raw_hb, config)
    long_names = hb_table.loc[hb_table['group'] == 'LS', 'channel_name'].astype(str).tolist()
    bads = set(raw_hb.info['bads'])
    return sorted([name for name in long_names if name not in bads])

def get_available_hb_channel_names(raw_hb, include_short, config):
    hb_table = _get_hb_channel_table_base(raw_hb, config)
    keep = hb_table['group'].isin(['LS', 'SS']) if include_short else (hb_table['group'] == 'LS')
    names = hb_table.loc[keep, 'channel_name'].astype(str).tolist()
    bads = set(raw_hb.info['bads'])
    return [name for name in names if name not in bads]

def extract_sd_geometry(raw_hb, channel_names, config):
    hb_table = _get_hb_channel_table_base(raw_hb, config).set_index('channel_name')
    ch_index_map = _get_channel_index_map(raw_hb)
    src_keys = []
    det_keys = []
    src_positions = []
    det_positions = []
    source_ids = []
    detector_ids = []
    is_long = []
    is_short = []
    src_map = {}
    det_map = {}
    next_src = 1
    next_det = 1
    for channel_name in channel_names:
        idx = ch_index_map[channel_name]
        loc = np.asarray(raw_hb.info['chs'][idx]['loc'], dtype=float)
        src_pos = np.asarray(loc[3:6], dtype=float)
        det_pos = np.asarray(loc[6:9], dtype=float)
        src_key = tuple(np.round(src_pos, 8).tolist())
        det_key = tuple(np.round(det_pos, 8).tolist())
        if src_key not in src_map:
            src_map[src_key] = next_src
            next_src += 1
        if det_key not in det_map:
            det_map[det_key] = next_det
            next_det += 1
        source_ids.append(src_map[src_key])
        detector_ids.append(det_map[det_key])
        src_positions.append(src_pos)
        det_positions.append(det_pos)
        src_keys.append(src_key)
        det_keys.append(det_key)
        row = hb_table.loc[channel_name]
        is_long.append(bool(row['group'] == 'LS'))
        is_short.append(bool(row['group'] == 'SS'))
    return {
        'source_ids': np.asarray(source_ids, dtype=np.int32),
        'detector_ids': np.asarray(detector_ids, dtype=np.int32),
        'source_positions_m': np.asarray(src_positions, dtype=float),
        'detector_positions_m': np.asarray(det_positions, dtype=float),
        'channel_is_long': np.asarray(is_long, dtype=np.int32),
        'channel_is_short': np.asarray(is_short, dtype=np.int32),
    }

def get_available_short_channel_names(raw_hb, chromophore, config):
    hb_table = _get_hb_channel_table_base(raw_hb, config)
    short_names = hb_table.loc[(hb_table['group'] == 'SS') & (hb_table['chromophore'] == chromophore), 'channel_name'].astype(str).tolist()
    bads = set(raw_hb.info['bads'])
    return [name for name in short_names if name not in bads]

def read_auxiliary_dataframe(snirf_file_path, raw_cw):
    _, _, read_snirf_aux_data = import_mne_nirs_modules()
    try:
        aux_df = read_snirf_aux_data(str(snirf_file_path), raw_cw)
    except Exception:
        return pd.DataFrame(index=raw_cw.times)
    if aux_df is None or len(aux_df.columns) == 0:
        return pd.DataFrame(index=raw_cw.times)
    aux_df = aux_df.copy()
    kept_cols = []
    for col in aux_df.columns:
        values = np.asarray(aux_df[col], dtype=float)
        if not np.isfinite(values).any():
            continue
        if np.nanstd(values) < 1e-12:
            continue
        kept_cols.append(col)
    return aux_df[kept_cols] if kept_cols else pd.DataFrame(index=raw_cw.times)

def gamma_hrf_core(t_r, oversampling=50, peak_time=6.0, peak_disp=1.0, duration=32.0):
    dt = float(t_r) / float(oversampling)
    t = np.arange(0.0, duration + dt, dt)
    h = peak_disp ** peak_time * t ** (peak_time - 1.0) * np.exp(-peak_disp * t) / math.gamma(peak_time)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    total = h.sum()
    return h / total if total > 0 else h

def gamma_hrf_function(t_r, oversampling=50):
    return gamma_hrf_core(t_r, oversampling=oversampling, peak_time=6.0, peak_disp=1.0, duration=32.0)

def gamma_time_derivative(t_r, oversampling=50):
    eps = 0.1
    return (
        gamma_hrf_core(t_r, oversampling=oversampling, peak_time=6.0 + eps, peak_disp=1.0, duration=32.0)
        - gamma_hrf_core(t_r, oversampling=oversampling, peak_time=6.0 - eps, peak_disp=1.0, duration=32.0)
    ) / (2.0 * eps)

def gamma_dispersion_derivative(t_r, oversampling=50):
    eps = 0.05
    return (
        gamma_hrf_core(t_r, oversampling=oversampling, peak_time=6.0, peak_disp=1.0 + eps, duration=32.0)
        - gamma_hrf_core(t_r, oversampling=oversampling, peak_time=6.0, peak_disp=1.0 - eps, duration=32.0)
    ) / (2.0 * eps)

def analyzir_canonical_hrf(times_s, a1=4.0, a2=16.0, b1=1.0, b2=1.0, c=1.0/6.0):
    t = np.clip(times_s, 0.0, None)
    h = b1 ** a1 * t ** (a1 - 1.0) * np.exp(-b1 * t) / math.gamma(a1)
    h -= c * b2 ** a2 * t ** (a2 - 1.0) * np.exp(-b2 * t) / math.gamma(a2)
    total = np.sum(h)
    return h / total if abs(total) > 0 else h

def analyzir_canonical_derivatives(times_s):
    eps_a = 0.05
    eps_b = 0.02
    base = analyzir_canonical_hrf(times_s)
    d_a1 = (analyzir_canonical_hrf(times_s, a1=4.0 + eps_a) - analyzir_canonical_hrf(times_s, a1=4.0 - eps_a)) / (2.0 * eps_a)
    d_b1 = (analyzir_canonical_hrf(times_s, b1=1.0 + eps_b) - analyzir_canonical_hrf(times_s, b1=1.0 - eps_b)) / (2.0 * eps_b)
    return np.column_stack([base, d_a1, d_b1])

def get_truth_curve(templates, truth_template_key, epoch_times_s, chromophore):
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)

    if truth_template_key is None:
        return np.zeros_like(epoch_times_s, dtype=float)

    if truth_template_key not in templates:
        return np.full_like(epoch_times_s, np.nan, dtype=float)

    tmpl = templates[truth_template_key]
    return np.interp(
        epoch_times_s,
        tmpl["time_s"],
        tmpl[chromophore],
        left=0.0,
        right=0.0,
    )

def get_nilearn_hrf_model(hrf_model):
    if hrf_model == 'spm':
        return 'spm'
    if hrf_model == 'spm_derivs':
        return 'spm + derivative + dispersion'
    if hrf_model == 'glover':
        return 'glover'
    if hrf_model == 'glover_derivs':
        return 'glover + derivative + dispersion'
    if hrf_model == 'gamma':
        return [gamma_hrf_function]
    if hrf_model == 'gamma_derivs':
        return [gamma_hrf_function, gamma_time_derivative, gamma_dispersion_derivative]
    raise ValueError(f'Unknown Nilearn HRF model {hrf_model!r}')

def get_annotation_events_df(raw_obj, config):
    if raw_obj is None or raw_obj.annotations is None or len(raw_obj.annotations) == 0:
        return pd.DataFrame(columns=['trial_type', 'onset', 'duration'])
    onsets = np.asarray(raw_obj.annotations.onset - raw_obj.first_time, dtype=float)
    durations = np.asarray(raw_obj.annotations.duration, dtype=float)
    descriptions = [str(desc) for desc in raw_obj.annotations.description]
    valid = np.isfinite(durations) & (durations > 0)
    if valid.any():
        fallback_duration = float(np.median(durations[valid]))
    elif config.get('stim_duration_s') is not None:
        fallback_duration = float(config['stim_duration_s'])
    else:
        fallback_duration = 1.0
    durations = np.where(valid, durations, fallback_duration)
    event_mask = np.isfinite(onsets)
    if not np.any(event_mask):
        return pd.DataFrame(columns=['trial_type', 'onset', 'duration'])
    return pd.DataFrame({
        'trial_type': np.asarray(descriptions, dtype=object)[event_mask],
        'onset': onsets[event_mask],
        'duration': durations[event_mask],
    })

def get_effective_stim_duration_s(raw_obj, config):
    if raw_obj is None or raw_obj.annotations is None or len(raw_obj.annotations) == 0:
        fallback = config.get('stim_duration_s')
        return float(fallback) if fallback is not None else 1.0
    durations = np.asarray(raw_obj.annotations.duration, dtype=float)
    durations = durations[np.isfinite(durations) & (durations > 0)]
    if durations.size == 0:
        fallback = config.get('stim_duration_s')
        return float(fallback) if fallback is not None else 1.0
    return float(np.median(durations))

def build_basis_curves(epoch_times_s, hrf_model, fir_delays=None, fir_betas=None, stim_duration_s=None):
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    compute_regressor, _ = import_nilearn_functions()
    if hrf_model in {'glover', 'glover_derivs', 'spm', 'spm_derivs', 'gamma', 'gamma_derivs'}:
        duration_s = float(stim_duration_s) if stim_duration_s is not None and np.isfinite(stim_duration_s) and stim_duration_s > 0 else 1.0
        signal, _ = compute_regressor(
            np.array([[0.0], [duration_s], [1.0]], dtype=float),
            get_nilearn_hrf_model(hrf_model),
            epoch_times_s,
            con_id='task',
            oversampling=50,
        )
        signal = np.asarray(signal, dtype=float)
        if signal.ndim == 1:
            signal = signal[:, None]
        return signal
    if hrf_model == 'canonical':
        return analyzir_canonical_hrf(epoch_times_s)[:, None]
    if hrf_model == 'canonical_derivs':
        return analyzir_canonical_derivatives(epoch_times_s)
    if hrf_model == 'fir':
        if fir_delays is None or fir_betas is None:
            raise ValueError('FIR basis requires delays and betas.')
        delays = np.asarray(fir_delays, dtype=float).reshape(-1)
        betas = np.asarray(fir_betas, dtype=float).reshape(-1)
        if delays.size != betas.size:
            raise ValueError('FIR delays and betas must have the same length.')
        if delays.size == 0:
            return np.zeros((len(epoch_times_s), 1), dtype=float)
        order = np.argsort(delays)
        delays = delays[order]
        betas = betas[order]
        if epoch_times_s.size > 1:
            sample_step_s = float(np.median(np.diff(epoch_times_s)))
            if not np.isfinite(sample_step_s) or sample_step_s <= 0:
                sample_step_s = 1.0
        else:
            sample_step_s = 1.0
        fir_delays_scans = [int(round(delay_s / sample_step_s)) for delay_s in delays]
        duration_s = float(stim_duration_s) if stim_duration_s is not None and np.isfinite(stim_duration_s) and stim_duration_s > 0 else sample_step_s
        _, nilearn_make_dm = import_nilearn_functions()
        fir_events = pd.DataFrame({'trial_type': ['task'], 'onset': [0.0], 'duration': [duration_s]})
        fir_dm = nilearn_make_dm(
            frame_times=epoch_times_s,
            events=fir_events,
            hrf_model='fir',
            drift_model=None,
            fir_delays=fir_delays_scans,
            oversampling=1,
        )
        basis_columns = []
        for delay_scan in fir_delays_scans:
            candidates = [
                f'task_delay_{delay_scan}',
                f'task_{delay_scan}',
                f'task_{int(delay_scan)}',
            ]
            match = next((col for col in fir_dm.columns if str(col) in candidates), None)
            if match is None:
                match = next((col for col in fir_dm.columns if str(col).startswith('task') and str(col).endswith(str(delay_scan))), None)
            if match is None:
                raise ValueError(f'Could not find FIR basis column for delay {delay_scan} in design matrix columns {list(fir_dm.columns)}')
            basis_columns.append(match)
        basis = fir_dm.loc[:, basis_columns].to_numpy(dtype=float)
        curve = basis @ betas.reshape(-1, 1)
        return curve
    raise ValueError(f'Unknown hrf_model {hrf_model!r}')

def build_basis_curve(epoch_times_s, hrf_model, fir_delays=None, fir_betas=None, stim_duration_s=None):
    return build_basis_curves(epoch_times_s, hrf_model, fir_delays=fir_delays, fir_betas=fir_betas, stim_duration_s=stim_duration_s)[:, 0]

def peak_value_and_time(signal, time_s, chromophore):
    if chromophore == 'hbo':
        idx = int(np.nanargmax(signal))
    else:
        idx = int(np.nanargmin(signal))
    return (float(signal[idx]), float(time_s[idx]))

def compute_shape_metrics(recovered_curve, truth_curve, epoch_times_s, chromophore):
    recovered_curve = np.asarray(recovered_curve, dtype=float)
    truth_curve = np.asarray(truth_curve, dtype=float)
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    metric_names = ['curve_corr', 'curve_rmse', 'curve_nrmse', 'peak_latency_error_s', 'peak_amplitude_bias', 'peak_amplitude_ratio', 'auc_bias', 'recovered_peak_amplitude', 'recovered_auc', 'truth_peak_amplitude', 'truth_auc']
    mask = np.isfinite(recovered_curve) & np.isfinite(truth_curve)
    if mask.sum() < 3:
        return {k: np.nan for k in metric_names}
    rec = recovered_curve[mask]
    tru = truth_curve[mask]
    t = epoch_times_s[mask]
    corr = np.nan if np.std(rec) < 1e-12 or np.std(tru) < 1e-12 else float(np.corrcoef(rec, tru)[0, 1])
    rmse = float(np.sqrt(np.mean((rec - tru) ** 2)))
    scale = float(np.max(tru) - np.min(tru))
    nrmse = float(rmse / scale) if scale > 1e-12 else np.nan
    rec_peak, rec_t = peak_value_and_time(rec, t, chromophore)
    tru_peak, tru_t = peak_value_and_time(tru, t, chromophore)
    peak_ratio = float(rec_peak / tru_peak) if abs(tru_peak) > 1e-12 else np.nan
    rec_auc = float(scipy.integrate.trapezoid(rec, t))
    tru_auc = float(scipy.integrate.trapezoid(tru, t))
    auc_bias = float(rec_auc - tru_auc)
    return {'curve_corr': corr, 'curve_rmse': rmse, 'curve_nrmse': nrmse, 'peak_latency_error_s': float(rec_t - tru_t), 'peak_amplitude_bias': float(rec_peak - tru_peak), 'peak_amplitude_ratio': peak_ratio, 'auc_bias': auc_bias, 'recovered_peak_amplitude': float(rec_peak), 'recovered_auc': rec_auc, 'truth_peak_amplitude': float(tru_peak), 'truth_auc': tru_auc}

def wavelet_motion_correct_array(signal, config):
    try:
        import pywt
    except Exception as exc:
        raise BenchmarkError('Wavelet motion-correction requires PyWavelets (`pywt`).') from exc
    signal = np.asarray(signal, dtype=float).reshape(-1)
    if signal.size < 8 or not np.isfinite(signal).any():
        return signal.copy()
    level = min(pywt.dwt_max_level(signal.size, pywt.Wavelet(config['wavelet_name']).dec_len), 6)
    if level <= 0:
        return signal.copy()
    coeffs = pywt.wavedec(signal, config['wavelet_name'], mode=config['wavelet_padding_mode'], level=level)
    corrected = [coeffs[0]]
    for detail in coeffs[1:]:
        detail = np.asarray(detail, dtype=float)
        if detail.size == 0:
            corrected.append(detail)
            continue
        q25, q75 = np.percentile(detail, [25, 75])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 0:
            corrected.append(detail)
            continue
        center = float(np.median(detail))
        lo = center - config['wavelet_iqr_multiplier'] * iqr
        hi = center + config['wavelet_iqr_multiplier'] * iqr
        filt = detail.copy()
        mask = (filt < lo) | (filt > hi)
        filt[mask] = center
        corrected.append(filt)
    reconstructed = pywt.waverec(corrected, config['wavelet_name'], mode=config['wavelet_padding_mode'])
    return np.asarray(reconstructed[:signal.size], dtype=float)

def apply_motion_correction_od(raw_od, motion_method, config):
    from mne.preprocessing.nirs import temporal_derivative_distribution_repair as tddr
    motion_method = str(motion_method).lower()
    if motion_method == 'tddr':
        return tddr(raw_od.copy())
    if motion_method == 'none':
        return raw_od.copy()
    if motion_method == 'wavelet':
        corrected = raw_od.copy()
        data = corrected.get_data()
        corrected_data = np.vstack([wavelet_motion_correct_array(data[idx], config) for idx in range(data.shape[0])])
        corrected._data = corrected_data
        return corrected
    raise ValueError(f'Unknown motion_method {motion_method!r}')

def apply_filter_mode(raw_obj, filter_mode, config):
    filter_mode = str(filter_mode).lower()
    filtered = raw_obj.copy()
    if filter_mode == 'none':
        return filtered
    if filter_mode == 'bandpass':
        return filtered.filter(config['filter_low_hz'], config['filter_high_hz'], verbose=False)
    if filter_mode == 'highpass_only':
        return filtered.filter(config['filter_highpass_only_hz'], None, verbose=False)
    raise ValueError(f'Unknown filter_mode {filter_mode!r}')

def preprocess_raw_to_hb(raw_cw, pair_quality, pipeline, config):
    from mne.preprocessing.nirs import optical_density, beer_lambert_law
    bad_pair_names = get_bad_pair_names(pair_quality, pipeline['pruning_style'], config)
    raw_od = optical_density(raw_cw.copy())
    processed_od = apply_motion_correction_od(raw_od, pipeline['motion_method'], config)
    processed_od = apply_filter_mode(processed_od, pipeline['filter_mode'], config)
    raw_hb = beer_lambert_law(processed_od, ppf=config['ppf_value'])
    raw_hb = apply_bad_pairs_to_hb(raw_hb, bad_pair_names)
    return (raw_hb, bad_pair_names)

def standardize_glm_dataframe(glm_df):
    glm_df = glm_df.copy()
    glm_df.columns = [str(col).strip().lower() for col in glm_df.columns]
    return glm_df

def find_first_matching_column(column_names, candidates):
    names = [str(c) for c in column_names]
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None

def get_task_regressor_names(design_matrix):
    names = []
    for col in design_matrix.columns:
        s = str(col)
        if s == 'constant' or s.startswith('drift') or s.startswith('ss_') or s.startswith('aux_') or s.startswith('reg_'):
            continue
        names.append(s)
    return names

def build_design_matrix(raw_hb, hrf_model, nuisance_df, config):
    make_first_level_design_matrix, _, _ = import_mne_nirs_modules()
    _, nilearn_make_dm = import_nilearn_functions()
    add_regs = None if nuisance_df is None else nuisance_df
    add_names = None if nuisance_df is None else list(nuisance_df.columns)
    events = get_annotation_events_df(raw_hb, config)
    if len(events) == 0:
        raise BenchmarkError('No valid annotations available to build the design matrix.')
    common_kwargs = dict(
        frame_times=raw_hb.times,
        events=events,
        drift_model='cosine',
        high_pass=config['drift_high_pass_hz'],
        add_regs=add_regs,
        add_reg_names=add_names,
    )
    if hrf_model == 'fir':
        # Build the FIR design directly from the event table so the model uses the
        # actual annotation durations. Nilearn's FIR basis remains convolutional and
        # uses the event durations, so hard-coding stim_dur=1.0 can mis-specify block
        # lengths for the semisynthetic benchmark.
        return nilearn_make_dm(
            **common_kwargs,
            hrf_model='fir',
            fir_delays=config['fir_delays_scans'],
            oversampling=1,
        )
    if hrf_model in {'canonical', 'canonical_derivs'}:
        # Placeholder design only used for task-name bookkeeping in the native MATLAB path.
        return pd.DataFrame({'task': np.ones(len(raw_hb.times), dtype=float)}, index=np.arange(len(raw_hb.times)))
    if hrf_model in {'glover', 'spm'}:
        return nilearn_make_dm(
            **common_kwargs,
            hrf_model=hrf_model,
            oversampling=50,
        )
    if hrf_model in {'spm_derivs', 'glover_derivs', 'gamma', 'gamma_derivs'}:
        return nilearn_make_dm(
            **common_kwargs,
            hrf_model=get_nilearn_hrf_model(hrf_model),
            oversampling=50,
        )
    raise ValueError(f'Unsupported hrf_model {hrf_model!r}')

def parse_mne_glm_to_channel_rows(glm_df, design_matrix, channel_names, subject, file_label, amplitude_value, pipeline, target_pair_names, nuisance_metadata_by_channel):
    glm_df = standardize_glm_dataframe(glm_df)
    ch_col = find_first_matching_column(glm_df.columns, ['ch_name', 'channel', 'name'])
    cond_col = find_first_matching_column(glm_df.columns, ['condition', 'cond', 'regressor', 'variable', 'name'])
    beta_col = find_first_matching_column(glm_df.columns, ['theta', 'beta', 'coef', 'estimate', 'effect'])
    t_col = find_first_matching_column(glm_df.columns, ['t', 't_value', 'tstat', 't_stat'])
    p_col = find_first_matching_column(glm_df.columns, ['p_value', 'pvalue', 'p'])
    if beta_col is None or cond_col is None:
        raise BenchmarkError('Could not parse GLM dataframe columns.')
    task_names = list(get_task_regressor_names(design_matrix))
    rows = []
    for _, row in glm_df.iterrows():
        condition = str(row[cond_col])
        matched_name = None
        for name in task_names:
            if condition == name:
                matched_name = name
                break
        if matched_name is None:
            for name in task_names:
                if condition.startswith(name):
                    matched_name = name
                    break
        if matched_name is None:
            continue
        task_order = task_names.index(matched_name) + 1
        channel_name = str(row[ch_col]) if ch_col is not None else None
        if channel_name is None or channel_name not in channel_names:
            continue
        pair_name = channel_name.split(' ')[0]
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        out = {
            'subject': subject,
            'file_label': file_label,
            'amplitude_value': amplitude_value,
            'pipeline_label': pipeline['label'],
            'backend': pipeline['backend'],
            'hrf_model': pipeline['hrf_model'],
            'solver': pipeline['solver'],
            'channel_name': channel_name,
            'pair_name': pair_name,
            'chromophore': chrom,
            'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target',
            'task_regressor': condition,
            'task_regressor_order': task_order,
            'is_primary_task_regressor': task_order == 1,
            'beta': float(row[beta_col]),
            **nuisance_metadata_by_channel.get(channel_name, {}),
        }
        if t_col is not None:
            out['t_value'] = float(row[t_col])
        if p_col is not None:
            out['p_value'] = float(row[p_col])
        rows.append(out)
    return pd.DataFrame(rows)

def resample_for_fir(raw_hb, config):
    raw_fir = raw_hb.copy().load_data()
    raw_fir.resample(config['fir_resample_sfreq_hz'], npad='auto')
    raw_fir = sanitize_annotations_to_single_task(raw_fir)
    return raw_fir

def make_epochs(raw_hb, config):
    events, event_id = mne.events_from_annotations(raw_hb, verbose=False)
    if len(events) == 0:
        raise BenchmarkError('No events available for epoching.')
    return mne.Epochs(raw_hb, events=events, event_id=event_id, tmin=config['epoch_tmin'], tmax=config['epoch_tmax'], baseline=config['baseline_window'], preload=True, detrend=None, reject_by_annotation=False, verbose=False)

def compute_block_average_channel_metrics(epochs_hb, channel_names, subject, file_label, amplitude_value, pipeline, target_pair_names, config):
    base_mask = (epochs_hb.times >= config['baseline_window'][0]) & (epochs_hb.times <= config['baseline_window'][1])
    resp_mask = (epochs_hb.times >= config['response_window'][0]) & (epochs_hb.times <= config['response_window'][1])
    metric_rows, shape_rows, roi_rows = ([], [], [])
    for channel_name in channel_names:
        if channel_name not in epochs_hb.ch_names:
            continue
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        pair_name = channel_name.split(' ')[0]
        channel_data = epochs_hb.copy().pick([channel_name]).get_data()[:, 0, :]
        score = channel_data[:, resp_mask].mean(axis=1) - channel_data[:, base_mask].mean(axis=1)
        mean_curve = channel_data.mean(axis=0)
        metric_rows.append({'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'hrf_model': 'block_average', 'solver': 'none', 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target', 'score': float(score.mean()), 'score_std': float(score.std())})
        shape_rows.append({'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target', 'shape_source': 'block_average_mean_epoch', **compute_shape_metrics(mean_curve, get_truth_curve({}, file_label, epochs_hb.times, chrom), epochs_hb.times, chrom)})
    metric_df = pd.DataFrame(metric_rows)
    if len(metric_df) > 0:
        for (chrom, target_status), group in metric_df.groupby(['chromophore', 'target_status']):
            names = group['channel_name'].tolist()
            mean_curve = epochs_hb.copy().pick(names).get_data().mean(axis=(0, 1))
            for t_s, y in zip(epochs_hb.times, mean_curve):
                roi_rows.append({'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': chrom, 'target_status': target_status, 'curve_source': 'block_average_mean_epoch', 'time_s': float(t_s), 'signal': float(y)})
    return (metric_df, pd.DataFrame(shape_rows), pd.DataFrame(roi_rows))

def parse_fir_dataframe(glm_df, design_matrix, channel_names, subject, file_label, amplitude_value, pipeline, target_pair_names, nuisance_metadata_by_channel, config):
    glm_df = standardize_glm_dataframe(glm_df)
    ch_col = find_first_matching_column(glm_df.columns, ['ch_name', 'channel', 'name'])
    cond_col = find_first_matching_column(glm_df.columns, ['condition', 'cond', 'regressor', 'variable', 'name'])
    beta_col = find_first_matching_column(glm_df.columns, ['theta', 'beta', 'coef', 'estimate', 'effect'])
    if beta_col is None or cond_col is None:
        raise BenchmarkError('Could not parse FIR GLM dataframe columns.')
    task_names = get_task_regressor_names(design_matrix)
    rows = []
    for _, row in glm_df.iterrows():
        condition = str(row[cond_col])
        if condition not in task_names and (not any((condition.startswith(name) for name in task_names))):
            continue
        try:
            delay_scan = float(condition.split('_')[-1])
        except Exception:
            continue
        channel_name = str(row[ch_col]) if ch_col is not None else None
        if channel_name is None or channel_name not in channel_names:
            continue
        pair_name = channel_name.split(' ')[0]
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        rows.append({'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'hrf_model': pipeline['hrf_model'], 'solver': pipeline['solver'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target', 'fir_regressor': condition, 'delay_scan': delay_scan, 'delay_s': delay_scan / float(config['fir_resample_sfreq_hz']), 'beta': float(row[beta_col]), **nuisance_metadata_by_channel.get(channel_name, {})})
    return pd.DataFrame(rows)

def summarize_fir_shapes(fir_df, truth_template_key, pipeline, truth_templates, config, stim_duration_s=None):
    epoch_times = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, 1.0 / config['fir_resample_sfreq_hz'])
    shape_rows, roi_rows = ([], [])
    for channel_name, group in fir_df.groupby('channel_name'):
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        pair_name = channel_name.split(' ')[0]
        group = group.sort_values('delay_s')
        recovered = build_basis_curve(epoch_times, 'fir', fir_delays=group['delay_s'].tolist(), fir_betas=group['beta'].to_numpy(), stim_duration_s=stim_duration_s)
        truth_curve = get_truth_curve(truth_templates, truth_template_key, epoch_times, chrom)
        shape_rows.append({'subject': group['subject'].iloc[0], 'file_label': group['file_label'].iloc[0], 'amplitude_value': int(group['amplitude_value'].iloc[0]), 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': group['target_status'].iloc[0], 'shape_source': 'fir_beta_curve', **compute_shape_metrics(recovered, truth_curve, epoch_times, chrom)})
    for (chrom, target_status), group in fir_df.groupby(['chromophore', 'target_status']):
        curve = group.groupby('delay_s', as_index=False)['beta'].mean().sort_values('delay_s')
        roi_curve = build_basis_curve(epoch_times, 'fir', fir_delays=curve['delay_s'].tolist(), fir_betas=curve['beta'].to_numpy(), stim_duration_s=stim_duration_s)
        for t_s, y in zip(epoch_times, roi_curve):
            roi_rows.append({'subject': group['subject'].iloc[0], 'file_label': group['file_label'].iloc[0], 'amplitude_value': int(group['amplitude_value'].iloc[0]), 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': chrom, 'target_status': target_status, 'curve_source': 'fir_roi_mean_curve', 'time_s': float(t_s), 'signal': float(y)})
    return (shape_rows, roi_rows)

def write_matlab_bundle(observed_specs, shift_specs, job_dir):
    specs = [spec for spec in observed_specs + shift_specs if spec is not None]
    if not specs:
        return None
    in_dir = ensure_dir(job_dir / 'matlab_inputs')
    bundle = {'input_mat_files': [], 'output_csv_files': []}
    for idx, spec in enumerate(specs, start=1):
        pipeline_label = spec['pipeline_label']
        shift_index = int(spec.get('shift_index', 0))
        suffix = f'{pipeline_label}__shift{shift_index:03d}' if shift_index > 0 else f'{pipeline_label}__observed'
        in_path = in_dir / f'{idx:04d}__{suffix}__input.mat'
        out_path = in_dir / f'{idx:04d}__{suffix}__output.csv'
        mat_payload = {
            'snirf_file': np.asarray([str(spec['snirf_file'])], dtype=object),
            'subject': np.asarray([spec['subject']], dtype=object),
            'file_label': np.asarray([spec['file_label']], dtype=object),
            'pipeline_label': np.asarray([spec['pipeline_label']], dtype=object),
            'backend': np.asarray([spec['backend']], dtype=object),
            'hrf_model': np.asarray([spec['hrf_model']], dtype=object),
            'solver': np.asarray([spec['solver']], dtype=object),
            'amplitude_value': np.asarray([[spec['amplitude_value']]], dtype=np.int32),
            'shift_index': np.asarray([[spec['shift_index']]], dtype=np.int32),
            'shift_s': np.asarray([[spec['shift_s']]], dtype=float),
            'bad_pair_names': np.asarray(spec.get('bad_pair_names', []), dtype=object),
            'short_pair_names': np.asarray(spec.get('short_pair_names', []), dtype=object),
            'long_pair_names': np.asarray(spec.get('long_pair_names', []), dtype=object),
            'target_pair_names': np.asarray(spec['target_pair_names'], dtype=object),
            'analyzir_resample_fs_hz': np.asarray([[spec['analyzir_resample_fs_hz']]], dtype=float),
            'analyzir_use_tddr': np.asarray([[1 if spec.get('analyzir_use_tddr', False) else 0]], dtype=np.int32),
            'ppf_value': np.asarray([[float(spec.get('ppf_value', 6.0))]], dtype=float),
            'hrf_export_fs_hz': np.asarray([[float(spec.get('hrf_export_fs_hz', spec.get('analyzir_resample_fs_hz', 1.0)))]], dtype=float),
            'short_separation_threshold_m': np.asarray([[spec['short_separation_threshold_m']]], dtype=float),
            'stim_onset_s': np.asarray(spec.get('stim_onset_s', []), dtype=float),
            'stim_dur_s': np.asarray(spec.get('stim_dur_s', []), dtype=float),
            'stim_amp': np.asarray(spec.get('stim_amp', []), dtype=float),
        }
        scipy.io.savemat(in_path, mat_payload, do_compression=True)
        bundle['input_mat_files'].append(str(in_path))
        bundle['output_csv_files'].append(str(out_path))
    bundle_path = in_dir / 'matlab_bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2), encoding='utf-8')
    return bundle_path

def load_matlab_outputs_from_bundle(bundle_json_path):
    bundle = json.loads(bundle_json_path.read_text(encoding='utf-8'))
    tables = []
    for csv_path_str in bundle['output_csv_files']:
        csv_path = Path(csv_path_str)
        if csv_path.exists():
            tables.append(pd.read_csv(csv_path))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

def run_matlab_sidecar_batch(config, bundle_json_path, job_dir, helper_m):
    bundle = json.loads(bundle_json_path.read_text(encoding='utf-8'))
    helper_dir = helper_m.parent.as_posix()
    helper_name = helper_m.stem
    command_base = [config["matlab_cmd"]]
    if config.get("matlab_startup_options"):
        command_base.extend(str(config["matlab_startup_options"]).split())

    all_tables = []
    log_chunks = []
    for i, (in_path, out_path) in enumerate(zip(bundle['input_mat_files'], bundle['output_csv_files']), start=1):
        single_bundle_path = Path(in_path).with_suffix('.bundle.json')
        single_bundle = {'input_mat_files': [in_path], 'output_csv_files': [out_path]}
        single_bundle_path.write_text(json.dumps(single_bundle, indent=2), encoding='utf-8')

        env = os.environ.copy()
        env['FNIRS_BUNDLE_JSON'] = str(single_bundle_path)
        if config['analyzir_path']:
            env['FNIRS_ANALYZIR_PATH'] = str(config['analyzir_path'])

        command = command_base + [
            '-batch',
            f"try, addpath('{helper_dir}'); {helper_name}; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=config['matlab_timeout_s'])
        log_chunks.append(
            f"SPEC {i}\n{'='*80}\nINPUT: {in_path}\nOUTPUT: {out_path}\nCOMMAND\n{'='*80}\n{' '.join(command)}\n\nRETURN_CODE\n{'='*80}\n{result.returncode}\n\nSTDOUT\n{'='*80}\n{result.stdout}\n\nSTDERR\n{'='*80}\n{result.stderr}\n\n"
        )
        if result.returncode != 0:
            (job_dir / 'matlab_batch.log').write_text('\n'.join(log_chunks), encoding='utf-8')
            raise BenchmarkError(f"MATLAB sidecar failed; see {job_dir / 'matlab_batch.log'}")
        if Path(out_path).exists():
            all_tables.append(pd.read_csv(out_path))

    (job_dir / 'matlab_batch.log').write_text('\n'.join(log_chunks), encoding='utf-8')
    return pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()

def run_matlab_sidecar_engine(config, bundle_json_path, job_dir, helper_m):
    # Use the batch path for native AnalyzIR repair to avoid long-lived MATLAB engine sessions
    # retaining memory across multiple per-file pipeline specs.
    return run_matlab_sidecar_batch(config, bundle_json_path, job_dir, helper_m)

def run_matlab_sidecar(config, bundle_json_path, job_dir):
    if not config['use_matlab']:
        return pd.DataFrame()
    helper_m = Path(__file__).with_name('analyzir_glm_native_batch_v7.m')
    if not helper_m.exists():
        raise BenchmarkError(f'Missing MATLAB helper: {helper_m}')
    return run_matlab_sidecar_batch(config, bundle_json_path, job_dir, helper_m)

def run_homer3_sidecar(config, spec_json_path, job_dir):
    helper_m = Path(__file__).with_name('homer3_tcca_glm_batch_repaired_v2.m')
    if not helper_m.exists():
        raise BenchmarkError(f'Missing Homer3 MATLAB helper: {helper_m}')
    env = os.environ.copy()
    env['FNIRS_HOMER3_SPEC_JSON'] = str(spec_json_path)
    if config.get('homer3_path'):
        env['FNIRS_HOMER3_PATH'] = str(config['homer3_path'])
    helper_dir = helper_m.parent.as_posix()
    helper_name = helper_m.stem
    command = [config['matlab_cmd']]
    if config.get('matlab_startup_options'):
        command.extend(str(config['matlab_startup_options']).split())
    command.extend([
        '-batch',
        f"try, addpath('{helper_dir}'); {helper_name}; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);",
    ])
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=config['matlab_timeout_s'])
    (job_dir / 'homer3_batch.log').write_text(
        f"COMMAND\n{'=' * 80}\n{' '.join(command)}\n\nRETURN_CODE\n{'=' * 80}\n{result.returncode}\n\nSTDOUT\n{'=' * 80}\n{result.stdout}\n\nSTDERR\n{'=' * 80}\n{result.stderr}\n",
        encoding='utf-8',
    )
    if result.returncode != 0:
        raise BenchmarkError(f"Homer3 sidecar failed; see {job_dir / 'homer3_batch.log'}")


def run_homer3_pipeline(subject, file_spec, pipeline, snirf_file_path, raw_cw, target_pair_names, truth_templates, config):
    clear_job_runtime_caches()
    job_dir = ensure_dir(jobs_path(config) / subject / file_spec['label'])
    input_dir = ensure_dir(job_dir / 'homer3_inputs')
    resting_snirf_path = snirf_file_path.parent / 'resting_clean.snirf'
    if not resting_snirf_path.exists():
        raise BenchmarkError(f'Missing resting_clean.snirf required for Homer3 tCCA training: {resting_snirf_path}')

    output_roi_csv_path = input_dir / f"{pipeline['label']}__roi_timecourses.csv"
    output_channel_csv_path = input_dir / f"{pipeline['label']}__channel_hrf.csv"
    output_beta_csv_path = input_dir / f"{pipeline['label']}__beta.csv"
    output_metadata_csv_path = input_dir / f"{pipeline['label']}__glm_metadata.csv"

    stim_onsets_s = np.asarray(raw_cw.annotations.onset - raw_cw.first_time, dtype=float)
    stim_durations_s = np.asarray(raw_cw.annotations.duration, dtype=float)
    stim_descriptions = [str(desc) for desc in raw_cw.annotations.description]

    spec = {
        'active_snirf_file': str(snirf_file_path),
        'resting_snirf_file': str(resting_snirf_path),
        'output_roi_csv_file': str(output_roi_csv_path),
        'output_channel_csv_file': str(output_channel_csv_path),
        'output_beta_csv_file': str(output_beta_csv_path),
        'output_metadata_csv_file': str(output_metadata_csv_path),
        'subject': subject,
        'file_label': file_spec['label'],
        'amplitude_value': int(file_spec['amplitude_value']),
        'pipeline_label': pipeline['label'],
        'backend': pipeline['backend'],
        'hrf_model': pipeline['hrf_model'],
        'solver': pipeline['solver'],
        'target_pair_names': sorted(list(target_pair_names)),
        'stim_onsets_s': stim_onsets_s.tolist(),
        'stim_durations_s': stim_durations_s.tolist(),
        'stim_descriptions': stim_descriptions,
        'short_separation_threshold_mm': float(config['short_separation_threshold_m'] * 1000.0),
        'long_separation_threshold_mm': float(config['long_separation_threshold_m'] * 1000.0),
        'tcca_params': [float(x) for x in config.get('homer3_tcca_params', HOMER3_TCCA_PARAMS)],
        'tcca_rest_window_s': [float(x) for x in config.get('homer3_tcca_rest_window_s', HOMER3_TCCA_REST_WINDOW_S)],
        'glm_trange_s': [float(x) for x in config.get('homer3_glm_trange_s', HOMER3_GLM_TRANGE_S)],
        'lowpass_hz': float(config.get('homer3_lowpass_hz', HOMER3_LOWPASS_HZ)),
        'aux_label_allowlist': [str(x) for x in config.get('homer3_aux_label_allowlist', HOMER3_AUX_LABEL_ALLOWLIST)],
        'ss_channel_selection': [float(x) for x in config.get('homer3_ss_channel_selection', HOMER3_SS_CHANNEL_SELECTION)],
        'ppf': [float(config.get('ppf_value', 6.0)), float(config.get('ppf_value', 6.0))],
    }
    spec_json_path = input_dir / f"{pipeline['label']}__spec.json"
    spec_json_path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    run_homer3_sidecar(config, spec_json_path, job_dir)

    def _read_optional_csv(csv_path):
        if not csv_path.exists():
            return pd.DataFrame()
        try:
            if csv_path.stat().st_size <= 1:
                return pd.DataFrame()
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    channel_hrf_df = _read_optional_csv(output_channel_csv_path)
    beta_df = _read_optional_csv(output_beta_csv_path)
    metadata_df = _read_optional_csv(output_metadata_csv_path)

    homer3_note = 'Native Homer3 tCCA + GLM pipeline'
    if len(channel_hrf_df) == 0:
        homer3_note = 'Native Homer3 tCCA + GLM pipeline; no channel-level HRF CSV produced (treated as no detected signal)'

    shape_df = pd.DataFrame()
    roi_df = pd.DataFrame()
    availability_row = None
    if len(channel_hrf_df) > 0:
        shape_rows = collect_shape_metrics_from_timecourse_table(
            channel_hrf_df,
            file_spec.get('truth_template_key'),
            truth_templates,
            shape_source='homer3_channel_hrf',
        )
        roi_rows = collect_roi_timecourses_from_timecourse_table(
            channel_hrf_df,
            curve_source='homer3_roi_mean_channel_hrf',
        )
        shape_df = pd.DataFrame(shape_rows)
        roi_df = pd.DataFrame(roi_rows)
        availability_row = build_channel_availability_row_from_timecourse_table(
            subject,
            file_spec,
            pipeline,
            channel_hrf_df,
            sorted(list(target_pair_names)),
            [],
        )

    nuisance_detail_payload = {
        'subject': subject,
        'file_label': file_spec['label'],
        'pipeline_label': pipeline['label'],
        'backend': pipeline['backend'],
        'nuisance_method_used': 'homer3_tcca',
        'notes': homer3_note,
    }
    if len(metadata_df) > 0:
        meta_row = metadata_df.iloc[0].to_dict()
        nuisance_detail_payload.update({
            'homer3_idx_basis': meta_row.get('idx_basis', np.nan),
            'homer3_params_basis_json': meta_row.get('params_basis_json', ''),
            'stim_duration_s_effective': meta_row.get('stim_duration_s_effective', np.nan),
        })
    nuisance_detail_df = pd.DataFrame([nuisance_detail_payload])

    return {
        'canonical_channel_metrics': pd.DataFrame(),
        'block_average_channel_metrics': pd.DataFrame(),
        'fir_channel_metrics': pd.DataFrame(),
        'shape_metrics': shape_df,
        'roi_timecourses': roi_df,
        'nuisance_detail': nuisance_detail_df,
        'matlab_input_specs_list': [],
        'matlab_shift_specs_list': [],
        'homer3_channel_hrf': channel_hrf_df,
        'homer3_beta': beta_df,
        'homer3_glm_metadata': metadata_df,
        'channel_availability_row': availability_row,
    }


def make_shifted_annotations(source_annotations, total_duration_s, shift_s):
    shifted_onsets, shifted_durations, shifted_descriptions = ([], [], [])
    for onset_s, duration_s, description in zip(source_annotations.onset, source_annotations.duration, source_annotations.description):
        shifted_onset = float(onset_s) + float(shift_s)
        while shifted_onset >= total_duration_s:
            shifted_onset -= total_duration_s
        shifted_onsets.append(shifted_onset)
        shifted_durations.append(float(duration_s))
        shifted_descriptions.append(str(description))
    order = np.argsort(shifted_onsets)
    return mne.Annotations(onset=np.asarray(shifted_onsets)[order], duration=np.asarray(shifted_durations)[order], description=np.asarray(shifted_descriptions)[order].tolist())

def get_shift_values(total_duration_s, shift_count, min_shift_s):
    if shift_count <= 0:
        return []
    max_shift_s = max(min_shift_s + 1.0, total_duration_s - min_shift_s)
    values = np.linspace(min_shift_s, max_shift_s, shift_count + 2)[1:-1]
    return [float(v) for v in values]

def summarize_target_minus_non_target(df, value_column, group_columns=None, score_type=None):
    if df is None or len(df) == 0 or value_column not in df.columns:
        return pd.DataFrame()
    group_columns = list(group_columns or ['chromophore'])
    grouped = df.groupby([*group_columns, 'target_status'], as_index=False).agg(mean_score=(value_column, 'mean'), n_channels=(value_column, 'count'))
    wide = grouped.pivot_table(index=group_columns, columns='target_status', values='mean_score').reset_index().rename(columns={'true_target': 'mean_target_score', 'true_non_target': 'mean_non_target_score'})
    counts = grouped.groupby(group_columns, as_index=False).agg(n_channels=('n_channels', 'sum'))
    wide = wide.merge(counts, on=group_columns, how='left')
    if 'mean_target_score' not in wide.columns:
        wide['mean_target_score'] = np.nan
    if 'mean_non_target_score' not in wide.columns:
        wide['mean_non_target_score'] = np.nan
    wide['target_minus_non_target_score'] = wide['mean_target_score'] - wide['mean_non_target_score']
    if score_type is not None:
        wide['score_type'] = score_type
    return wide

def build_global_nuisance_regressors(raw_hb, snirf_file_path, raw_cw, nuisance_method, config):
    """
    Build one nuisance regressor set per chromophore.
    """
    cache = runtime_cache_bucket('global_nuisance')
    cache_key = (id(raw_hb), str(snirf_file_path), nuisance_method, config['pooled_ss_n_components'], config['ss_aux_n_components'])
    if cache_key in cache:
        return cache[cache_key]

    nuisance_by_chromophore = {}
    auxiliary_df = read_auxiliary_dataframe(snirf_file_path, raw_cw)
    auxiliary_matrix = auxiliary_df.to_numpy().T if len(auxiliary_df.columns) > 0 else None
    raw_data = _get_raw_array(raw_hb)
    ch_index_map = _get_channel_index_map(raw_hb)
    for chromophore in ['hbo', 'hbr']:
        if nuisance_method == 'none':
            nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none'}}
            continue
        short_channel_names = get_available_short_channel_names(raw_hb, chromophore, config)
        short_indices = [ch_index_map[name] for name in short_channel_names]
        short_channel_matrix = raw_data[short_indices, :] if short_indices else None
        if nuisance_method == 'pooled_pca2':
            if short_channel_matrix is None or short_channel_matrix.size == 0:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_no_short_channels'}}
                continue
            nuisance_components = principal_components_rows(short_channel_matrix, config['pooled_ss_n_components'])
            if nuisance_components is None:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_failed_pca'}}
                continue
            nuisance_df = pd.DataFrame(nuisance_components, columns=[f'ss_pc{i + 1}_{chromophore}' for i in range(nuisance_components.shape[1])])
            nuisance_by_chromophore[chromophore] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'pooled_pca2', 'n_short_channels_used': len(short_channel_names), 'n_nuisance_components': nuisance_components.shape[1]}}
            continue
        if nuisance_method == 'ss_aux_pca':
            stacked_blocks = []
            if short_channel_matrix is not None and short_channel_matrix.size > 0:
                stacked_blocks.append(short_channel_matrix)
            if auxiliary_matrix is not None:
                stacked_blocks.append(auxiliary_matrix)
            if not stacked_blocks:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_no_short_or_aux'}}
                continue
            nuisance_components = principal_components_rows(np.vstack(stacked_blocks), config['ss_aux_n_components'])
            if nuisance_components is None:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_failed_pca'}}
                continue
            nuisance_df = pd.DataFrame(nuisance_components, columns=[f'aux_pc{i + 1}_{chromophore}' for i in range(nuisance_components.shape[1])])
            nuisance_by_chromophore[chromophore] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'ss_aux_pca', 'n_short_channels_used': len(short_channel_names), 'n_aux_channels_used': 0 if auxiliary_matrix is None else auxiliary_matrix.shape[0], 'aux_channel_names': '|'.join(auxiliary_df.columns.astype(str).tolist()) if len(auxiliary_df.columns) > 0 else '', 'n_nuisance_components': nuisance_components.shape[1]}}
            continue
        raise ValueError(f'Unsupported global nuisance_method {nuisance_method!r}')
    cache[cache_key] = nuisance_by_chromophore
    return nuisance_by_chromophore

def build_channel_nuisance_regressors(raw_hb, nuisance_method, config):
    """
    Build one nuisance regressor set per long-separation channel.
    The expensive geometry/data lookups are cached once per raw object.
    """
    cache = runtime_cache_bucket('channel_nuisance')
    cache_key = (id(raw_hb), nuisance_method, config['local_ss_max_distance_m'], config['multi_ss_k'])
    if cache_key in cache:
        return cache[cache_key]

    nuisance_by_channel = {}
    long_channel_names = get_available_long_channel_names(raw_hb, config)
    raw_data = _get_raw_array(raw_hb)
    ch_index_map = _get_channel_index_map(raw_hb)
    position_map = _get_channel_position_map(raw_hb)

    short_names_by_chrom = {}
    short_indices_by_chrom = {}
    short_positions_by_chrom = {}
    short_data_by_chrom = {}
    for chromophore in ['hbo', 'hbr']:
        short_channel_names = get_available_short_channel_names(raw_hb, chromophore, config)
        short_names_by_chrom[chromophore] = short_channel_names
        short_indices = [ch_index_map[name] for name in short_channel_names]
        short_indices_by_chrom[chromophore] = short_indices
        short_positions_by_chrom[chromophore] = np.vstack([position_map[name] for name in short_channel_names]) if short_channel_names else np.empty((0, 3), dtype=float)
        short_data_by_chrom[chromophore] = raw_data[short_indices, :] if short_indices else np.empty((0, raw_data.shape[1]), dtype=float)

    for long_channel_name in long_channel_names:
        chromophore = 'hbo' if long_channel_name.endswith('hbo') else 'hbr'
        if nuisance_method in {'none', 'native_ss_regressors', 'native_ss_filter'}:
            nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': nuisance_method}}
            continue

        short_channel_names = short_names_by_chrom[chromophore]
        if not short_channel_names:
            nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_no_short_channels'}}
            continue

        long_channel_position = position_map[long_channel_name]
        short_positions = short_positions_by_chrom[chromophore]
        distances = np.linalg.norm(short_positions - long_channel_position.reshape(1, 3), axis=1)
        order = np.argsort(distances)

        if nuisance_method == 'local_nearest':
            nearest_idx = int(order[0])
            nearest_short_name = short_channel_names[nearest_idx]
            nearest_short_distance_m = float(distances[nearest_idx])
            if nearest_short_distance_m <= config['local_ss_max_distance_m']:
                nuisance_trace = short_data_by_chrom[chromophore][nearest_idx]
                nuisance_df = pd.DataFrame({f'ss_local_{chromophore}': nuisance_trace})
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'nearest_short_channel', 'nuisance_regressor_label': nearest_short_name, 'nearest_short_distance_m': nearest_short_distance_m, 'n_short_channels_used': 1}}
            else:
                fallback_trace = short_data_by_chrom[chromophore].mean(axis=0)
                nuisance_df = pd.DataFrame({f'ss_pooled_fallback_{chromophore}': fallback_trace})
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'pooled_fallback_average', 'nuisance_regressor_label': 'pooled_fallback_average', 'nearest_short_distance_m': nearest_short_distance_m, 'n_short_channels_used': len(short_channel_names)}}
            continue

        if nuisance_method == 'multi_ss_orth3':
            take_n = max(1, min(config['multi_ss_k'], len(short_channel_names)))
            selected_idx = order[:take_n]
            selected_short_names = [short_channel_names[int(i)] for i in selected_idx]
            selected_short_matrix = short_data_by_chrom[chromophore][selected_idx, :].T
            selected_short_matrix = (selected_short_matrix - selected_short_matrix.mean(axis=0, keepdims=True)) / np.maximum(selected_short_matrix.std(axis=0, keepdims=True), 1e-12)
            orthogonal_components = qr_orth_columns(selected_short_matrix)
            if orthogonal_components.size == 0:
                nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_failed_qr'}}
            else:
                nuisance_df = pd.DataFrame(orthogonal_components, columns=[f'ss_qr{i + 1}_{chromophore}' for i in range(orthogonal_components.shape[1])])
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'multi_ss_orth3', 'n_short_channels_used': len(selected_short_names), 'n_nuisance_components': orthogonal_components.shape[1], 'nearest_short_distance_m': float(distances[int(selected_idx[0])]), 'short_regressor_labels': '|'.join(selected_short_names)}}
            continue

        raise ValueError(f'Unsupported channel-specific nuisance_method {nuisance_method!r}')

    cache[cache_key] = nuisance_by_channel
    return nuisance_by_channel

def apply_channelwise_nuisance_regression(raw_hb, nuisance_by_channel):
    """
    Time-domain nuisance regression for the block-average branch.
    Uses direct array indexing rather than repeated Raw.copy().pick() calls.
    """
    denoised_raw_hb = raw_hb.copy().load_data()
    nuisance_detail_rows = []
    ch_index_map = _get_channel_index_map(denoised_raw_hb)
    for channel_name, nuisance_info in nuisance_by_channel.items():
        nuisance_df = nuisance_info['design_df']
        nuisance_metadata = nuisance_info['metadata']
        chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        nuisance_detail_rows.append({'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
        if nuisance_df is None or nuisance_df.shape[1] == 0:
            continue
        channel_idx = ch_index_map[channel_name]
        signal_vector = denoised_raw_hb._data[channel_idx, :]
        design_matrix = np.column_stack([np.ones(signal_vector.shape[0], dtype=float), nuisance_df.to_numpy()])
        regression_coefficients = np.linalg.lstsq(design_matrix, signal_vector, rcond=None)[0]
        fitted_signal = design_matrix @ regression_coefficients
        denoised_raw_hb._data[channel_idx, :] = signal_vector - fitted_signal + regression_coefficients[0]
    return (denoised_raw_hb, pd.DataFrame(nuisance_detail_rows))

def reconstruct_curve_from_channel_group(channel_group_df, epoch_times_s, pipeline, stim_duration_s=None):
    basis_curves = build_basis_curves(epoch_times_s, pipeline['hrf_model'], stim_duration_s=stim_duration_s)
    ordered = channel_group_df.sort_values('task_regressor_order') if 'task_regressor_order' in channel_group_df.columns else channel_group_df
    betas = ordered['beta'].to_numpy(dtype=float)
    n_components = min(len(betas), basis_curves.shape[1])
    if n_components <= 0:
        return np.zeros_like(epoch_times_s, dtype=float)
    return basis_curves[:, :n_components] @ betas[:n_components]

def collect_canonical_shape_metrics(channel_metrics_df, epoch_times_s, truth_template_key, pipeline, truth_templates, stim_duration_s=None):
    if channel_metrics_df is None or len(channel_metrics_df) == 0:
        return []
    shape_metric_rows = []
    for channel_name, channel_group_df in channel_metrics_df.groupby('channel_name'):
        metric_row = channel_group_df.iloc[0]
        chromophore = metric_row['chromophore']
        recovered_curve = reconstruct_curve_from_channel_group(channel_group_df, epoch_times_s, pipeline, stim_duration_s=stim_duration_s)
        truth_curve = get_truth_curve(truth_templates, truth_template_key, epoch_times_s, chromophore)
        shape_metric_rows.append({
            'subject': metric_row['subject'],
            'file_label': metric_row['file_label'],
            'amplitude_value': metric_row['amplitude_value'],
            'pipeline_label': metric_row['pipeline_label'],
            'backend': metric_row['backend'],
            'channel_name': metric_row['channel_name'],
            'pair_name': metric_row['pair_name'],
            'chromophore': chromophore,
            'target_status': metric_row['target_status'],
            'shape_source': f"basis_combination_{pipeline['hrf_model']}",
            **compute_shape_metrics(recovered_curve, truth_curve, epoch_times_s, chromophore),
        })
    return shape_metric_rows

def collect_canonical_roi_timecourses(channel_metrics_df, epoch_times_s, pipeline, stim_duration_s=None):
    if channel_metrics_df is None or len(channel_metrics_df) == 0:
        return []
    roi_timecourse_rows = []
    for (chromophore, target_status), group_df in channel_metrics_df.groupby(['chromophore', 'target_status']):
        curves = []
        for _, channel_group_df in group_df.groupby('channel_name'):
            curves.append(reconstruct_curve_from_channel_group(channel_group_df, epoch_times_s, pipeline, stim_duration_s=stim_duration_s))
        if not curves:
            continue
        roi_curve = np.nanmean(np.vstack(curves), axis=0)
        for time_s, signal_value in zip(epoch_times_s, roi_curve):
            roi_timecourse_rows.append({
                'subject': group_df['subject'].iloc[0],
                'file_label': group_df['file_label'].iloc[0],
                'amplitude_value': int(group_df['amplitude_value'].iloc[0]),
                'pipeline_label': pipeline['label'],
                'backend': pipeline['backend'],
                'chromophore': chromophore,
                'target_status': target_status,
                'curve_source': f"roi_mean_basis_combination_{pipeline['hrf_model']}",
                'time_s': float(time_s),
                'signal': float(signal_value),
            })
    return roi_timecourse_rows

def collect_shape_metrics_from_timecourse_table(channel_timecourse_df, truth_template_key, truth_templates, shape_source):
    if channel_timecourse_df is None or len(channel_timecourse_df) == 0:
        return []
    shape_rows = []
    group_cols = ['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'channel_name', 'pair_name', 'chromophore', 'target_status']
    for group_key, group_df in channel_timecourse_df.groupby(group_cols):
        group_df = group_df.sort_values('time_s')
        subject, file_label, amplitude_value, pipeline_label, backend, channel_name, pair_name, chromophore, target_status = group_key
        time_s = group_df['time_s'].to_numpy(dtype=float)
        recovered_curve = group_df['signal'].to_numpy(dtype=float)
        truth_curve = get_truth_curve(truth_templates, truth_template_key, time_s, chromophore)
        shape_rows.append({
            'subject': subject,
            'file_label': file_label,
            'amplitude_value': amplitude_value,
            'pipeline_label': pipeline_label,
            'backend': backend,
            'channel_name': channel_name,
            'pair_name': pair_name,
            'chromophore': chromophore,
            'target_status': target_status,
            'shape_source': shape_source,
            **compute_shape_metrics(recovered_curve, truth_curve, time_s, chromophore),
        })
    return shape_rows


def collect_roi_timecourses_from_timecourse_table(channel_timecourse_df, curve_source):
    if channel_timecourse_df is None or len(channel_timecourse_df) == 0:
        return []
    roi_rows = []
    group_cols = ['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status', 'time_s']
    grouped = channel_timecourse_df.groupby(group_cols, as_index=False)['signal'].mean()
    for _, row in grouped.iterrows():
        roi_rows.append({
            'subject': row['subject'],
            'file_label': row['file_label'],
            'amplitude_value': int(row['amplitude_value']),
            'pipeline_label': row['pipeline_label'],
            'backend': row['backend'],
            'chromophore': row['chromophore'],
            'target_status': row['target_status'],
            'curve_source': curve_source,
            'time_s': float(row['time_s']),
            'signal': float(row['signal']),
        })
    return roi_rows


def build_channel_availability_row_from_timecourse_table(subject, file_spec, pipeline, channel_timecourse_df, target_pair_names, non_target_pair_names):
    if channel_timecourse_df is None or len(channel_timecourse_df) == 0:
        return None
    channel_names = sorted(set(channel_timecourse_df['channel_name'].dropna().astype(str)))
    pair_names = {name.split(' ')[0] for name in channel_names}
    target_pair_names = list(target_pair_names or [])
    non_target_pair_names = list(non_target_pair_names or [])
    available_target_pairs = sorted(pair_names & set(target_pair_names))
    if non_target_pair_names:
        available_non_target_pairs = sorted(pair_names & set(non_target_pair_names))
        n_total_non_target_pairs = len(non_target_pair_names)
    else:
        available_non_target_pairs = sorted(pair_names - set(target_pair_names))
        n_total_non_target_pairs = np.nan
    return {
        'subject': subject,
        'file_label': file_spec['label'],
        'amplitude_value': file_spec['amplitude_value'],
        'pipeline_label': pipeline['label'],
        'backend': pipeline['backend'],
        'n_total_true_target_pairs': len(target_pair_names),
        'n_available_true_target_pairs': len(available_target_pairs),
        'n_total_true_non_target_pairs': n_total_non_target_pairs,
        'n_available_true_non_target_pairs': len(available_non_target_pairs),
        'n_available_long_channels': len(channel_names),
        'target_pair_retention_fraction': len(available_target_pairs) / len(target_pair_names) if target_pair_names else np.nan,
        'non_target_pair_retention_fraction': len(available_non_target_pairs) / len(non_target_pair_names) if non_target_pair_names else np.nan,
        'n_bad_pairs': np.nan,
    }


def run_block_average_pipeline(subject, file_spec, pipeline, raw_hb, target_pair_names, truth_templates, config):
    """
    Block-average branch:
      1. build channel-specific nuisance regressors
      2. regress them out in the time domain
      3. epoch and average
      4. compute channel, shape, and ROI summaries
    """
    channel_nuisance = build_channel_nuisance_regressors(raw_hb, pipeline['nuisance_method'], config)
    denoised_raw_hb, nuisance_detail_df = apply_channelwise_nuisance_regression(raw_hb, channel_nuisance)
    epochs_hb = make_epochs(denoised_raw_hb, config)
    long_channel_names = get_available_long_channel_names(raw_hb, config)
    block_metric_df, shape_metric_df, roi_timecourse_df = compute_block_average_channel_metrics(epochs_hb, long_channel_names, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, config)
    if len(shape_metric_df) > 0:
        corrected_shape_rows = []
        columns_to_drop = ['curve_corr', 'curve_rmse', 'curve_nrmse', 'peak_latency_error_s', 'peak_amplitude_bias', 'peak_amplitude_ratio', 'auc_bias', 'recovered_peak_amplitude', 'recovered_auc', 'truth_peak_amplitude', 'truth_auc']
        for _, shape_row in shape_metric_df.iterrows():
            channel_name = shape_row['channel_name']
            chromophore = shape_row['chromophore']
            mean_epoch_curve = epochs_hb.copy().pick([channel_name]).get_data()[:, 0, :].mean(axis=0)
            truth_curve = get_truth_curve(truth_templates, file_spec.get('truth_template_key'), epochs_hb.times, chromophore)
            corrected_shape_rows.append({**shape_row.drop(labels=[c for c in columns_to_drop if c in shape_row.index]).to_dict(), **compute_shape_metrics(mean_epoch_curve, truth_curve, epochs_hb.times, chromophore)})
        shape_metric_df = pd.DataFrame(corrected_shape_rows)
    if len(nuisance_detail_df) > 0:
        nuisance_detail_df = nuisance_detail_df.copy()
        nuisance_detail_df['subject'] = subject
        nuisance_detail_df['file_label'] = file_spec['label']
        nuisance_detail_df['pipeline_label'] = pipeline['label']
    return {'canonical_channel_metrics': pd.DataFrame(), 'block_average_channel_metrics': block_metric_df, 'fir_channel_metrics': pd.DataFrame(), 'shape_metrics': shape_metric_df, 'roi_timecourses': roi_timecourse_df, 'nuisance_detail': nuisance_detail_df, 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}

def resample_nuisance_dataframe_to_fir_times(nuisance_df, original_times_s, fir_times_s):
    if nuisance_df is None:
        return None
    resampled_columns = {}
    for column_name in nuisance_df.columns:
        resampled_columns[column_name] = np.interp(fir_times_s, original_times_s, nuisance_df[column_name].to_numpy())
    return pd.DataFrame(resampled_columns)

def run_python_glm_pipeline(subject, file_spec, pipeline, raw_cw, raw_hb, target_pair_names, truth_templates, snirf_file_path, config):
    """
    Main Python analysis branch.

    Runtime-focused changes:
      - reuse FIR-resampled raws
      - cache nuisance constructions and design matrices
      - for channel-specific nuisance branches, group channels that share an
        identical nuisance design and fit them together instead of one-by-one
    """
    _, run_glm, _ = import_mne_nirs_modules()
    canonical_metric_frames = []
    fir_metric_frames = []
    nuisance_detail_rows = []
    shape_metric_rows = []
    roi_timecourse_rows = []
    long_channel_names = get_available_long_channel_names(raw_hb, config)
    if not long_channel_names:
        return {'canonical_channel_metrics': pd.DataFrame(), 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': pd.DataFrame(), 'shape_metrics': pd.DataFrame(), 'roi_timecourses': pd.DataFrame(), 'nuisance_detail': pd.DataFrame(), 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}

    def _append_nuisance_rows(channel_names, chromophore, nuisance_metadata):
        for channel_name in channel_names:
            nuisance_detail_rows.append({'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})

    uses_global_nuisance = pipeline['nuisance_method'] in {'none', 'pooled_pca2', 'ss_aux_pca'}
    if uses_global_nuisance:
        nuisance_by_chromophore = build_global_nuisance_regressors(raw_hb, snirf_file_path, raw_cw, pipeline['nuisance_method'], config)
        if pipeline['hrf_model'] == 'fir':
            raw_fir = _resample_for_fir_cached(raw_hb, config)
            fir_long_channel_names = get_available_long_channel_names(raw_fir, config)
            for chromophore in ['hbo', 'hbr']:
                nuisance_info = nuisance_by_chromophore[chromophore]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                channel_subset = [name for name in fir_long_channel_names if name.endswith(chromophore)]
                _append_nuisance_rows(channel_subset, chromophore, nuisance_metadata)
                if not channel_subset:
                    continue
                nuisance_signature = _get_nuisance_signature(nuisance_df, nuisance_metadata)
                nuisance_df_fir = resample_nuisance_dataframe_to_fir_times(nuisance_df, raw_hb.times, raw_fir.times)
                design_matrix = _build_design_matrix_cached(raw_fir, 'fir', nuisance_df_fir, nuisance_signature, config)
                glm_result = run_glm(raw_fir.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                fir_metric_df = parse_fir_dataframe(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {name: nuisance_metadata for name in channel_subset}, config)
                if len(fir_metric_df) > 0:
                    fir_metric_frames.append(fir_metric_df)
        else:
            epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
            effective_stim_duration_s = get_effective_stim_duration_s(raw_hb, config)
            for chromophore in ['hbo', 'hbr']:
                nuisance_info = nuisance_by_chromophore[chromophore]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                channel_subset = [name for name in long_channel_names if name.endswith(chromophore)]
                _append_nuisance_rows(channel_subset, chromophore, nuisance_metadata)
                if not channel_subset:
                    continue
                nuisance_signature = _get_nuisance_signature(nuisance_df, nuisance_metadata)
                design_matrix = _build_design_matrix_cached(raw_hb, pipeline['hrf_model'], nuisance_df, nuisance_signature, config)
                glm_result = run_glm(raw_hb.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                channel_metric_df = parse_mne_glm_to_channel_rows(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {name: nuisance_metadata for name in channel_subset})
                if len(channel_metric_df) == 0:
                    continue
                canonical_metric_frames.append(channel_metric_df)
                shape_metric_rows.extend(collect_canonical_shape_metrics(channel_metric_df, epoch_times_s, file_spec.get('truth_template_key'), pipeline, truth_templates, stim_duration_s=effective_stim_duration_s))
                roi_timecourse_rows.extend(collect_canonical_roi_timecourses(channel_metric_df, epoch_times_s, pipeline, stim_duration_s=effective_stim_duration_s))
    else:
        channel_nuisance = build_channel_nuisance_regressors(raw_hb, pipeline['nuisance_method'], config)
        if pipeline['hrf_model'] == 'fir':
            raw_fir = _resample_for_fir_cached(raw_hb, config)
            fir_long_channel_names = get_available_long_channel_names(raw_fir, config)
            fir_channel_nuisance = build_channel_nuisance_regressors(raw_fir, pipeline['nuisance_method'], config)
            grouped = {}
            for channel_name in fir_long_channel_names:
                chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
                nuisance_info = fir_channel_nuisance[channel_name]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                _append_nuisance_rows([channel_name], chromophore, nuisance_metadata)
                nuisance_signature = _get_nuisance_signature(nuisance_df, nuisance_metadata)
                grouped.setdefault((chromophore, nuisance_signature), {'channels': [], 'nuisance_df': nuisance_df, 'nuisance_metadata': nuisance_metadata})['channels'].append(channel_name)
            for (chromophore, nuisance_signature), group in grouped.items():
                design_matrix = _build_design_matrix_cached(raw_fir, 'fir', group['nuisance_df'], nuisance_signature, config)
                channel_subset = group['channels']
                glm_result = run_glm(raw_fir.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                nuisance_metadata_by_channel = {name: group['nuisance_metadata'] for name in channel_subset}
                fir_metric_df = parse_fir_dataframe(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, nuisance_metadata_by_channel, config)
                if len(fir_metric_df) > 0:
                    fir_metric_frames.append(fir_metric_df)
        else:
            epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
            effective_stim_duration_s = get_effective_stim_duration_s(raw_hb, config)
            grouped = {}
            for channel_name in long_channel_names:
                chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
                nuisance_info = channel_nuisance[channel_name]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                _append_nuisance_rows([channel_name], chromophore, nuisance_metadata)
                nuisance_signature = _get_nuisance_signature(nuisance_df, nuisance_metadata)
                grouped.setdefault((chromophore, nuisance_signature), {'channels': [], 'nuisance_df': nuisance_df, 'nuisance_metadata': nuisance_metadata})['channels'].append(channel_name)
            for (chromophore, nuisance_signature), group in grouped.items():
                design_matrix = _build_design_matrix_cached(raw_hb, pipeline['hrf_model'], group['nuisance_df'], nuisance_signature, config)
                channel_subset = group['channels']
                glm_result = run_glm(raw_hb.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                nuisance_metadata_by_channel = {name: group['nuisance_metadata'] for name in channel_subset}
                channel_metric_df = parse_mne_glm_to_channel_rows(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, nuisance_metadata_by_channel)
                if len(channel_metric_df) > 0:
                    canonical_metric_frames.append(channel_metric_df)
            canonical_metric_df = pd.concat(canonical_metric_frames, ignore_index=True) if canonical_metric_frames else pd.DataFrame()
            shape_metric_rows.extend(collect_canonical_shape_metrics(canonical_metric_df, epoch_times_s, file_spec.get('truth_template_key'), pipeline, truth_templates, stim_duration_s=effective_stim_duration_s))
            roi_timecourse_rows.extend(collect_canonical_roi_timecourses(canonical_metric_df, epoch_times_s, pipeline, stim_duration_s=effective_stim_duration_s))
            return {'canonical_channel_metrics': canonical_metric_df, 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': pd.DataFrame(), 'shape_metrics': pd.DataFrame(shape_metric_rows), 'roi_timecourses': pd.DataFrame(roi_timecourse_rows), 'nuisance_detail': pd.DataFrame(nuisance_detail_rows), 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}

    canonical_metric_df = pd.concat(canonical_metric_frames, ignore_index=True) if canonical_metric_frames else pd.DataFrame()
    fir_metric_df = pd.concat(fir_metric_frames, ignore_index=True) if fir_metric_frames else pd.DataFrame()
    if len(fir_metric_df) > 0:
        fir_effective_stim_duration_s = get_effective_stim_duration_s(raw_fir if 'raw_fir' in locals() else raw_hb, config)
        fir_shape_rows, fir_roi_rows = summarize_fir_shapes(fir_metric_df, file_spec.get('truth_template_key'), pipeline, truth_templates, config, stim_duration_s=fir_effective_stim_duration_s)
        shape_metric_rows.extend(fir_shape_rows)
        roi_timecourse_rows.extend(fir_roi_rows)
    return {'canonical_channel_metrics': canonical_metric_df, 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': fir_metric_df, 'shape_metrics': pd.DataFrame(shape_metric_rows), 'roi_timecourses': pd.DataFrame(roi_timecourse_rows), 'nuisance_detail': pd.DataFrame(nuisance_detail_rows), 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}

def build_matlab_input_spec(subject, file_label, amplitude_value, pipeline, snirf_file_path, raw_cw, pair_quality_df, target_pair_names, shift_index, shift_s, config):
    """
    Prepare a minimal spec for a toolbox-native AnalyzIR run.

    Python is used only for orchestration and benchmark bookkeeping. The actual
    AR-IRLS pipeline is executed natively in nirs-toolbox. To avoid Python-side
    preprocessing/modeling mismatches, the spec now passes only:
      - stimulus schedule fallback
      - target-pair truth labels
      - QC-derived bad pair names
      - short-separation pair names from raw probe geometry
      - scaling/export options
    The MATLAB helper then performs Resample->OD->(TDDR)->BeerLambertLaw->GLM.
    """
    if pair_quality_df is None or len(pair_quality_df) == 0:
        return None

    bad_pair_names = get_bad_pair_names(pair_quality_df, pipeline['pruning_style'], config)
    keep_pairs_df = pair_quality_df.loc[~pair_quality_df['pair_name'].isin(bad_pair_names)].copy()
    if len(keep_pairs_df) == 0:
        return None

    short_pair_names = sorted(
        keep_pairs_df.loc[keep_pairs_df['group'] == 'SS', 'pair_name'].astype(str).unique().tolist()
    )
    long_pair_names = sorted(
        keep_pairs_df.loc[keep_pairs_df['group'] == 'LS', 'pair_name'].astype(str).unique().tolist()
    )

    return {
        'snirf_file': str(snirf_file_path),
        'subject': subject,
        'file_label': file_label,
        'amplitude_value': amplitude_value,
        'pipeline_label': pipeline['label'],
        'backend': pipeline['backend'],
        'hrf_model': pipeline['hrf_model'],
        'solver': pipeline['solver'],
        'shift_index': int(shift_index),
        'shift_s': float(shift_s),
        'target_pair_names': sorted(list(target_pair_names)),
        'bad_pair_names': sorted(list(bad_pair_names)),
        'short_pair_names': short_pair_names,
        'long_pair_names': long_pair_names,
        'analyzir_resample_fs_hz': float(config.get('analyzir_resample_fs_hz', 1.0)),
        'analyzir_use_tddr': bool(pipeline.get('motion_method') == 'tddr'),
        'ppf_value': float(config.get('ppf_value', 6.0)),
        'hrf_export_fs_hz': float(config.get('analyzir_hrf_export_fs_hz', config.get('analyzir_resample_fs_hz', 1.0))),
        'short_separation_threshold_m': float(config['short_separation_threshold_m']),
        'stim_onset_s': np.asarray([], dtype=float),
        'stim_dur_s': np.asarray([], dtype=float),
        'stim_amp': np.asarray([], dtype=float),
    }


def prepare_matlab_pipeline_inputs(subject, file_spec, pipeline, snirf_file_path, raw_cw, pair_quality_df, target_pair_names, config):
    observed_spec = build_matlab_input_spec(
        subject=subject,
        file_label=file_spec['label'],
        amplitude_value=file_spec['amplitude_value'],
        pipeline=pipeline,
        snirf_file_path=snirf_file_path,
        raw_cw=raw_cw,
        pair_quality_df=pair_quality_df,
        target_pair_names=target_pair_names,
        shift_index=0,
        shift_s=0.0,
        config=config,
    )
    if observed_spec is not None:
        stim_events_df = get_annotation_events_df(raw_cw, config)
        if len(stim_events_df) > 0:
            observed_spec['stim_onset_s'] = stim_events_df['onset'].to_numpy(dtype=float)
            observed_spec['stim_dur_s'] = stim_events_df['duration'].to_numpy(dtype=float)
            observed_spec['stim_amp'] = np.ones(len(stim_events_df), dtype=float)

    return {
        'canonical_channel_metrics': pd.DataFrame(),
        'block_average_channel_metrics': pd.DataFrame(),
        'fir_channel_metrics': pd.DataFrame(),
        'shape_metrics': pd.DataFrame(),
        'roi_timecourses': pd.DataFrame(),
        'nuisance_detail': pd.DataFrame(),
        'matlab_input_specs_list': [] if observed_spec is None else [observed_spec],
        'matlab_shift_specs_list': [],
    }


def run_single_pipeline(subject, file_spec, pipeline, raw_cw, raw_hb, pair_quality_df, target_pair_names, truth_templates, snirf_file_path, config):
    """
    One readable dispatch point for all pipeline families.
    """
    if pipeline['use_block_average']:
        return run_block_average_pipeline(subject, file_spec, pipeline, raw_hb, target_pair_names, truth_templates, config)
    if pipeline['backend'] in {'matlab_arirls', 'matlab_arirls_native'}:
        return prepare_matlab_pipeline_inputs(subject, file_spec, pipeline, snirf_file_path, raw_cw, pair_quality_df, target_pair_names, config)
    if pipeline['backend'] == 'matlab_homer3':
        return run_homer3_pipeline(subject, file_spec, pipeline, snirf_file_path, raw_cw, target_pair_names, truth_templates, config)
    return run_python_glm_pipeline(subject, file_spec, pipeline, raw_cw, raw_hb, target_pair_names, truth_templates, snirf_file_path, config)

def load_subject_file_inputs(subject, file_spec, config):
    """
    Load everything needed for one subject/file job.

    This keeps the top-level job function easier to read:
      - raw data
      - cleaned annotations
      - truth target definitions
      - QC tables
      - truth templates
    """
    dataset_dir = dataset_path(config)
    subject_dir = dataset_dir / subject
    snirf_file_path = subject_dir / file_spec['filename']
    annotation_source_path = subject_dir / file_spec['annotation_source_filename'] if file_spec['annotation_source_filename'] else None

    reference_filename = file_spec.get('truth_label_source_filename', 'resting_hrf_20.snirf')
    reference_path = subject_dir / reference_filename

    if not snirf_file_path.exists():
        raise BenchmarkError(f'Missing input file: {snirf_file_path}')
    if not reference_path.exists():
        raise BenchmarkError(f'Reference truth file missing: {reference_path}')

    raw_cw = mne.io.read_raw_snirf(snirf_file_path, preload=True, verbose=False)

    if file_spec['is_null']:
        if annotation_source_path is None or not annotation_source_path.exists():
            raise BenchmarkError('Null file needs a valid annotation source file.')
        raw_cw = copy_valid_annotations(raw_cw, annotation_source_path)

    raw_cw = sanitize_annotations_to_single_task(raw_cw)

    reference_raw = mne.io.read_raw_snirf(reference_path, preload=True, verbose=False)
    data_type_labels = read_measurement_data_type_labels(reference_path)
    if data_type_labels is None:
        raise BenchmarkError(f'Could not read truth labels from reference file: {reference_path}')

    reference_cw_indices = get_cw_channel_indices(reference_raw)
    reference_cw_names = np.asarray(reference_raw.ch_names)[reference_cw_indices]

    if len(data_type_labels) == len(reference_cw_names):
        aligned_reference_names = reference_cw_names
    elif len(data_type_labels) == len(reference_raw.ch_names):
        aligned_reference_names = np.asarray(reference_raw.ch_names)
    else:
        raise BenchmarkError(
            f'Truth-label alignment failed for {reference_path}: '
            f'{len(data_type_labels)=}, {len(reference_cw_names)=}, {len(reference_raw.ch_names)=}'
        )

    target_pair_names = sorted(
        set(name.split(' ')[0] for name in aligned_reference_names[data_type_labels == 1].tolist())
    )
    target_pair_name_set = set(target_pair_names)

    cw_channel_table = build_cw_channel_table(reference_raw, subject, file_spec['label'], config)
    long_pair_names = sorted(
        cw_channel_table.loc[cw_channel_table['group'] == 'LS', 'pair_name'].astype(str).unique()
    )
    non_target_pair_names = [
        pair_name for pair_name in long_pair_names
        if pair_name not in target_pair_name_set
    ]

    channel_quality_df, pair_quality_df = build_quality_tables(raw_cw, subject, file_spec['label'], config)
    truth_templates = load_truth_templates(config)

    return {
        'subject_dir': subject_dir,
        'snirf_file_path': snirf_file_path,
        'reference_raw': reference_raw,
        'raw_cw': raw_cw,
        'target_pair_names': target_pair_names,
        'target_pair_name_set': target_pair_name_set,
        'non_target_pair_names': non_target_pair_names,
        'truth_templates': truth_templates,
        'channel_quality': channel_quality_df,
        'pair_quality': pair_quality_df,
        'cw_channel_table': cw_channel_table,
    }

def summarize_shift_scores_from_payload(result_payload):
    summary_tables = []
    canonical_df = result_payload.get('canonical_channel_metrics', pd.DataFrame())
    if canonical_df is not None and len(canonical_df) > 0:
        summary_tables.append(summarize_target_minus_non_target(canonical_df, 'beta', score_type='canonical_beta'))
    block_df = result_payload.get('block_average_channel_metrics', pd.DataFrame())
    if block_df is not None and len(block_df) > 0 and ('score' in block_df.columns):
        summary_tables.append(summarize_target_minus_non_target(block_df, 'score', score_type='block_average_score'))
    shape_df = result_payload.get('shape_metrics', pd.DataFrame())
    if shape_df is not None and len(shape_df) > 0:
        for value_column, score_type in [('recovered_peak_amplitude', 'shape_peak_amplitude'), ('recovered_auc', 'shape_auc')]:
            if value_column in shape_df.columns:
                summary_tables.append(summarize_target_minus_non_target(shape_df, value_column, score_type=score_type))
    return pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()

def build_channel_availability_row(subject, file_spec, pipeline, raw_hb, target_pair_names, non_target_pair_names, bad_pair_names, config):
    available_long_channel_names = get_available_long_channel_names(raw_hb, config)
    available_target_pairs = sorted({name.split(' ')[0] for name in available_long_channel_names if name.split(' ')[0] in set(target_pair_names)})
    available_non_target_pairs = sorted({name.split(' ')[0] for name in available_long_channel_names if name.split(' ')[0] in set(non_target_pair_names)})
    return {'subject': subject, 'file_label': file_spec['label'], 'amplitude_value': file_spec['amplitude_value'], 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'n_total_true_target_pairs': len(target_pair_names), 'n_available_true_target_pairs': len(available_target_pairs), 'n_total_true_non_target_pairs': len(non_target_pair_names), 'n_available_true_non_target_pairs': len(available_non_target_pairs), 'n_available_long_channels': len(available_long_channel_names), 'target_pair_retention_fraction': len(available_target_pairs) / len(target_pair_names) if target_pair_names else np.nan, 'non_target_pair_retention_fraction': len(available_non_target_pairs) / len(non_target_pair_names) if non_target_pair_names else np.nan, 'n_bad_pairs': len(bad_pair_names)}

def collect_python_empirical_null_rows(subject, file_spec, pipeline, raw_cw, raw_hb, pair_quality_df, target_pair_name_set, truth_templates, snirf_file_path, config, errors):
    """
    For null files, re-run the Python pipeline with shifted annotations and store
    target-vs-non-target summaries for the empirical null.
    """
    empirical_null_tables = []
    total_duration_s = float(raw_hb.times[-1])
    shift_values_s = get_shift_values(total_duration_s, config['empirical_null_shift_count'], config['empirical_null_min_shift_s'])
    for shift_index, shift_s in enumerate(shift_values_s, start=1):
        shifted_raw_hb = raw_hb.copy()
        shifted_raw_hb.set_annotations(make_shifted_annotations(raw_hb.annotations, total_duration_s, shift_s))
        try:
            shifted_payload = run_single_pipeline(subject=subject, file_spec=file_spec, pipeline=pipeline, raw_cw=raw_cw, raw_hb=shifted_raw_hb, pair_quality_df=pair_quality_df, target_pair_names=target_pair_name_set, truth_templates=truth_templates, snirf_file_path=snirf_file_path, config=config)
        except Exception as exc:
            errors.append(make_error_row(subject, file_spec['label'], pipeline['label'], 'empirical_null_shift_python', str(exc)))
            continue
        shift_summary_df = summarize_shift_scores_from_payload(shifted_payload)
        if len(shift_summary_df) == 0:
            continue
        for _, summary_row in shift_summary_df.iterrows():
            empirical_null_tables.append(pd.DataFrame([{'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': summary_row['chromophore'], 'score_type': summary_row.get('score_type', 'unknown'), 'shift_index': shift_index, 'shift_s': shift_s, 'mean_target_score': summary_row['mean_target_score'], 'mean_non_target_score': summary_row['mean_non_target_score'], 'target_minus_non_target_score': summary_row['target_minus_non_target_score'], 'n_channels': summary_row.get('n_channels', np.nan)}]))
    return empirical_null_tables

def lookup_pipeline_spec(config, pipeline_label, fallback_hrf_model=None, fallback_backend='matlab_arirls'):
    for pipeline in config.get('pipeline_specs', []):
        if pipeline.get('label') == pipeline_label:
            return pipeline
    return {
        'label': pipeline_label,
        'backend': fallback_backend,
        'hrf_model': fallback_hrf_model or 'canonical',
        'solver': 'arirls',
    }


def append_matlab_outputs_to_job_tables(all_canonical_tables, all_shape_tables, all_roi_tables, all_empirical_tables, matlab_df, subject, file_spec, raw_hb, truth_templates, config):
    if matlab_df is None or len(matlab_df) == 0:
        return

    observed_matlab_df = matlab_df.loc[matlab_df['shift_index'] == 0].copy() if 'shift_index' in matlab_df.columns else matlab_df.copy()
    shifted_matlab_df = matlab_df.loc[matlab_df['shift_index'] > 0].copy() if 'shift_index' in matlab_df.columns else pd.DataFrame()

    if 'data_row_kind' in observed_matlab_df.columns:
        observed_beta_df = observed_matlab_df.loc[observed_matlab_df['data_row_kind'] == 'beta'].copy()
        observed_curve_df = observed_matlab_df.loc[observed_matlab_df['data_row_kind'] == 'hrf_curve'].copy()
    else:
        observed_beta_df = observed_matlab_df.copy()
        observed_curve_df = pd.DataFrame()

    if len(observed_beta_df) > 0:
        all_canonical_tables.append(observed_beta_df)

    if len(observed_curve_df) > 0:
        for curve_source, curve_df in observed_curve_df.groupby(observed_curve_df.get('curve_source', pd.Series('matlab_native_timecourse', index=observed_curve_df.index)).fillna('matlab_native_timecourse')):
            shape_rows = collect_shape_metrics_from_timecourse_table(curve_df, file_spec.get('truth_template_key'), truth_templates, shape_source=str(curve_source))
            if shape_rows:
                all_shape_tables.append(pd.DataFrame(shape_rows))
            roi_rows = collect_roi_timecourses_from_timecourse_table(curve_df, curve_source=str(curve_source))
            if roi_rows:
                all_roi_tables.append(pd.DataFrame(roi_rows))
    elif len(observed_beta_df) > 0:
        epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
        effective_stim_duration_s = get_effective_stim_duration_s(raw_hb, config)
        for pipeline_label, pipeline_observed_df in observed_beta_df.groupby('pipeline_label'):
            fallback_hrf_model = pipeline_observed_df['hrf_model'].iloc[0] if 'hrf_model' in pipeline_observed_df.columns and len(pipeline_observed_df) > 0 else ''
            pipeline_spec = lookup_pipeline_spec(config, pipeline_label, fallback_hrf_model=fallback_hrf_model)
            shape_rows = collect_canonical_shape_metrics(pipeline_observed_df, epoch_times_s, file_spec.get('truth_template_key'), pipeline_spec, truth_templates, stim_duration_s=effective_stim_duration_s)
            if shape_rows:
                all_shape_tables.append(pd.DataFrame(shape_rows))
            roi_rows = collect_canonical_roi_timecourses(pipeline_observed_df, epoch_times_s, pipeline_spec, stim_duration_s=effective_stim_duration_s)
            if roi_rows:
                all_roi_tables.append(pd.DataFrame(roi_rows))

    if len(shifted_matlab_df) == 0:
        return

    if 'data_row_kind' in shifted_matlab_df.columns:
        shifted_beta_df = shifted_matlab_df.loc[shifted_matlab_df['data_row_kind'] == 'beta'].copy()
        shifted_curve_df = shifted_matlab_df.loc[shifted_matlab_df['data_row_kind'] == 'hrf_curve'].copy()
    else:
        shifted_beta_df = shifted_matlab_df.copy()
        shifted_curve_df = pd.DataFrame()

    score_tables = []

    if len(shifted_beta_df) > 0 and 'beta' in shifted_beta_df.columns:
        separation_df = shifted_beta_df.groupby(['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore', 'target_status'], as_index=False).agg(mean_score=('beta', 'mean'), n_channels=('beta', 'count'))
        separation_wide_df = separation_df.pivot_table(index=['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore'], columns='target_status', values='mean_score').reset_index().rename(columns={'true_target': 'mean_target_score', 'true_non_target': 'mean_non_target_score'})
        count_df = separation_df.groupby(['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore'], as_index=False)['n_channels'].sum()
        separation_wide_df = separation_wide_df.merge(count_df, on=['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore'], how='left')
        if 'mean_target_score' not in separation_wide_df.columns:
            separation_wide_df['mean_target_score'] = np.nan
        if 'mean_non_target_score' not in separation_wide_df.columns:
            separation_wide_df['mean_non_target_score'] = np.nan
        separation_wide_df['target_minus_non_target_score'] = separation_wide_df['mean_target_score'] - separation_wide_df['mean_non_target_score']
        separation_wide_df['score_type'] = 'canonical_beta'
        score_tables.append(separation_wide_df)

    if len(shifted_curve_df) > 0:
        curve_metrics = shifted_curve_df.groupby(['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore', 'target_status', 'channel_name'], as_index=False).apply(
            lambda g: pd.Series({
                'recovered_peak_amplitude': float(np.nanmax(g['signal'].to_numpy(dtype=float))) if str(g['chromophore'].iloc[0]).lower() == 'hbo' else float(np.nanmin(g['signal'].to_numpy(dtype=float))),
                'recovered_auc': float(scipy.integrate.trapezoid(g.sort_values('time_s')['signal'].to_numpy(dtype=float), g.sort_values('time_s')['time_s'].to_numpy(dtype=float))),
            })
        ).reset_index(drop=True)
        for value_column, score_type in [('recovered_peak_amplitude', 'shape_peak_amplitude'), ('recovered_auc', 'shape_auc')]:
            extra_df = summarize_target_minus_non_target(curve_metrics, value_column, group_columns=['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore'], score_type=score_type)
            if len(extra_df) > 0:
                score_tables.append(extra_df)

    if score_tables:
        all_empirical_tables.append(pd.concat(score_tables, ignore_index=True))


def resolve_matlab_hrf_model(model_value=None, pipeline_label=None):
    model = str(model_value or '').strip().lower()
    if model in {'glover', 'glover_derivs', 'spm', 'spm_derivs', 'gamma', 'gamma_derivs', 'canonical', 'canonical_derivs', 'fir'}:
        return model

    label = str(pipeline_label or '').strip().lower()
    if 'canonical' in label and 'deriv' in label:
        return 'canonical_derivs'
    if 'gamma' in label and 'deriv' in label:
        return 'gamma_derivs'
    if 'spm' in label and 'deriv' in label:
        return 'spm_derivs'
    if 'glover' in label and 'deriv' in label:
        return 'glover_derivs'
    if 'gamma' in label:
        return 'gamma'
    if 'spm' in label:
        return 'spm'
    if 'glover' in label:
        return 'glover'
    if 'fir' in label:
        return 'fir'
    return 'canonical'

def process_subject_file_job(subject, file_spec_dict, config_dict):
    """
    Main per-job analysis function.

    This is intentionally linear and stage-oriented so it reads like a notebook:
      - set up job folder
      - load raw data and truth labels
      - write QC tables
      - run each pipeline
      - optionally run MATLAB sidecar
      - write all per-job outputs
    """
    config = config_dict
    file_spec = file_spec_dict
    clear_job_runtime_caches()
    job_dir = ensure_dir(jobs_path(config) / subject / file_spec['label'])
    success_marker_path = job_dir / 'SUCCESS.json'
    if success_marker_path.exists() and (not config['overwrite']):
        return {'subject': subject, 'file_label': file_spec['label'], 'status': 'skipped_existing', 'job_dir': str(job_dir)}
    if config['overwrite'] and job_dir.exists():
        for child_path in job_dir.iterdir():
            if child_path.is_file():
                child_path.unlink(missing_ok=True)
            else:
                shutil.rmtree(child_path, ignore_errors=True)
        ensure_dir(job_dir)
    (job_dir / 'STARTED.json').write_text(json.dumps({'subject': subject, 'file_label': file_spec['label'], 'started_utc': now_utc_iso()}, indent=2), encoding='utf-8')
    errors = []
    try:
        loaded = load_subject_file_inputs(subject, file_spec, config)
        raw_cw = loaded['raw_cw']
        target_pair_names = loaded['target_pair_names']
        target_pair_name_set = loaded['target_pair_name_set']
        non_target_pair_names = loaded['non_target_pair_names']
        truth_templates = loaded['truth_templates']
        channel_quality_df = loaded['channel_quality']
        pair_quality_df = loaded['pair_quality']
        snirf_file_path = loaded['snirf_file_path']
        truth_summary_df = pd.DataFrame([{'subject': subject, 'file_label': file_spec['label'], 'amplitude_value': file_spec['amplitude_value'], 'n_true_target_pairs': len(target_pair_names), 'n_true_non_target_pairs': len(non_target_pair_names), 'target_pair_names': '|'.join(target_pair_names)}])
        maybe_write_table(truth_summary_df, job_dir / 'truth_summary', config['write_csv'], config['write_parquet'])
        maybe_write_table(channel_quality_df, job_dir / 'channel_quality', config['write_csv'], config['write_parquet'])
        maybe_write_table(pair_quality_df, job_dir / 'pair_quality', config['write_csv'], config['write_parquet'])
        preprocessed_cache = {}

        def get_preprocessed_hb_for_pipeline(pipeline):
            cache_key = (pipeline['pruning_style'], pipeline['motion_method'], pipeline['filter_mode'])
            if cache_key not in preprocessed_cache:
                preprocessed_cache[cache_key] = preprocess_raw_to_hb(raw_cw, pair_quality_df, pipeline, config)
            return preprocessed_cache[cache_key]
        all_canonical_tables = []
        all_block_tables = []
        all_fir_tables = []
        all_shape_tables = []
        all_roi_tables = []
        all_nuisance_tables = []
        all_empirical_tables = []
        all_homer3_hrf_tables = []
        all_homer3_beta_tables = []
        all_homer3_metadata_tables = []
        channel_availability_rows = []
        matlab_observed_specs = []
        matlab_shift_specs = []
        epoch_raw_hb = None
        
        for pipeline in config['pipeline_specs']:
            if pipeline['backend'] in {'matlab_arirls', 'matlab_arirls_native', 'matlab_homer3'} and (not config['use_matlab']):
                continue
            if pipeline['backend'] == 'matlab_homer3':
                raw_hb = None
                bad_pair_names = []
            else:
                raw_hb, bad_pair_names = get_preprocessed_hb_for_pipeline(pipeline)
                if epoch_raw_hb is None:
                    epoch_raw_hb = raw_hb
            try:
                pipeline_payload = run_single_pipeline(subject=subject, file_spec=file_spec, pipeline=pipeline, raw_cw=raw_cw, raw_hb=raw_hb, pair_quality_df=pair_quality_df, target_pair_names=target_pair_name_set, truth_templates=truth_templates, snirf_file_path=snirf_file_path, config=config)
            except Exception as exc:
                errors.append(make_error_row(subject, file_spec['label'], pipeline['label'], 'pipeline_execute', f'{exc}\n\n{traceback.format_exc()}'))
                continue
            if len(pipeline_payload['canonical_channel_metrics']) > 0:
                all_canonical_tables.append(pipeline_payload['canonical_channel_metrics'])
            if len(pipeline_payload['block_average_channel_metrics']) > 0:
                all_block_tables.append(pipeline_payload['block_average_channel_metrics'])
            if len(pipeline_payload['fir_channel_metrics']) > 0:
                all_fir_tables.append(pipeline_payload['fir_channel_metrics'])
            if len(pipeline_payload['shape_metrics']) > 0:
                all_shape_tables.append(pipeline_payload['shape_metrics'])
            if len(pipeline_payload['roi_timecourses']) > 0:
                all_roi_tables.append(pipeline_payload['roi_timecourses'])
            if len(pipeline_payload['nuisance_detail']) > 0:
                all_nuisance_tables.append(pipeline_payload['nuisance_detail'])
            if len(pipeline_payload.get('homer3_channel_hrf', pd.DataFrame())) > 0:
                all_homer3_hrf_tables.append(pipeline_payload['homer3_channel_hrf'])
            if len(pipeline_payload.get('homer3_beta', pd.DataFrame())) > 0:
                all_homer3_beta_tables.append(pipeline_payload['homer3_beta'])
            if len(pipeline_payload.get('homer3_glm_metadata', pd.DataFrame())) > 0:
                all_homer3_metadata_tables.append(pipeline_payload['homer3_glm_metadata'])
            if pipeline_payload['matlab_input_specs_list']:
                matlab_observed_specs.extend(pipeline_payload['matlab_input_specs_list'])
            if pipeline_payload['matlab_shift_specs_list']:
                matlab_shift_specs.extend(pipeline_payload['matlab_shift_specs_list'])
            if raw_hb is not None:
                channel_availability_rows.append(build_channel_availability_row(subject, file_spec, pipeline, raw_hb, target_pair_names, non_target_pair_names, bad_pair_names, config))
            elif pipeline_payload.get('channel_availability_row') is not None:
                availability_row = dict(pipeline_payload['channel_availability_row'])
                availability_row['n_total_true_non_target_pairs'] = len(non_target_pair_names)
                availability_row['n_available_true_non_target_pairs'] = availability_row.get('n_available_true_non_target_pairs', np.nan)
                availability_row['non_target_pair_retention_fraction'] = availability_row['n_available_true_non_target_pairs'] / len(non_target_pair_names) if non_target_pair_names else np.nan
                channel_availability_rows.append(availability_row)
            if file_spec['is_null'] and pipeline['include_in_empirical_null'] and (pipeline['backend'] in {'python', 'python_mne'}):
                all_empirical_tables.extend(collect_python_empirical_null_rows(subject, file_spec, pipeline, raw_cw, raw_hb, pair_quality_df, target_pair_name_set, truth_templates, snirf_file_path, config, errors))
                
        if config['use_matlab'] and (matlab_observed_specs or matlab_shift_specs):
            matlab_bundle_json_path = write_matlab_bundle(matlab_observed_specs, matlab_shift_specs, job_dir)
            if matlab_bundle_json_path is not None:
                try:
                    matlab_df = run_matlab_sidecar(config, matlab_bundle_json_path, job_dir)
                except Exception as exc:
                    errors.append(make_error_row(subject, file_spec['label'], 'MATLAB_ARIRLS', 'matlab_sidecar', f'{exc}\n\n{traceback.format_exc()}'))
                    matlab_df = pd.DataFrame()
        
                if epoch_raw_hb is not None:
                    append_matlab_outputs_to_job_tables(
                        all_canonical_tables,
                        all_shape_tables,
                        all_roi_tables,
                        all_empirical_tables,
                        matlab_df,
                        subject,
                        file_spec,
                        epoch_raw_hb,
                        truth_templates,
                        config,
                    )
                else:
                    errors.append(
                        make_error_row(
                            subject,
                            file_spec['label'],
                            'MATLAB_APPEND',
                            'matlab_append',
                            'No valid raw_hb available for MATLAB output reconstruction.',
                        )
                    )                
                
        maybe_write_table(pd.DataFrame(channel_availability_rows), job_dir / 'channel_availability', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_canonical_tables, ignore_index=True) if all_canonical_tables else pd.DataFrame(), job_dir / 'canonical_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_block_tables, ignore_index=True) if all_block_tables else pd.DataFrame(), job_dir / 'block_average_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_fir_tables, ignore_index=True) if all_fir_tables else pd.DataFrame(), job_dir / 'fir_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_shape_tables, ignore_index=True) if all_shape_tables else pd.DataFrame(), job_dir / 'shape_fidelity', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_roi_tables, ignore_index=True) if all_roi_tables else pd.DataFrame(), job_dir / 'roi_timecourses', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_nuisance_tables, ignore_index=True) if all_nuisance_tables else pd.DataFrame(), job_dir / 'nuisance_detail', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_homer3_hrf_tables, ignore_index=True) if all_homer3_hrf_tables else pd.DataFrame(), job_dir / 'homer3_channel_hrf', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_homer3_beta_tables, ignore_index=True) if all_homer3_beta_tables else pd.DataFrame(), job_dir / 'homer3_beta', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_homer3_metadata_tables, ignore_index=True) if all_homer3_metadata_tables else pd.DataFrame(), job_dir / 'homer3_glm_metadata', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_empirical_tables, ignore_index=True) if all_empirical_tables else pd.DataFrame(), job_dir / 'empirical_null_shift', config['write_csv'], config['write_parquet'])
        if errors:
            maybe_write_table(pd.DataFrame(errors), job_dir / 'error_log', config['write_csv'], config['write_parquet'])
        success_marker_path.write_text(json.dumps({'subject': subject, 'file_label': file_spec['label'], 'finished_utc': now_utc_iso(), 'job_dir': str(job_dir)}, indent=2), encoding='utf-8')
        return {'subject': subject, 'file_label': file_spec['label'], 'status': 'finished', 'job_dir': str(job_dir)}
    except Exception as exc:
        errors.append(make_error_row(subject, file_spec['label'], None, 'job_crash', f'{exc}\n\n{traceback.format_exc()}'))
        maybe_write_table(pd.DataFrame(errors), job_dir / 'error_log', config['write_csv'], config['write_parquet'])
        return {'subject': subject, 'file_label': file_spec['label'], 'status': 'crashed', 'job_dir': str(job_dir), 'error': str(exc)}

def aggregate_outputs(config, run_results):
    aggregate_dir = ensure_dir(aggregate_path(config))
    table_names = ['truth_summary', 'channel_quality', 'pair_quality', 'channel_availability', 'canonical_channel_metrics', 'block_average_channel_metrics', 'fir_channel_metrics', 'shape_fidelity', 'roi_timecourses', 'nuisance_detail', 'homer3_channel_hrf', 'homer3_beta', 'homer3_glm_metadata', 'empirical_null_shift', 'error_log']
    aggregated = {name: [] for name in table_names}
    for result in run_results:
        job_dir = Path(result['job_dir'])
        for name in table_names:
            df = read_any_table(job_dir / name)
            if len(df) > 0:
                aggregated[name].append(df)
    combined = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for name, frames in aggregated.items()}
    for name, df in list(combined.items()):
        combined[name] = attach_pipeline_metadata(df, config)
    manifest = pipeline_manifest_df(config)
    canonical_df = combined['canonical_channel_metrics']
    if len(canonical_df) > 0 and 'is_primary_task_regressor' in canonical_df.columns:
        canonical_primary_df = canonical_df.loc[canonical_df['is_primary_task_regressor'] == True].copy()
    else:
        canonical_primary_df = canonical_df.copy()
    block_df = combined['block_average_channel_metrics']
    fir_df = combined['fir_channel_metrics']
    shape_df = combined['shape_fidelity']
    pair_quality_df = combined['pair_quality']
    empirical_null_shift_df = combined['empirical_null_shift']
    if len(canonical_df) > 0 and 'p_value' in canonical_df.columns:
        canonical_df = canonical_df.copy()
        canonical_df['q_value_bh'] = np.nan
        for _, group in canonical_df.groupby(['subject', 'file_label', 'pipeline_label', 'chromophore']):
            idx = group.index.to_list()
            canonical_df.loc[idx, 'q_value_bh'] = benjamini_hochberg(group['p_value'].to_numpy())
        combined['canonical_channel_metrics'] = canonical_df

    def make_target_sep_table(df, value_column, score_type):
        if df is None or len(df) == 0 or value_column not in df.columns:
            return pd.DataFrame()
        sep = summarize_target_minus_non_target(df, value_column, group_columns=['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore'], score_type=score_type)
        return attach_pipeline_metadata(sep, config)
    target_sep_tables = [make_target_sep_table(canonical_primary_df, 'beta', 'canonical_beta'), make_target_sep_table(block_df, 'score', 'block_average_score'), make_target_sep_table(shape_df, 'recovered_peak_amplitude', 'shape_peak_amplitude'), make_target_sep_table(shape_df, 'recovered_auc', 'shape_auc')]
    target_vs_nontarget = pd.concat([df for df in target_sep_tables if len(df) > 0], ignore_index=True) if any((len(df) > 0 for df in target_sep_tables)) else pd.DataFrame()
    roi_tables = []
    if len(canonical_primary_df) > 0:
        c_roi = canonical_primary_df.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status'], as_index=False).agg(roi_mean_score=('beta', 'mean'), roi_std_score=('beta', 'std'), n_channels=('channel_name', 'count'))
        c_roi['score_type'] = 'canonical_beta'
        roi_tables.append(c_roi)
    if len(block_df) > 0:
        b_roi = block_df.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status'], as_index=False).agg(roi_mean_score=('score', 'mean'), roi_std_score=('score', 'std'), n_channels=('channel_name', 'count'))
        b_roi['score_type'] = 'block_average_score'
        roi_tables.append(b_roi)
    if len(shape_df) > 0:
        sp_roi = shape_df.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status'], as_index=False).agg(roi_mean_score=('recovered_peak_amplitude', 'mean'), roi_std_score=('recovered_peak_amplitude', 'std'), n_channels=('channel_name', 'count'))
        sp_roi['score_type'] = 'shape_peak_amplitude'
        roi_tables.append(sp_roi)
    roi_scores = pd.concat(roi_tables, ignore_index=True) if roi_tables else pd.DataFrame()
    roi_scores = attach_pipeline_metadata(roi_scores, config)
    dlpfc_pair_rows = []
    if len(pair_quality_df) > 0 and {'midpoint_x', 'midpoint_y', 'group'}.issubset(pair_quality_df.columns):
        ls_pairs = pair_quality_df.loc[pair_quality_df['group'] == 'LS'].copy()
        for (subject, file_label), group in ls_pairs.groupby(['subject', 'file_label']):
            group = group.loc[np.isfinite(group['midpoint_x']) & np.isfinite(group['midpoint_y'])].copy()
            if len(group) == 0:
                continue
            frontal_thresh = float(group['midpoint_y'].quantile(config['dlpfc_frontal_quantile']))
            lateral_thresh = float(group['midpoint_x'].abs().quantile(config['dlpfc_lateral_quantile']))
            bilateral = group.loc[(group['midpoint_y'] >= frontal_thresh) & (group['midpoint_x'].abs() >= lateral_thresh)].copy()
            if len(bilateral) == 0:
                bilateral = group.loc[group['midpoint_y'] >= frontal_thresh].copy()
            bilateral['roi_name'] = 'dlpfc_bilateral'
            left = bilateral.loc[bilateral['midpoint_x'] < 0].copy()
            left['roi_name'] = 'dlpfc_left'
            right = bilateral.loc[bilateral['midpoint_x'] > 0].copy()
            right['roi_name'] = 'dlpfc_right'
            dlpfc_pair_rows.extend([bilateral, left, right])
    dlpfc_pairs = pd.concat([df for df in dlpfc_pair_rows if len(df) > 0], ignore_index=True) if dlpfc_pair_rows else pd.DataFrame()

    def build_roi_subset_scores(metric_df, value_column, score_type):
        if len(metric_df) == 0 or len(dlpfc_pairs) == 0 or value_column not in metric_df.columns:
            return pd.DataFrame()
        merged = metric_df.merge(dlpfc_pairs[['subject', 'file_label', 'pair_name', 'roi_name']].drop_duplicates(), on=['subject', 'file_label', 'pair_name'], how='inner')
        if len(merged) == 0:
            return pd.DataFrame()
        out = merged.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'roi_name'], as_index=False).agg(roi_mean_score=(value_column, 'mean'), roi_std_score=(value_column, 'std'), n_channels=('channel_name', 'count'))
        out['score_type'] = score_type
        return attach_pipeline_metadata(out, config)
    dlpfc_roi_scores = pd.concat([build_roi_subset_scores(canonical_df, 'beta', 'canonical_beta'), build_roi_subset_scores(block_df, 'score', 'block_average_score'), build_roi_subset_scores(shape_df, 'recovered_peak_amplitude', 'shape_peak_amplitude')], ignore_index=True) if len(dlpfc_pairs) > 0 else pd.DataFrame()
    parametric_null_summary = pd.DataFrame()
    if len(canonical_primary_df) > 0 and 'p_value' in canonical_primary_df.columns:
        null_df = canonical_primary_df.loc[canonical_primary_df['amplitude_value'] == 0].copy()
        if len(null_df) > 0:
            null_df['is_false_positive_p_lt_0_05'] = null_df['p_value'] < 0.05
            null_df['is_false_positive_q_lt_0_05'] = null_df['q_value_bh'] < 0.05 if 'q_value_bh' in null_df.columns else np.nan
            parametric_null_summary = null_df.groupby(['subject', 'pipeline_label', 'backend', 'chromophore', 'target_status'], as_index=False).agg(false_positive_rate_p_lt_0_05=('is_false_positive_p_lt_0_05', 'mean'), false_positive_rate_q_lt_0_05=('is_false_positive_q_lt_0_05', 'mean'), mean_abs_beta=('beta', lambda x: float(np.mean(np.abs(x)))), n_channels=('channel_name', 'count'))
            parametric_null_summary = attach_pipeline_metadata(parametric_null_summary, config)
    empirical_null_shift_df = attach_pipeline_metadata(empirical_null_shift_df, config)
    empirical_null_pvalues_rows = []
    if len(target_vs_nontarget) > 0 and len(empirical_null_shift_df) > 0:
        observed_df = target_vs_nontarget.copy()
        for _, row in observed_df.iterrows():
            null_rows = empirical_null_shift_df.loc[(empirical_null_shift_df['subject'] == row['subject']) & (empirical_null_shift_df['pipeline_label'] == row['pipeline_label']) & (empirical_null_shift_df['chromophore'] == row['chromophore']) & (empirical_null_shift_df['score_type'] == row['score_type'])]
            if len(null_rows) == 0:
                continue
            observed_value = float(row['target_minus_non_target_score'])
            null_values = null_rows['target_minus_non_target_score'].to_numpy(dtype=float)
            if row['chromophore'] == 'hbo':
                empirical_p = (np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1)
            else:
                empirical_p = (np.sum(null_values <= observed_value) + 1) / (len(null_values) + 1)
            empirical_null_pvalues_rows.append({'subject': row['subject'], 'file_label': row['file_label'], 'amplitude_value': row.get('amplitude_value', np.nan), 'pipeline_label': row['pipeline_label'], 'backend': row['backend'], 'chromophore': row['chromophore'], 'score_type': row['score_type'], 'observed_target_minus_non_target_score': observed_value, 'null_shift_mean': float(np.mean(null_values)), 'null_shift_std': float(np.std(null_values)), 'null_shift_min': float(np.min(null_values)), 'null_shift_max': float(np.max(null_values)), 'empirical_p_value': empirical_p, 'n_null_shifts': len(null_values)})
    empirical_null_pvalues = attach_pipeline_metadata(pd.DataFrame(empirical_null_pvalues_rows), config)

    def variability_tables(target_sep_df, subset_name, subset_mask=None):
        if len(target_sep_df) == 0:
            return (pd.DataFrame(), pd.DataFrame())
        df = target_sep_df.copy()
        if subset_mask is not None:
            df = df.loc[subset_mask].copy()
        if len(df) == 0:
            return (pd.DataFrame(), pd.DataFrame())
        summary = df.groupby(['subject', 'file_label', 'amplitude_value', 'chromophore', 'score_type'], as_index=False).agg(mean_target_minus_non_target=('target_minus_non_target_score', 'mean'), std_across_pipelines=('target_minus_non_target_score', 'std'), min_across_pipelines=('target_minus_non_target_score', 'min'), max_across_pipelines=('target_minus_non_target_score', 'max'), n_pipelines=('pipeline_label', 'nunique'))
        summary['range_across_pipelines'] = summary['max_across_pipelines'] - summary['min_across_pipelines']
        summary['subset_name'] = subset_name
        pairwise_rows = []
        for keys, grp in df.groupby(['subject', 'file_label', 'amplitude_value', 'chromophore', 'score_type']):
            rows = grp[['pipeline_label', 'target_minus_non_target_score']].dropna().drop_duplicates()
            for left, right in itertools.combinations(rows.itertuples(index=False), 2):
                pairwise_rows.append({'subject': keys[0], 'file_label': keys[1], 'amplitude_value': keys[2], 'chromophore': keys[3], 'score_type': keys[4], 'left_pipeline': left.pipeline_label, 'right_pipeline': right.pipeline_label, 'left_minus_right': float(left.target_minus_non_target_score - right.target_minus_non_target_score), 'abs_left_minus_right': float(abs(left.target_minus_non_target_score - right.target_minus_non_target_score)), 'subset_name': subset_name})
        pairwise = pd.DataFrame(pairwise_rows)
        return (summary, pairwise)
    variability_summary_all, pairwise_deltas_all = variability_tables(target_vs_nontarget, 'all_pipelines')
    variability_summary_core, pairwise_deltas_core = variability_tables(target_vs_nontarget, 'core_only', target_vs_nontarget.get('secondary_pipeline', pd.Series(False, index=target_vs_nontarget.index)) == False if len(target_vs_nontarget) > 0 else None)
    variability_summary = pd.concat([df for df in [variability_summary_all, variability_summary_core] if len(df) > 0], ignore_index=True) if len(variability_summary_all) > 0 or len(variability_summary_core) > 0 else pd.DataFrame()
    pairwise_deltas = pd.concat([df for df in [pairwise_deltas_all, pairwise_deltas_core] if len(df) > 0], ignore_index=True) if len(pairwise_deltas_all) > 0 or len(pairwise_deltas_core) > 0 else pd.DataFrame()

    def summarize_shape_subset(df, subset_name):
        if len(df) == 0:
            return pd.DataFrame()
        out = df.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore'], as_index=False).agg(mean_curve_corr=('curve_corr', 'mean'), median_curve_corr=('curve_corr', 'median'), mean_curve_rmse=('curve_rmse', 'mean'), mean_curve_nrmse=('curve_nrmse', 'mean'), mean_peak_latency_error_s=('peak_latency_error_s', 'mean'), mean_peak_amplitude_bias=('peak_amplitude_bias', 'mean'), mean_peak_amplitude_ratio=('peak_amplitude_ratio', 'mean'), mean_auc_bias=('auc_bias', 'mean'), mean_recovered_peak_amplitude=('recovered_peak_amplitude', 'mean'), mean_recovered_auc=('recovered_auc', 'mean'), n_channels=('channel_name', 'count'))
        out['shape_subset'] = subset_name
        return attach_pipeline_metadata(out, config)
    shape_target_primary = summarize_shape_subset(shape_df.loc[(shape_df['target_status'] == 'true_target') & (shape_df['amplitude_value'] > 0)].copy() if len(shape_df) > 0 else pd.DataFrame(), 'true_target_active')
    shape_specificity_secondary = summarize_shape_subset(shape_df.loc[(shape_df['target_status'] == 'true_non_target') & (shape_df['amplitude_value'] > 0)].copy() if len(shape_df) > 0 else pd.DataFrame(), 'true_non_target_active')
    shape_null_summary = summarize_shape_subset(shape_df.loc[shape_df['amplitude_value'] == 0].copy() if len(shape_df) > 0 else pd.DataFrame(), 'null_condition')
    shape_fidelity_summary = pd.concat([df for df in [shape_target_primary, shape_specificity_secondary, shape_null_summary] if len(df) > 0], ignore_index=True) if any((len(df) > 0 for df in [shape_target_primary, shape_specificity_secondary, shape_null_summary])) else pd.DataFrame()
    pipeline_performance_summary = pd.DataFrame()
    if len(target_vs_nontarget) > 0:
        pipeline_performance_summary = target_vs_nontarget.groupby(['file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'score_type'], as_index=False).agg(mean_target_score=('mean_target_score', 'mean'), mean_non_target_score=('mean_non_target_score', 'mean'), mean_target_minus_non_target_score=('target_minus_non_target_score', 'mean'), std_target_minus_non_target_score=('target_minus_non_target_score', 'std'), median_target_minus_non_target_score=('target_minus_non_target_score', 'median'), n_subject_rows=('subject', 'count'))
        pipeline_performance_summary = attach_pipeline_metadata(pipeline_performance_summary, config)
    all_tables = {**combined, 'pipeline_manifest': manifest, 'master_run_log': pd.DataFrame(run_results), 'roi_scores': roi_scores, 'dlpfc_roi_scores': dlpfc_roi_scores, 'target_vs_nontarget_summary': target_vs_nontarget, 'parametric_null_summary': parametric_null_summary, 'empirical_null_pvalues': empirical_null_pvalues, 'variability_summary': variability_summary, 'pairwise_pipeline_deltas': pairwise_deltas, 'shape_fidelity_summary': shape_fidelity_summary, 'pipeline_performance_summary': pipeline_performance_summary, 'config': pd.DataFrame([config])}
    for name, df in all_tables.items():
        maybe_write_table(df, aggregate_dir / name, config['write_csv'], config['write_parquet'])

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default=None)
    parser.add_argument("--filename", default=None)
    parser.add_argument("--pipeline-label", default=None)
    parser.add_argument("--pipeline-labels", default=None, help="Comma-separated pipeline labels")
    parser.add_argument("--output-dirname", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--empirical-null-shift-count", type=int, default=None)
    return parser

if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    config = config_snapshot()

    if args.overwrite:
        config["overwrite"] = True
    if args.empirical_null_shift_count is not None:
        config["empirical_null_shift_count"] = int(args.empirical_null_shift_count)
    if args.output_dirname is not None:
        config["output_dirname"] = str(args.output_dirname)

    ensure_dir(output_path(config))
    ensure_dir(jobs_path(config))
    ensure_dir(aggregate_path(config))

    if not config["subject_names"]:
        config["subject_names"] = [
            subject_dir.name
            for subject_dir in sorted(dataset_path(config).glob("Subj*"))
            if subject_dir.is_dir()
        ]

    if args.subject is not None:
        config["subject_names"] = [subject for subject in config["subject_names"] if subject == args.subject]
        if not config["subject_names"]:
            raise BenchmarkError(f"Requested subject not found: {args.subject}")

    config["file_specs"] = discover_file_specs_by_subject(dataset_path(config), config["subject_names"])

    if args.pipeline_label is not None:
        config["pipeline_specs"] = [
            p for p in config["pipeline_specs"]
            if p["label"] == args.pipeline_label
        ]
        if not config["pipeline_specs"]:
            raise BenchmarkError(f"Unknown pipeline label: {args.pipeline_label}")
    if args.pipeline_labels is not None:
        requested = {x.strip() for x in str(args.pipeline_labels).split(',') if x.strip()}
        config["pipeline_specs"] = [p for p in config["pipeline_specs"] if p["label"] in requested]
        if not config["pipeline_specs"]:
            raise BenchmarkError(f"No pipeline labels matched: {args.pipeline_labels}")

    if args.aggregate_only:
        run_results = []
        for subject in config["subject_names"]:
            for file_spec in config["file_specs"][subject]:
                run_results.append({
                    "subject": subject,
                    "file_label": file_spec["label"],
                    "status": "discovered",
                    "job_dir": str(jobs_path(config) / subject / file_spec["label"]),
                })
        aggregate_outputs(config, run_results)
        print(f"Aggregate outputs written to: {aggregate_path(config)}")
        raise SystemExit(0)

    jobs = []
    for subject in config["subject_names"]:
        for file_spec in config["file_specs"][subject]:
            if args.filename is not None and file_spec["filename"] != args.filename:
                continue
            jobs.append((subject, file_spec))

    print(f"Running {len(jobs)} subject-file jobs")
    run_results = []
    for subject, file_spec in jobs:
        try:
            result = process_subject_file_job(subject, file_spec, config)
            run_results.append(result)
            print(f"[{result['status']}] {subject} {file_spec['label']}")
        except Exception as exc:
            result = {
                "subject": subject,
                "file_label": file_spec["label"],
                "status": "crashed-submit",
                "error": str(exc),
                "job_dir": str(jobs_path(config) / subject / file_spec["label"]),
            }
            run_results.append(result)
            print(f"[crashed-submit] {subject} {file_spec['label']}: {exc}")

    if not args.skip_aggregate:
        aggregate_outputs(config, run_results)
        print(f"Done. Aggregate outputs written to: {aggregate_path(config)}")
