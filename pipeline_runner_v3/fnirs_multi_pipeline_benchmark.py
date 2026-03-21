#!/usr/bin/env python3
"""
Plain, notebook-style rewrite of the semisynthetic fNIRS benchmark.

This is still a normal .py script.
The goal is not to be short. The goal is to be traceable:
- plain settings near the top
- obvious subject/file loops near the bottom
- fewer framework ideas
- easier places to print, inspect, and debug

Scientifically important behavior from the original benchmark is still kept:
- multi-subject runs
- multi-pipeline runs
- QC / preprocessing / nuisance logic
- HRF / solver / FIR / ROI / shape-fidelity logic
- MATLAB AR-IRLS handoff
- per-job outputs and aggregate outputs
"""

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


# -----------------------------------------------------------------------------
# Edit these settings first
# -----------------------------------------------------------------------------

ROOT = str(Path.home() / "fnirs-representation-learning")
DATASET_DIRNAME = "snirf_dataset_2"
OUTPUT_DIRNAME = "benchmark_outputs"
OUTPUT_PREFIX = "alpha_v3"

# Leave this empty to run every Subj* folder in the dataset directory.
SUBJECT_NAMES = ["Subj94", "Subj100"]

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
PPF_VALUE = 0.1
STIM_DURATION_S = 1.0
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

EMPIRICAL_NULL_SHIFT_COUNT = 1
EMPIRICAL_NULL_MIN_SHIFT_S = 20.0

OVERWRITE = False
WRITE_CSV = True
WRITE_PARQUET = True

# Set this to False if you want to skip MATLAB/AnalyzIR pipelines.
USE_MATLAB = True
USE_MATLAB_ENGINE = True
PREFER_MATLAB_ENGINE = True
MATLAB_CMD = "matlab"
MATLAB_TIMEOUT_S = 7200
MATLAB_STARTUP_OPTIONS = ""
ANALYZIR_PATH = "/home/asunkari/nirs-toolbox"

TRUTH_TEMPLATE_DIR = "/home/asunkari/fnirs-representation-learning/synthetic_hrf_generation"
TRUTH_TEMPLATE_MAP = {
    "hrf_20": "hrf_20.mat",
    "hrf_50": "hrf_50.mat",
    "hrf_100": "hrf_100.mat",
}

FILE_SPECS = [
    {
        "label": "no_hrf",
        "filename": "resting_clean.snirf",
        "amplitude_value": 0,
        "is_null": True,
        "annotation_source_filename": "resting_hrf_20.snirf",
    },
    {
        "label": "hrf_20",
        "filename": "resting_hrf_20.snirf",
        "amplitude_value": 20,
        "is_null": False,
        "annotation_source_filename": None,
    },
    {
        "label": "hrf_50",
        "filename": "resting_hrf_50.snirf",
        "amplitude_value": 50,
        "is_null": False,
        "annotation_source_filename": None,
    },
    {
        "label": "hrf_100",
        "filename": "resting_hrf_100.snirf",
        "amplitude_value": 100,
        "is_null": False,
        "annotation_source_filename": None,
    },
]

PIPELINE_SPECS = [
    {
        "label": "NoSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "none",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> NoSS -> Glover -> AUTO",
    },
    {
        "label": "LocalSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> Glover -> AUTO",
    },
    {
        "label": "PooledPCA2_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "pooled_pca2",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> PooledPCA2 -> Glover -> AUTO",
    },
    {
        "label": "SSAuxPCA_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "ss_aux_pca",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> SSAuxPCA -> Glover -> AUTO",
    },
    {
        "label": "MultiSSOrth3_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "multi_ss_orth3",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> MultiSSOrth3 -> Glover -> AUTO",
    },
    {
        "label": "NoSS_Glover_OLS",
        "backend": "python",
        "nuisance_method": "none",
        "hrf_model": "glover",
        "solver": "ols",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> NoSS -> Glover -> OLS",
    },
    {
        "label": "NoSS_Glover_ARIRLS",
        "backend": "matlab_arirls",
        "nuisance_method": "none",
        "hrf_model": "glover",
        "solver": "arirls",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> NoSS -> Glover -> AR-IRLS",
    },
    {
        "label": "LocalSS_Glover_OLS",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "ols",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> Glover -> OLS",
    },
    {
        "label": "LocalSS_Glover_ARIRLS",
        "backend": "matlab_arirls",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "arirls",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> Glover -> AR-IRLS",
    },
    {
        "label": "LocalSS_SPM_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "spm",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> SPM -> AUTO",
    },
    {
        "label": "LocalSS_SPM_OLS",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "spm",
        "solver": "ols",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> SPM -> OLS",
    },
    {
        "label": "LocalSS_FIR_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "fir",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> FIR -> AUTO",
    },
    {
        "label": "LocalSS_Gamma_ARIRLS",
        "backend": "matlab_arirls",
        "nuisance_method": "local_nearest",
        "hrf_model": "gamma",
        "solver": "arirls",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": False,
        "comparison_group": "core",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> Gamma -> AR-IRLS",
    },
    {
        "label": "LooseQC_LocalSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "loose_sci",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": True,
        "comparison_group": "secondary",
        "description": "LooseQC -> TDDR -> band-pass -> LocalSS -> Glover -> AUTO",
    },
    {
        "label": "WaveletMC_LocalSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "wavelet",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": True,
        "comparison_group": "secondary",
        "description": "StrictQC -> WaveletMC -> band-pass -> LocalSS -> Glover -> AUTO",
    },
    {
        "label": "HighPassOnly_LocalSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "highpass_only",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": True,
        "comparison_group": "secondary",
        "description": "StrictQC -> TDDR -> high-pass-only -> LocalSS -> Glover -> AUTO",
    },
    {
        "label": "NoMotion_LocalSS_Glover_AUTO",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "glover",
        "solver": "auto",
        "pruning_style": "strict_combined",
        "motion_method": "none",
        "filter_mode": "bandpass",
        "use_block_average": False,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": True,
        "comparison_group": "secondary",
        "description": "StrictQC -> no motion correction -> band-pass -> LocalSS -> Glover -> AUTO",
    },
    {
        "label": "BlockAvg_LocalSS",
        "backend": "python",
        "nuisance_method": "local_nearest",
        "hrf_model": "block_average",
        "solver": "none",
        "pruning_style": "strict_combined",
        "motion_method": "tddr",
        "filter_mode": "bandpass",
        "use_block_average": True,
        "include_in_empirical_null": True,
        "include_in_primary_variability": True,
        "secondary_pipeline": True,
        "comparison_group": "secondary",
        "description": "StrictQC -> TDDR -> band-pass -> LocalSS -> block average",
    },
]


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
        "truth_template_dir": TRUTH_TEMPLATE_DIR,
        "truth_template_map": dict(TRUTH_TEMPLATE_MAP),
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
    templates = {}
    if config['truth_template_dir'] is None:
        return templates
    template_dir = Path(config['truth_template_dir']).expanduser().resolve()
    for file_label, template_name in config['truth_template_map'].items():
        template_path = template_dir / template_name
        if not template_path.exists():
            continue
        mat = scipy.io.loadmat(template_path, squeeze_me=True, struct_as_record=False)
        if 'hrf' not in mat:
            continue
        hrf = mat['hrf']
        hrf_conc = np.asarray(hrf.hrf_conc, dtype=float)
        templates[file_label] = {'time_s': np.asarray(hrf.t_hrf, dtype=float).reshape(-1), 'hbo': hrf_conc[:, 0].reshape(-1), 'hbr': hrf_conc[:, 1].reshape(-1), 'hbt': hrf_conc[:, 2].reshape(-1)}
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

def build_hb_channel_table(raw_hb, subject, file_label, config):
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
    df = pd.DataFrame({'subject': subject, 'file_label': file_label, 'channel_name': hb_names, 'pair_name': pair_names, 'chromophore': hb_types, 'distance_m': distances_hb, 'midpoint_x': positions[:, 0], 'midpoint_y': positions[:, 1], 'midpoint_z': positions[:, 2], 'is_ss': short_mask, 'is_ls': long_mask})
    df['hemisphere'] = np.where(df['midpoint_x'] < 0, 'left', np.where(df['midpoint_x'] > 0, 'right', 'midline'))
    df['group'] = np.select([df['is_ss'], df['is_ls']], ['SS', 'LS'], default='MID')
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
    idx = raw.ch_names.index(channel_name)
    return np.asarray(raw.info['chs'][idx]['loc'][:3], dtype=float)

def get_available_long_channel_names(raw_hb, config):
    hb_table = build_hb_channel_table(raw_hb, '', '', config)
    long_names = hb_table.loc[hb_table['group'] == 'LS', 'channel_name'].astype(str).tolist()
    return sorted([name for name in long_names if name not in raw_hb.info['bads']])

def get_available_short_channel_names(raw_hb, chromophore, config):
    hb_table = build_hb_channel_table(raw_hb, '', '', config)
    short_names = hb_table.loc[(hb_table['group'] == 'SS') & (hb_table['chromophore'] == chromophore), 'channel_name'].astype(str).tolist()
    return [name for name in short_names if name not in raw_hb.info['bads']]

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

def gamma_hrf_function(t_r, oversampling=50):
    peak_time = 6.0
    peak_disp = 1.0
    duration = 32.0
    dt = float(t_r) / float(oversampling)
    t = np.arange(0.0, duration + dt, dt)
    h = peak_disp ** peak_time * t ** (peak_time - 1.0) * np.exp(-peak_disp * t) / math.gamma(peak_time)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    total = h.sum()
    return h / total if total > 0 else h

def analyzir_canonical_hrf(times_s):
    a1, a2, b1, b2, c = (4.0, 16.0, 1.0, 1.0, 1.0 / 6.0)
    t = np.clip(times_s, 0.0, None)
    h = b1 ** a1 * t ** (a1 - 1.0) * np.exp(-b1 * t) / math.gamma(a1)
    h -= c * b2 ** a2 * t ** (a2 - 1.0) * np.exp(-b2 * t) / math.gamma(a2)
    total = np.sum(h)
    return h / total if abs(total) > 0 else h

def get_truth_curve(templates, file_label, epoch_times_s, chromophore):
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    if file_label == 'no_hrf':
        return np.zeros_like(epoch_times_s, dtype=float)
    if file_label not in templates:
        return np.full_like(epoch_times_s, np.nan, dtype=float)
    tmpl = templates[file_label]
    return np.interp(epoch_times_s, tmpl['time_s'], tmpl[chromophore], left=0.0, right=0.0)

def build_basis_curve(epoch_times_s, hrf_model, fir_delays=None, fir_betas=None):
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)
    compute_regressor, _ = import_nilearn_functions()
    if hrf_model in {'glover', 'spm'}:
        signal, _ = compute_regressor(np.array([[0.0], [1.0], [1.0]]), hrf_model, epoch_times_s, con_id='task', oversampling=50)
        return signal[:, 0]
    if hrf_model == 'gamma':
        dt = np.median(np.diff(epoch_times_s))
        signal, _ = compute_regressor(np.array([[0.0], [1.0], [1.0]]), [gamma_hrf_function], epoch_times_s, con_id='task', oversampling=50)
        return signal[:, 0]
    if hrf_model == 'canonical':
        return analyzir_canonical_hrf(epoch_times_s)
    if hrf_model == 'fir':
        if fir_delays is None or fir_betas is None:
            raise ValueError('FIR basis requires delays and betas.')
        return np.interp(epoch_times_s, np.asarray(fir_delays, dtype=float), np.asarray(fir_betas, dtype=float), left=0.0, right=0.0)
    raise ValueError(f'Unknown hrf_model {hrf_model!r}')

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
    add_regs = None if nuisance_df is None else nuisance_df
    add_names = None
    if hrf_model in {'glover', 'spm', 'fir'}:
        fir_delays = config['fir_delays_scans'] if hrf_model == 'fir' else None
        return make_first_level_design_matrix(raw_hb, stim_dur=config['stim_duration_s'], hrf_model=hrf_model, drift_model='cosine', high_pass=config['drift_high_pass_hz'], fir_delays=fir_delays, add_regs=add_regs, add_reg_names=add_names, oversampling=1 if hrf_model == 'fir' else 50)
    if hrf_model == 'gamma':
        _, nilearn_make_dm = import_nilearn_functions()
        events = pd.DataFrame({'trial_type': list(raw_hb.annotations.description), 'onset': raw_hb.annotations.onset - raw_hb.first_time, 'duration': config['stim_duration_s'] * np.ones(len(raw_hb.annotations))})
        return nilearn_make_dm(frame_times=raw_hb.times, events=events, drift_model='cosine', high_pass=config['drift_high_pass_hz'], hrf_model=[gamma_hrf_function], add_regs=add_regs, add_reg_names=None if nuisance_df is None else list(nuisance_df.columns), oversampling=50)
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
    task_names = set(get_task_regressor_names(design_matrix))
    rows = []
    for _, row in glm_df.iterrows():
        condition = str(row[cond_col])
        if condition not in task_names and (not any((condition.startswith(name) for name in task_names))):
            continue
        channel_name = str(row[ch_col]) if ch_col is not None else None
        if channel_name is None or channel_name not in channel_names:
            continue
        pair_name = channel_name.split(' ')[0]
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        out = {'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'hrf_model': pipeline['hrf_model'], 'solver': pipeline['solver'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target', 'task_regressor': condition, 'beta': float(row[beta_col]), **nuisance_metadata_by_channel.get(channel_name, {})}
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

def parse_fir_dataframe(glm_df, design_matrix, channel_names, subject, file_label, amplitude_value, pipeline, target_pair_names, nuisance_metadata_by_channel):
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
        rows.append({'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'hrf_model': pipeline['hrf_model'], 'solver': pipeline['solver'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': 'true_target' if pair_name in target_pair_names else 'true_non_target', 'fir_regressor': condition, 'delay_scan': delay_scan, 'delay_s': delay_scan / 1.0, 'beta': float(row[beta_col]), **nuisance_metadata_by_channel.get(channel_name, {})})
    return pd.DataFrame(rows)

def summarize_fir_shapes(fir_df, file_label, pipeline, truth_templates, config):
    epoch_times = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, 1.0 / config['fir_resample_sfreq_hz'])
    shape_rows, roi_rows = ([], [])
    for channel_name, group in fir_df.groupby('channel_name'):
        chrom = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        pair_name = channel_name.split(' ')[0]
        group = group.sort_values('delay_s')
        recovered = build_basis_curve(epoch_times, 'fir', fir_delays=group['delay_s'].tolist(), fir_betas=group['beta'].to_numpy())
        truth_curve = get_truth_curve(truth_templates, file_label, epoch_times, chrom)
        shape_rows.append({'subject': group['subject'].iloc[0], 'file_label': file_label, 'amplitude_value': int(group['amplitude_value'].iloc[0]), 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'channel_name': channel_name, 'pair_name': pair_name, 'chromophore': chrom, 'target_status': group['target_status'].iloc[0], 'shape_source': 'fir_beta_curve', **compute_shape_metrics(recovered, truth_curve, epoch_times, chrom)})
    for (chrom, target_status), group in fir_df.groupby(['chromophore', 'target_status']):
        curve = group.groupby('delay_s', as_index=False)['beta'].mean().sort_values('delay_s')
        roi_curve = build_basis_curve(epoch_times, 'fir', fir_delays=curve['delay_s'].tolist(), fir_betas=curve['beta'].to_numpy())
        for t_s, y in zip(epoch_times, roi_curve):
            roi_rows.append({'subject': group['subject'].iloc[0], 'file_label': file_label, 'amplitude_value': int(group['amplitude_value'].iloc[0]), 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': chrom, 'target_status': target_status, 'curve_source': 'fir_roi_mean_curve', 'time_s': float(t_s), 'signal': float(y)})
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
        scipy.io.savemat(in_path, {'times_s': np.asarray(spec['times_s'], dtype=float), 'stim_onsets_s': np.asarray(spec['stim_onsets_s'], dtype=float), 'stim_durations_s': np.asarray(spec['stim_durations_s'], dtype=float), 'stim_amplitudes': np.asarray(spec['stim_amplitudes'], dtype=float), 'Y': np.asarray(spec['Y'], dtype=float), 'X': np.asarray(spec['X'], dtype=float), 'n_reg': np.asarray(spec['n_reg'], dtype=np.int32), 'task_reg_index': np.asarray(spec['task_reg_index'], dtype=np.int32), 'nuisance_values': np.asarray(spec['nuisance_values'], dtype=float), 'nuisance_n_reg': np.asarray(spec['nuisance_n_reg'], dtype=np.int32), 'nuisance_names': np.asarray(spec['nuisance_names'], dtype=object), 'matlab_resample_fs': np.asarray(spec['matlab_resample_fs'], dtype=float), 'channel_names': np.asarray(spec['channel_names'], dtype=object), 'pair_names': np.asarray(spec['pair_names'], dtype=object), 'chromophores': np.asarray(spec['chromophores'], dtype=object), 'target_status': np.asarray(spec['target_status'], dtype=object), 'subject': np.asarray([spec['subject']], dtype=object), 'file_label': np.asarray([spec['file_label']], dtype=object), 'pipeline_label': np.asarray([spec['pipeline_label']], dtype=object), 'backend': np.asarray([spec['backend']], dtype=object), 'hrf_model': np.asarray([spec['hrf_model']], dtype=object), 'solver': np.asarray([spec['solver']], dtype=object), 'amplitude_value': np.asarray([[spec['amplitude_value']]], dtype=np.int32), 'shift_index': np.asarray([[spec['shift_index']]], dtype=np.int32), 'shift_s': np.asarray([[spec['shift_s']]], dtype=float)}, do_compression=True)
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
    env = os.environ.copy()
    env['FNIRS_BUNDLE_JSON'] = str(bundle_json_path)
    if config['analyzir_path']:
        env['FNIRS_ANALYZIR_PATH'] = str(config['analyzir_path'])
    helper_dir = helper_m.parent.as_posix()
    helper_name = helper_m.stem
    command = [config['matlab_cmd'], '-batch', f"try, addpath('{helper_dir}'); {helper_name}; catch ME, disp(getReport(ME,'extended')); exit(1); end; exit(0);"]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=config['matlab_timeout_s'])
    (job_dir / 'matlab_batch.log').write_text(f"STDOUT\n{'=' * 80}\n{result.stdout}\n\nSTDERR\n{'=' * 80}\n{result.stderr}\n", encoding='utf-8')
    if result.returncode != 0:
        raise BenchmarkError(f"MATLAB sidecar failed; see {job_dir / 'matlab_batch.log'}")
    return load_matlab_outputs_from_bundle(bundle_json_path)

def run_matlab_sidecar_engine(config, bundle_json_path, job_dir, helper_m):
    _, matlab_engine = optional_import_matlab_engine()
    if matlab_engine is None:
        raise BenchmarkError('MATLAB Engine for Python is not installed in this environment.')
    helper_dir = helper_m.parent.as_posix()
    eng = matlab_engine.start_matlab(config['matlab_startup_options'])
    try:
        eng.addpath(helper_dir, nargout=0)
        eng.setenv('FNIRS_BUNDLE_JSON', str(bundle_json_path), nargout=0)
        if config['analyzir_path']:
            eng.setenv('FNIRS_ANALYZIR_PATH', str(config['analyzir_path']), nargout=0)
        helper_visible = eng.eval("which('analyzir_arirls_batch')", nargout=1)
        glm_visible = eng.eval("which('nirs.modules.GLM')", nargout=1)
        resample_visible = eng.eval("which('nirs.modules.Resample')", nargout=1)
        arirls_visible = eng.eval("which('nirs.modules.AR_IRLS')", nargout=1)
        (job_dir / 'matlab_engine_preflight.log').write_text(f'helper_visible: {helper_visible}\nglm_visible: {glm_visible}\nresample_visible: {resample_visible}\narirls_visible: {arirls_visible}\n', encoding='utf-8')
        matlab_output = eng.evalc('analyzir_arirls_batch', nargout=1)
        (job_dir / 'matlab_engine_output.log').write_text(matlab_output, encoding='utf-8')
    except Exception as exc:
        (job_dir / 'matlab_engine.log').write_text(str(exc), encoding='utf-8')
        raise BenchmarkError(f"MATLAB engine execution failed; see {job_dir / 'matlab_engine.log'}") from exc
    finally:
        try:
            eng.quit()
        except Exception:
            pass
    return load_matlab_outputs_from_bundle(bundle_json_path)

def run_matlab_sidecar(config, bundle_json_path, job_dir):
    if not config['use_matlab']:
        return pd.DataFrame()
    helper_m = Path(__file__).with_name('analyzir_arirls_batch.m')
    if not helper_m.exists():
        raise BenchmarkError(f'Missing MATLAB helper: {helper_m}')
    matlab_mod, matlab_engine = optional_import_matlab_engine()
    engine_available = matlab_engine is not None
    use_engine = bool(config['use_matlab_engine'] or config['prefer_matlab_engine'])
    if use_engine and engine_available:
        try:
            return run_matlab_sidecar_engine(config, bundle_json_path, job_dir, helper_m)
        except Exception as exc:
            (job_dir / 'matlab_engine_fallback.log').write_text(str(exc), encoding='utf-8')
            raise
    if config['use_matlab_engine'] and (not engine_available):
        raise BenchmarkError('MATLAB Engine for Python was requested but is not installed in this environment.')
    return run_matlab_sidecar_batch(config, bundle_json_path, job_dir, helper_m)

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

    This is the simpler, more notebook-like version of the old bundle class
    layer. Each entry is a plain dictionary with:
        {
            "design_df": DataFrame or None,
            "metadata": {...}
        }
    """
    nuisance_by_chromophore = {}
    auxiliary_df = read_auxiliary_dataframe(snirf_file_path, raw_cw)
    auxiliary_matrix = auxiliary_df.to_numpy().T if len(auxiliary_df.columns) > 0 else None
    for chromophore in ['hbo', 'hbr']:
        if nuisance_method == 'none':
            nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none'}}
            continue
        if nuisance_method == 'pooled_pca2':
            short_channel_names = get_available_short_channel_names(raw_hb, chromophore, config)
            if not short_channel_names:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_no_short_channels'}}
                continue
            short_channel_matrix = raw_hb.copy().pick(short_channel_names).get_data()
            nuisance_components = principal_components_rows(short_channel_matrix, config['pooled_ss_n_components'])
            if nuisance_components is None:
                nuisance_by_chromophore[chromophore] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_failed_pca'}}
                continue
            nuisance_df = pd.DataFrame(nuisance_components, columns=[f'ss_pc{i + 1}_{chromophore}' for i in range(nuisance_components.shape[1])])
            nuisance_by_chromophore[chromophore] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'pooled_pca2', 'n_short_channels_used': len(short_channel_names), 'n_nuisance_components': nuisance_components.shape[1]}}
            continue
        if nuisance_method == 'ss_aux_pca':
            stacked_blocks = []
            short_channel_names = get_available_short_channel_names(raw_hb, chromophore, config)
            if short_channel_names:
                stacked_blocks.append(raw_hb.copy().pick(short_channel_names).get_data())
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
    return nuisance_by_chromophore

def build_channel_nuisance_regressors(raw_hb, nuisance_method, config):
    """
    Build one nuisance regressor set per long-separation channel.
    """
    nuisance_by_channel = {}
    for long_channel_name in get_available_long_channel_names(raw_hb, config):
        chromophore = 'hbo' if long_channel_name.endswith('hbo') else 'hbr'
        if nuisance_method == 'none':
            nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none'}}
            continue
        short_channel_names = get_available_short_channel_names(raw_hb, chromophore, config)
        if not short_channel_names:
            nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_no_short_channels'}}
            continue
        long_channel_position = get_channel_position(raw_hb, long_channel_name)
        short_channel_distances = []
        for short_channel_name in short_channel_names:
            short_channel_position = get_channel_position(raw_hb, short_channel_name)
            distance_m = float(np.linalg.norm(long_channel_position - short_channel_position))
            short_channel_distances.append((short_channel_name, distance_m))
        short_channel_distances.sort(key=lambda item: item[1])
        if nuisance_method == 'local_nearest':
            nearest_short_name, nearest_short_distance_m = short_channel_distances[0]
            if nearest_short_distance_m <= config['local_ss_max_distance_m']:
                nuisance_trace = raw_hb.copy().pick([nearest_short_name]).get_data()[0]
                nuisance_df = pd.DataFrame({f'ss_local_{chromophore}': nuisance_trace})
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'nearest_short_channel', 'nuisance_regressor_label': nearest_short_name, 'nearest_short_distance_m': nearest_short_distance_m, 'n_short_channels_used': 1}}
            else:
                fallback_trace = raw_hb.copy().pick(short_channel_names).get_data().mean(axis=0)
                nuisance_df = pd.DataFrame({f'ss_pooled_fallback_{chromophore}': fallback_trace})
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'pooled_fallback_average', 'nuisance_regressor_label': 'pooled_fallback_average', 'nearest_short_distance_m': nearest_short_distance_m, 'n_short_channels_used': len(short_channel_names)}}
            continue
        if nuisance_method == 'multi_ss_orth3':
            selected_rows = short_channel_distances[:max(1, min(config['multi_ss_k'], len(short_channel_distances)))]
            selected_short_names = [name for name, _ in selected_rows]
            selected_short_matrix = raw_hb.copy().pick(selected_short_names).get_data().T
            selected_short_matrix = (selected_short_matrix - selected_short_matrix.mean(axis=0, keepdims=True)) / np.maximum(selected_short_matrix.std(axis=0, keepdims=True), 1e-12)
            orthogonal_components = qr_orth_columns(selected_short_matrix)
            if orthogonal_components.size == 0:
                nuisance_by_channel[long_channel_name] = {'design_df': None, 'metadata': {'nuisance_method_used': 'none_failed_qr'}}
            else:
                nuisance_df = pd.DataFrame(orthogonal_components, columns=[f'ss_qr{i + 1}_{chromophore}' for i in range(orthogonal_components.shape[1])])
                nuisance_by_channel[long_channel_name] = {'design_df': nuisance_df, 'metadata': {'nuisance_method_used': 'multi_ss_orth3', 'n_short_channels_used': len(selected_short_names), 'n_nuisance_components': orthogonal_components.shape[1], 'nearest_short_distance_m': selected_rows[0][1], 'short_regressor_labels': '|'.join(selected_short_names)}}
            continue
        raise ValueError(f'Unsupported channel-specific nuisance_method {nuisance_method!r}')
    return nuisance_by_channel

def apply_channelwise_nuisance_regression(raw_hb, nuisance_by_channel):
    """
    Time-domain nuisance regression for the block-average branch.
    """
    denoised_raw_hb = raw_hb.copy().load_data()
    nuisance_detail_rows = []
    for channel_name, nuisance_info in nuisance_by_channel.items():
        nuisance_df = nuisance_info['design_df']
        nuisance_metadata = nuisance_info['metadata']
        chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        nuisance_detail_rows.append({'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
        if nuisance_df is None or nuisance_df.shape[1] == 0:
            continue
        design_matrix = np.column_stack([np.ones(len(denoised_raw_hb.times)), nuisance_df.to_numpy()])
        signal_vector = denoised_raw_hb.copy().pick([channel_name]).get_data()[0]
        regression_coefficients = np.linalg.lstsq(design_matrix, signal_vector, rcond=None)[0]
        fitted_signal = design_matrix @ regression_coefficients
        cleaned_signal = signal_vector - fitted_signal + regression_coefficients[0]
        denoised_raw_hb._data[denoised_raw_hb.ch_names.index(channel_name), :] = cleaned_signal
    return (denoised_raw_hb, pd.DataFrame(nuisance_detail_rows))

def collect_canonical_shape_metrics(channel_metrics_df, epoch_times_s, file_label, pipeline, truth_templates):
    if channel_metrics_df is None or len(channel_metrics_df) == 0:
        return []
    shape_metric_rows = []
    basis_curve_cache = {}
    for _, metric_row in channel_metrics_df.iterrows():
        chromophore = metric_row['chromophore']
        if pipeline['hrf_model'] not in basis_curve_cache:
            basis_curve_cache[pipeline['hrf_model']] = build_basis_curve(epoch_times_s, pipeline['hrf_model'])
        recovered_curve = float(metric_row['beta']) * basis_curve_cache[pipeline['hrf_model']]
        truth_curve = get_truth_curve(truth_templates, file_label, epoch_times_s, chromophore)
        shape_metric_rows.append({'subject': metric_row['subject'], 'file_label': metric_row['file_label'], 'amplitude_value': metric_row['amplitude_value'], 'pipeline_label': metric_row['pipeline_label'], 'backend': metric_row['backend'], 'channel_name': metric_row['channel_name'], 'pair_name': metric_row['pair_name'], 'chromophore': chromophore, 'target_status': metric_row['target_status'], 'shape_source': f"beta_scaled_{pipeline['hrf_model']}", **compute_shape_metrics(recovered_curve, truth_curve, epoch_times_s, chromophore)})
    return shape_metric_rows

def collect_canonical_roi_timecourses(channel_metrics_df, epoch_times_s, pipeline):
    if channel_metrics_df is None or len(channel_metrics_df) == 0:
        return []
    roi_timecourse_rows = []
    basis_curve = build_basis_curve(epoch_times_s, pipeline['hrf_model'])
    for (chromophore, target_status), group_df in channel_metrics_df.groupby(['chromophore', 'target_status']):
        roi_curve = float(group_df['beta'].mean()) * basis_curve
        for time_s, signal_value in zip(epoch_times_s, roi_curve):
            roi_timecourse_rows.append({'subject': group_df['subject'].iloc[0], 'file_label': group_df['file_label'].iloc[0], 'amplitude_value': int(group_df['amplitude_value'].iloc[0]), 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': chromophore, 'target_status': target_status, 'curve_source': f"roi_mean_beta_scaled_{pipeline['hrf_model']}", 'time_s': float(time_s), 'signal': float(signal_value)})
    return roi_timecourse_rows

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
            truth_curve = get_truth_curve(truth_templates, file_spec['label'], epochs_hb.times, chromophore)
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

    The logic is explicit here instead of hidden behind multiple small wrappers:
      - decide whether nuisance regressors are global or channel-specific
      - decide whether the HRF model is canonical-style or FIR
      - fit GLMs
      - convert fits into per-channel, shape, ROI, and nuisance tables
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
    uses_global_nuisance = pipeline['nuisance_method'] in {'none', 'pooled_pca2', 'ss_aux_pca'}
    if uses_global_nuisance:
        nuisance_by_chromophore = build_global_nuisance_regressors(raw_hb, snirf_file_path, raw_cw, pipeline['nuisance_method'], config)
        if pipeline['hrf_model'] == 'fir':
            raw_fir = resample_for_fir(raw_hb, config)
            fir_long_channel_names = get_available_long_channel_names(raw_fir, config)
            for chromophore in ['hbo', 'hbr']:
                nuisance_info = nuisance_by_chromophore[chromophore]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                channel_subset = [name for name in fir_long_channel_names if name.endswith(chromophore)]
                for channel_name in channel_subset:
                    nuisance_detail_rows.append({'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
                if not channel_subset:
                    continue
                nuisance_df_fir = resample_nuisance_dataframe_to_fir_times(nuisance_df, raw_hb.times, raw_fir.times)
                design_matrix = build_design_matrix(raw_fir, 'fir', nuisance_df_fir, config)
                glm_result = run_glm(raw_fir.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                fir_metric_df = parse_fir_dataframe(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {name: nuisance_metadata for name in channel_subset})
                if len(fir_metric_df) > 0:
                    fir_metric_frames.append(fir_metric_df)
        else:
            epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
            for chromophore in ['hbo', 'hbr']:
                nuisance_info = nuisance_by_chromophore[chromophore]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                channel_subset = [name for name in long_channel_names if name.endswith(chromophore)]
                for channel_name in channel_subset:
                    nuisance_detail_rows.append({'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
                if not channel_subset:
                    continue
                design_matrix = build_design_matrix(raw_hb, pipeline['hrf_model'], nuisance_df, config)
                glm_result = run_glm(raw_hb.copy().pick(channel_subset), design_matrix, noise_model=pipeline['solver'])
                channel_metric_df = parse_mne_glm_to_channel_rows(glm_result.to_dataframe(), design_matrix, channel_subset, subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {name: nuisance_metadata for name in channel_subset})
                if len(channel_metric_df) == 0:
                    continue
                canonical_metric_frames.append(channel_metric_df)
                shape_metric_rows.extend(collect_canonical_shape_metrics(channel_metric_df, epoch_times_s, file_spec['label'], pipeline, truth_templates))
                roi_timecourse_rows.extend(collect_canonical_roi_timecourses(channel_metric_df, epoch_times_s, pipeline))
    else:
        channel_nuisance = build_channel_nuisance_regressors(raw_hb, pipeline['nuisance_method'], config)
        if pipeline['hrf_model'] == 'fir':
            raw_fir = resample_for_fir(raw_hb, config)
            fir_long_channel_names = get_available_long_channel_names(raw_fir, config)
            fir_channel_nuisance = build_channel_nuisance_regressors(raw_fir, pipeline['nuisance_method'], config)
            for channel_name in fir_long_channel_names:
                chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
                nuisance_info = fir_channel_nuisance[channel_name]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                nuisance_detail_rows.append({'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
                design_matrix = build_design_matrix(raw_fir, 'fir', nuisance_df, config)
                glm_result = run_glm(raw_fir.copy().pick([channel_name]), design_matrix, noise_model=pipeline['solver'])
                fir_metric_df = parse_fir_dataframe(glm_result.to_dataframe(), design_matrix, [channel_name], subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {channel_name: nuisance_metadata})
                if len(fir_metric_df) > 0:
                    fir_metric_frames.append(fir_metric_df)
        else:
            epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
            for channel_name in long_channel_names:
                chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
                nuisance_info = channel_nuisance[channel_name]
                nuisance_df = nuisance_info['design_df']
                nuisance_metadata = nuisance_info['metadata']
                nuisance_detail_rows.append({'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
                design_matrix = build_design_matrix(raw_hb, pipeline['hrf_model'], nuisance_df, config)
                glm_result = run_glm(raw_hb.copy().pick([channel_name]), design_matrix, noise_model=pipeline['solver'])
                channel_metric_df = parse_mne_glm_to_channel_rows(glm_result.to_dataframe(), design_matrix, [channel_name], subject, file_spec['label'], file_spec['amplitude_value'], pipeline, target_pair_names, {channel_name: nuisance_metadata})
                if len(channel_metric_df) == 0:
                    continue
                canonical_metric_frames.append(channel_metric_df)
            canonical_metric_df = pd.concat(canonical_metric_frames, ignore_index=True) if canonical_metric_frames else pd.DataFrame()
            shape_metric_rows.extend(collect_canonical_shape_metrics(canonical_metric_df, epoch_times_s, file_spec['label'], pipeline, truth_templates))
            roi_timecourse_rows.extend(collect_canonical_roi_timecourses(canonical_metric_df, epoch_times_s, pipeline))
            return {'canonical_channel_metrics': canonical_metric_df, 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': pd.DataFrame(), 'shape_metrics': pd.DataFrame(shape_metric_rows), 'roi_timecourses': pd.DataFrame(roi_timecourse_rows), 'nuisance_detail': pd.DataFrame(nuisance_detail_rows), 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}
    canonical_metric_df = pd.concat(canonical_metric_frames, ignore_index=True) if canonical_metric_frames else pd.DataFrame()
    fir_metric_df = pd.concat(fir_metric_frames, ignore_index=True) if fir_metric_frames else pd.DataFrame()
    if len(fir_metric_df) > 0:
        fir_shape_rows, fir_roi_rows = summarize_fir_shapes(fir_metric_df, file_spec['label'], pipeline, truth_templates, config)
        shape_metric_rows.extend(fir_shape_rows)
        roi_timecourse_rows.extend(fir_roi_rows)
    return {'canonical_channel_metrics': canonical_metric_df, 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': fir_metric_df, 'shape_metrics': pd.DataFrame(shape_metric_rows), 'roi_timecourses': pd.DataFrame(roi_timecourse_rows), 'nuisance_detail': pd.DataFrame(nuisance_detail_rows), 'matlab_input_specs_list': [], 'matlab_shift_specs_list': []}

def build_matlab_input_spec(subject, file_label, amplitude_value, pipeline, raw_hb, nuisance_by_channel, target_pair_names, shift_index, shift_s, config):
    long_channel_names = get_available_long_channel_names(raw_hb, config)
    if not long_channel_names:
        return None
    stimulus_onsets_s = np.asarray(raw_hb.annotations.onset, dtype=float)
    stimulus_durations_s = np.asarray(raw_hb.annotations.duration, dtype=float)
    if stimulus_onsets_s.size == 0:
        return None
    stimulus_amplitudes = np.ones_like(stimulus_onsets_s, dtype=float)
    channel_names = []
    pair_names = []
    chromophores = []
    target_status_labels = []
    response_vectors = []
    design_matrices = []
    design_column_counts = []
    task_regressor_indices = []
    nuisance_detail_rows = []
    nuisance_names_per_channel = []
    nuisance_values_per_channel = []
    nuisance_column_counts = []
    for channel_name in long_channel_names:
        chromophore = 'hbo' if channel_name.endswith('hbo') else 'hbr'
        nuisance_info = nuisance_by_channel[channel_name]
        nuisance_df = nuisance_info['design_df']
        nuisance_metadata = nuisance_info['metadata']
        nuisance_detail_rows.append({'subject': subject, 'file_label': file_label, 'pipeline_label': pipeline['label'], 'channel_name': channel_name, 'chromophore': chromophore, **nuisance_metadata})
        design_matrix = build_design_matrix(raw_hb, pipeline['hrf_model'], nuisance_df, config)
        task_regressor_names = get_task_regressor_names(design_matrix)
        if not task_regressor_names:
            continue
        task_regressor_index = list(design_matrix.columns).index(task_regressor_names[0]) + 1
        response_vector = raw_hb.copy().pick([channel_name]).get_data()[0]
        channel_names.append(channel_name)
        pair_names.append(channel_name.split(' ')[0])
        chromophores.append(chromophore)
        target_status_labels.append('true_target' if channel_name.split(' ')[0] in target_pair_names else 'true_non_target')
        response_vectors.append(response_vector)
        design_matrices.append(design_matrix.to_numpy())
        design_column_counts.append(design_matrix.shape[1])
        task_regressor_indices.append(task_regressor_index)
        if nuisance_df is None or nuisance_df.shape[1] == 0:
            nuisance_names_per_channel.append([])
            nuisance_values_per_channel.append(np.zeros((len(raw_hb.times), 0), dtype=float))
            nuisance_column_counts.append(0)
        else:
            nuisance_names_per_channel.append([str(column_name) for column_name in nuisance_df.columns])
            nuisance_values_per_channel.append(nuisance_df.to_numpy(dtype=float))
            nuisance_column_counts.append(nuisance_df.shape[1])
    if not channel_names:
        return None
    n_timepoints = len(raw_hb.times)
    n_channels = len(channel_names)
    max_design_columns = max(design_column_counts)
    design_tensor = np.zeros((n_timepoints, max_design_columns, n_channels), dtype=float)
    response_matrix = np.zeros((n_timepoints, n_channels), dtype=float)
    for channel_index, (design_matrix, response_vector) in enumerate(zip(design_matrices, response_vectors)):
        design_tensor[:, :design_matrix.shape[1], channel_index] = design_matrix
        response_matrix[:, channel_index] = response_vector
    max_nuisance_columns = max(nuisance_column_counts) if nuisance_column_counts else 0
    nuisance_tensor = np.zeros((n_timepoints, max_nuisance_columns, n_channels), dtype=float)
    nuisance_name_matrix = np.empty((max_nuisance_columns, n_channels), dtype=object)
    nuisance_name_matrix[:] = ''
    for channel_index, (nuisance_values, nuisance_names) in enumerate(zip(nuisance_values_per_channel, nuisance_names_per_channel)):
        if nuisance_values.shape[1] > 0:
            nuisance_tensor[:, :nuisance_values.shape[1], channel_index] = nuisance_values
            nuisance_name_matrix[:nuisance_values.shape[1], channel_index] = np.asarray(nuisance_names, dtype=object)
    return {'subject': subject, 'file_label': file_label, 'amplitude_value': amplitude_value, 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'hrf_model': pipeline['hrf_model'], 'solver': pipeline['solver'], 'times_s': raw_hb.times.astype(float), 'stim_onsets_s': stimulus_onsets_s, 'stim_durations_s': stimulus_durations_s, 'stim_amplitudes': stimulus_amplitudes, 'Y': response_matrix, 'X': design_tensor, 'n_reg': np.asarray(design_column_counts, dtype=np.int32), 'task_reg_index': np.asarray(task_regressor_indices, dtype=np.int32), 'nuisance_values': nuisance_tensor, 'nuisance_n_reg': np.asarray(nuisance_column_counts, dtype=np.int32), 'nuisance_names': nuisance_name_matrix, 'channel_names': np.asarray(channel_names, dtype=object), 'pair_names': np.asarray(pair_names, dtype=object), 'chromophores': np.asarray(chromophores, dtype=object), 'target_status': np.asarray(target_status_labels, dtype=object), 'shift_index': int(shift_index), 'shift_s': float(shift_s), 'matlab_resample_fs': np.asarray([[4.0]], dtype=float), 'nuisance_detail_rows': nuisance_detail_rows}

def prepare_matlab_pipeline_inputs(subject, file_spec, pipeline, raw_hb, target_pair_names, config):
    """
    Prepare Python-generated design matrices for MATLAB AR-IRLS.
    """
    nuisance_by_channel = build_channel_nuisance_regressors(raw_hb, pipeline['nuisance_method'], config)
    observed_spec = build_matlab_input_spec(subject=subject, file_label=file_spec['label'], amplitude_value=file_spec['amplitude_value'], pipeline=pipeline, raw_hb=raw_hb, nuisance_by_channel=nuisance_by_channel, target_pair_names=target_pair_names, shift_index=0, shift_s=0.0, config=config)
    shifted_specs = []
    if file_spec['is_null'] and pipeline['include_in_empirical_null']:
        total_duration_s = float(raw_hb.times[-1])
        shift_values_s = get_shift_values(total_duration_s, config['empirical_null_shift_count'], config['empirical_null_min_shift_s'])
        for shift_index, shift_s in enumerate(shift_values_s, start=1):
            shifted_raw_hb = raw_hb.copy()
            shifted_raw_hb.set_annotations(make_shifted_annotations(raw_hb.annotations, total_duration_s, shift_s))
            shifted_nuisance_by_channel = build_channel_nuisance_regressors(shifted_raw_hb, pipeline['nuisance_method'], config)
            shifted_spec = build_matlab_input_spec(subject=subject, file_label=file_spec['label'], amplitude_value=file_spec['amplitude_value'], pipeline=pipeline, raw_hb=shifted_raw_hb, nuisance_by_channel=shifted_nuisance_by_channel, target_pair_names=target_pair_names, shift_index=shift_index, shift_s=shift_s, config=config)
            if shifted_spec is not None:
                shifted_specs.append(shifted_spec)
    nuisance_detail_df = pd.DataFrame()
    if observed_spec is not None and observed_spec.get('nuisance_detail_rows'):
        nuisance_detail_df = pd.DataFrame(observed_spec['nuisance_detail_rows'])
    return {'canonical_channel_metrics': pd.DataFrame(), 'block_average_channel_metrics': pd.DataFrame(), 'fir_channel_metrics': pd.DataFrame(), 'shape_metrics': pd.DataFrame(), 'roi_timecourses': pd.DataFrame(), 'nuisance_detail': nuisance_detail_df, 'matlab_input_specs_list': [] if observed_spec is None else [observed_spec], 'matlab_shift_specs_list': shifted_specs}

def run_single_pipeline(subject, file_spec, pipeline, raw_cw, raw_hb, target_pair_names, truth_templates, snirf_file_path, config):
    """
    One readable dispatch point for all pipeline families.
    """
    if pipeline['use_block_average']:
        return run_block_average_pipeline(subject, file_spec, pipeline, raw_hb, target_pair_names, truth_templates, config)
    if pipeline['backend'] == 'matlab_arirls':
        return prepare_matlab_pipeline_inputs(subject, file_spec, pipeline, raw_hb, target_pair_names, config)
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
    reference_path = subject_dir / 'resting_hrf_20.snirf'
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
        raise BenchmarkError('Could not read truth labels from reference file.')
    reference_cw_indices = get_cw_channel_indices(reference_raw)
    reference_cw_names = np.asarray(reference_raw.ch_names)[reference_cw_indices]
    if len(data_type_labels) == len(reference_cw_names):
        aligned_reference_names = reference_cw_names
    elif len(data_type_labels) == len(reference_raw.ch_names):
        aligned_reference_names = np.asarray(reference_raw.ch_names)
    else:
        raise BenchmarkError('Truth-label alignment failed.')
    target_pair_names = sorted(set((name.split(' ')[0] for name in aligned_reference_names[data_type_labels == 1].tolist())))
    target_pair_name_set = set(target_pair_names)
    cw_channel_table = build_cw_channel_table(reference_raw, subject, file_spec['label'], config)
    long_pair_names = sorted(cw_channel_table.loc[cw_channel_table['group'] == 'LS', 'pair_name'].astype(str).unique())
    non_target_pair_names = [pair_name for pair_name in long_pair_names if pair_name not in target_pair_name_set]
    channel_quality_df, pair_quality_df = build_quality_tables(raw_cw, subject, file_spec['label'], config)
    truth_templates = load_truth_templates(config)
    return {'subject_dir': subject_dir, 'snirf_file_path': snirf_file_path, 'reference_raw': reference_raw, 'raw_cw': raw_cw, 'target_pair_names': target_pair_names, 'target_pair_name_set': target_pair_name_set, 'non_target_pair_names': non_target_pair_names, 'truth_templates': truth_templates, 'channel_quality': channel_quality_df, 'pair_quality': pair_quality_df, 'cw_channel_table': cw_channel_table}

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

def collect_python_empirical_null_rows(subject, file_spec, pipeline, raw_cw, raw_hb, target_pair_name_set, truth_templates, snirf_file_path, config, errors):
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
            shifted_payload = run_single_pipeline(subject=subject, file_spec=file_spec, pipeline=pipeline, raw_cw=raw_cw, raw_hb=shifted_raw_hb, target_pair_names=target_pair_name_set, truth_templates=truth_templates, snirf_file_path=snirf_file_path, config=config)
        except Exception as exc:
            errors.append(make_error_row(subject, file_spec['label'], pipeline['label'], 'empirical_null_shift_python', str(exc)))
            continue
        shift_summary_df = summarize_shift_scores_from_payload(shifted_payload)
        if len(shift_summary_df) == 0:
            continue
        for _, summary_row in shift_summary_df.iterrows():
            empirical_null_tables.append(pd.DataFrame([{'subject': subject, 'file_label': file_spec['label'], 'pipeline_label': pipeline['label'], 'backend': pipeline['backend'], 'chromophore': summary_row['chromophore'], 'score_type': summary_row.get('score_type', 'unknown'), 'shift_index': shift_index, 'shift_s': shift_s, 'mean_target_score': summary_row['mean_target_score'], 'mean_non_target_score': summary_row['mean_non_target_score'], 'target_minus_non_target_score': summary_row['target_minus_non_target_score'], 'n_channels': summary_row.get('n_channels', np.nan)}]))
    return empirical_null_tables

def append_matlab_outputs_to_job_tables(all_canonical_tables, all_shape_tables, all_roi_tables, all_empirical_tables, matlab_df, subject, file_spec, raw_hb, truth_templates, config):
    if matlab_df is None or len(matlab_df) == 0:
        return
    observed_matlab_df = matlab_df.loc[matlab_df['shift_index'] == 0].copy() if 'shift_index' in matlab_df.columns else matlab_df.copy()
    if len(observed_matlab_df) > 0:
        all_canonical_tables.append(observed_matlab_df)
        epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
        for _, matlab_row in observed_matlab_df.iterrows():
            hrf_model_for_shape = 'gamma' if str(matlab_row.get('hrf_model', '')).lower() == 'gamma' else 'canonical'
            chromophore = matlab_row['chromophore']
            recovered_curve = float(matlab_row['beta']) * build_basis_curve(epoch_times_s, hrf_model_for_shape)
            truth_curve = get_truth_curve(truth_templates, file_spec['label'], epoch_times_s, chromophore)
            all_shape_tables.append(pd.DataFrame([{'subject': matlab_row['subject'], 'file_label': matlab_row['file_label'], 'amplitude_value': matlab_row['amplitude_value'], 'pipeline_label': matlab_row['pipeline_label'], 'backend': matlab_row['backend'], 'channel_name': matlab_row['channel_name'], 'pair_name': matlab_row['pair_name'], 'chromophore': chromophore, 'target_status': matlab_row['target_status'], 'shape_source': f'beta_scaled_{hrf_model_for_shape}', **compute_shape_metrics(recovered_curve, truth_curve, epoch_times_s, chromophore)}]))
        for (pipeline_label, chromophore, target_status), group_df in observed_matlab_df.groupby(['pipeline_label', 'chromophore', 'target_status']):
            hrf_model_for_roi = 'gamma' if 'Gamma' in pipeline_label or ('hrf_model' in group_df.columns and str(group_df['hrf_model'].iloc[0]).lower() == 'gamma') else 'canonical'
            roi_curve = float(group_df['beta'].mean()) * build_basis_curve(epoch_times_s, hrf_model_for_roi)
            all_roi_tables.append(pd.DataFrame({'subject': subject, 'file_label': file_spec['label'], 'amplitude_value': file_spec['amplitude_value'], 'pipeline_label': pipeline_label, 'backend': 'matlab_arirls', 'chromophore': chromophore, 'target_status': target_status, 'curve_source': f'roi_mean_beta_scaled_{hrf_model_for_roi}', 'time_s': epoch_times_s, 'signal': roi_curve}))
    shifted_matlab_df = matlab_df.loc[matlab_df['shift_index'] > 0].copy() if 'shift_index' in matlab_df.columns else pd.DataFrame()
    if len(shifted_matlab_df) == 0:
        return
    score_tables = []
    separation_df = shifted_matlab_df.groupby(['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore', 'target_status'], as_index=False).agg(mean_score=('beta', 'mean'), n_channels=('beta', 'count'))
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
    extra_score_tables = []
    epoch_times_s = np.arange(config['epoch_tmin'], config['epoch_tmax'] + 1e-09, np.median(np.diff(raw_hb.times)))
    for pipeline_label, pipeline_shift_df in shifted_matlab_df.groupby('pipeline_label'):
        hrf_model_for_shape = 'gamma' if 'Gamma' in str(pipeline_label) or ('hrf_model' in pipeline_shift_df.columns and str(pipeline_shift_df['hrf_model'].iloc[0]).lower() == 'gamma') else 'canonical'
        basis_curve = build_basis_curve(epoch_times_s, hrf_model_for_shape)
        hbo_peak_scale = float(np.nanmax(basis_curve))
        hbr_peak_scale = float(np.nanmin(basis_curve))
        auc_scale = float(scipy.integrate.trapezoid(basis_curve, epoch_times_s))
        tmp_df = pipeline_shift_df.copy()
        tmp_df['recovered_peak_amplitude'] = [float(beta * (hbo_peak_scale if chrom == 'hbo' else hbr_peak_scale)) for beta, chrom in zip(tmp_df['beta'], tmp_df['chromophore'])]
        tmp_df['recovered_auc'] = tmp_df['beta'].astype(float) * auc_scale
        for value_column, score_type in [('recovered_peak_amplitude', 'shape_peak_amplitude'), ('recovered_auc', 'shape_auc')]:
            extra_df = summarize_target_minus_non_target(tmp_df, value_column, group_columns=['subject', 'file_label', 'pipeline_label', 'backend', 'shift_index', 'shift_s', 'chromophore'], score_type=score_type)
            if len(extra_df) > 0:
                extra_score_tables.append(extra_df)
    score_tables.extend(extra_score_tables)
    if score_tables:
        all_empirical_tables.append(pd.concat(score_tables, ignore_index=True))

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
        channel_availability_rows = []
        matlab_observed_specs = []
        matlab_shift_specs = []
        for pipeline in config['pipeline_specs']:
            if pipeline['backend'] == 'matlab_arirls' and (not config['use_matlab']):
                continue
            raw_hb, bad_pair_names = get_preprocessed_hb_for_pipeline(pipeline)
            try:
                pipeline_payload = run_single_pipeline(subject=subject, file_spec=file_spec, pipeline=pipeline, raw_cw=raw_cw, raw_hb=raw_hb, target_pair_names=target_pair_name_set, truth_templates=truth_templates, snirf_file_path=snirf_file_path, config=config)
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
            if pipeline_payload['matlab_input_specs_list']:
                matlab_observed_specs.extend(pipeline_payload['matlab_input_specs_list'])
            if pipeline_payload['matlab_shift_specs_list']:
                matlab_shift_specs.extend(pipeline_payload['matlab_shift_specs_list'])
            channel_availability_rows.append(build_channel_availability_row(subject, file_spec, pipeline, raw_hb, target_pair_names, non_target_pair_names, bad_pair_names, config))
            if file_spec['is_null'] and pipeline['include_in_empirical_null'] and (pipeline['backend'] == 'python'):
                all_empirical_tables.extend(collect_python_empirical_null_rows(subject, file_spec, pipeline, raw_cw, raw_hb, target_pair_name_set, truth_templates, snirf_file_path, config, errors))
        if config['use_matlab'] and (matlab_observed_specs or matlab_shift_specs):
            matlab_bundle_json_path = write_matlab_bundle(matlab_observed_specs, matlab_shift_specs, job_dir)
            if matlab_bundle_json_path is not None:
                try:
                    matlab_df = run_matlab_sidecar(config, matlab_bundle_json_path, job_dir)
                except Exception as exc:
                    errors.append(make_error_row(subject, file_spec['label'], 'MATLAB_ARIRLS', 'matlab_sidecar', f'{exc}\n\n{traceback.format_exc()}'))
                    matlab_df = pd.DataFrame()
                append_matlab_outputs_to_job_tables(all_canonical_tables, all_shape_tables, all_roi_tables, all_empirical_tables, matlab_df, subject, file_spec, raw_hb, truth_templates, config)
        maybe_write_table(pd.DataFrame(channel_availability_rows), job_dir / 'channel_availability', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_canonical_tables, ignore_index=True) if all_canonical_tables else pd.DataFrame(), job_dir / 'canonical_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_block_tables, ignore_index=True) if all_block_tables else pd.DataFrame(), job_dir / 'block_average_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_fir_tables, ignore_index=True) if all_fir_tables else pd.DataFrame(), job_dir / 'fir_channel_metrics', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_shape_tables, ignore_index=True) if all_shape_tables else pd.DataFrame(), job_dir / 'shape_fidelity', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_roi_tables, ignore_index=True) if all_roi_tables else pd.DataFrame(), job_dir / 'roi_timecourses', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_nuisance_tables, ignore_index=True) if all_nuisance_tables else pd.DataFrame(), job_dir / 'nuisance_detail', config['write_csv'], config['write_parquet'])
        maybe_write_table(pd.concat(all_empirical_tables, ignore_index=True) if all_empirical_tables else pd.DataFrame(), job_dir / 'empirical_null_shift', config['write_csv'], config['write_parquet'])
        if errors:
            maybe_write_table(pd.DataFrame(errors), job_dir / 'error_log', True, config['write_parquet'])
        success_marker_path.write_text(json.dumps({'subject': subject, 'file_label': file_spec['label'], 'finished_utc': now_utc_iso(), 'job_dir': str(job_dir)}, indent=2), encoding='utf-8')
        return {'subject': subject, 'file_label': file_spec['label'], 'status': 'finished', 'job_dir': str(job_dir)}
    except Exception as exc:
        errors.append(make_error_row(subject, file_spec['label'], None, 'job_crash', f'{exc}\n\n{traceback.format_exc()}'))
        maybe_write_table(pd.DataFrame(errors), job_dir / 'error_log', True, config['write_parquet'])
        return {'subject': subject, 'file_label': file_spec['label'], 'status': 'crashed', 'job_dir': str(job_dir), 'error': str(exc)}

def aggregate_outputs(config, run_results):
    aggregate_dir = ensure_dir(aggregate_path(config))
    table_names = ['truth_summary', 'channel_quality', 'pair_quality', 'channel_availability', 'canonical_channel_metrics', 'block_average_channel_metrics', 'fir_channel_metrics', 'shape_fidelity', 'roi_timecourses', 'nuisance_detail', 'empirical_null_shift', 'error_log']
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
    target_sep_tables = [make_target_sep_table(canonical_df, 'beta', 'canonical_beta'), make_target_sep_table(block_df, 'score', 'block_average_score'), make_target_sep_table(shape_df, 'recovered_peak_amplitude', 'shape_peak_amplitude'), make_target_sep_table(shape_df, 'recovered_auc', 'shape_auc')]
    target_vs_nontarget = pd.concat([df for df in target_sep_tables if len(df) > 0], ignore_index=True) if any((len(df) > 0 for df in target_sep_tables)) else pd.DataFrame()
    roi_tables = []
    if len(canonical_df) > 0:
        c_roi = canonical_df.groupby(['subject', 'file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'target_status'], as_index=False).agg(roi_mean_score=('beta', 'mean'), roi_std_score=('beta', 'std'), n_channels=('channel_name', 'count'))
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
    if len(canonical_df) > 0 and 'p_value' in canonical_df.columns:
        null_df = canonical_df.loc[canonical_df['file_label'] == 'no_hrf'].copy()
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
    shape_target_primary = summarize_shape_subset(shape_df.loc[(shape_df['target_status'] == 'true_target') & (shape_df['file_label'] != 'no_hrf')].copy() if len(shape_df) > 0 else pd.DataFrame(), 'true_target_active')
    shape_specificity_secondary = summarize_shape_subset(shape_df.loc[(shape_df['target_status'] == 'true_non_target') & (shape_df['file_label'] != 'no_hrf')].copy() if len(shape_df) > 0 else pd.DataFrame(), 'true_non_target_active')
    shape_null_summary = summarize_shape_subset(shape_df.loc[shape_df['file_label'] == 'no_hrf'].copy() if len(shape_df) > 0 else pd.DataFrame(), 'null_condition')
    shape_fidelity_summary = pd.concat([df for df in [shape_target_primary, shape_specificity_secondary, shape_null_summary] if len(df) > 0], ignore_index=True) if any((len(df) > 0 for df in [shape_target_primary, shape_specificity_secondary, shape_null_summary])) else pd.DataFrame()
    pipeline_performance_summary = pd.DataFrame()
    if len(target_vs_nontarget) > 0:
        pipeline_performance_summary = target_vs_nontarget.groupby(['file_label', 'amplitude_value', 'pipeline_label', 'backend', 'chromophore', 'score_type'], as_index=False).agg(mean_target_score=('mean_target_score', 'mean'), mean_non_target_score=('mean_non_target_score', 'mean'), mean_target_minus_non_target_score=('target_minus_non_target_score', 'mean'), std_target_minus_non_target_score=('target_minus_non_target_score', 'std'), median_target_minus_non_target_score=('target_minus_non_target_score', 'median'), n_subject_rows=('subject', 'count'))
        pipeline_performance_summary = attach_pipeline_metadata(pipeline_performance_summary, config)
    all_tables = {**combined, 'pipeline_manifest': manifest, 'master_run_log': pd.DataFrame(run_results), 'roi_scores': roi_scores, 'dlpfc_roi_scores': dlpfc_roi_scores, 'target_vs_nontarget_summary': target_vs_nontarget, 'parametric_null_summary': parametric_null_summary, 'empirical_null_pvalues': empirical_null_pvalues, 'variability_summary': variability_summary, 'pairwise_pipeline_deltas': pairwise_deltas, 'shape_fidelity_summary': shape_fidelity_summary, 'pipeline_performance_summary': pipeline_performance_summary, 'config': pd.DataFrame([config_snapshot()])}
    for name, df in all_tables.items():
        maybe_write_table(df, aggregate_dir / name, config['write_csv'], config['write_parquet'])

if __name__ == "__main__":
    config = config_snapshot()
    ensure_dir(output_path(config))
    ensure_dir(jobs_path(config))
    ensure_dir(aggregate_path(config))
    if not config["subject_names"]:
        discovered_subject_names = []
        for subject_dir in sorted(dataset_path(config).glob("Subj*")):
            if subject_dir.is_dir():
                discovered_subject_names.append(subject_dir.name)
        config["subject_names"] = discovered_subject_names
    jobs = []
    for subject in config["subject_names"]:
        for file_spec in config["file_specs"]:
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
    aggregate_outputs(config, run_results)
    print(f"Done. Aggregate outputs written to: {aggregate_path(config)}")
