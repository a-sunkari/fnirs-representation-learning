import pandas as pd
import numpy as np
import mne
import mne_nirs
import h5py
import plotly.express as px

from pathlib import Path
from datetime import datetime

from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index,
    source_detector_distances,
    short_channels,
)
from mne.preprocessing.nirs import temporal_derivative_distribution_repair as tddr

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

root = Path.home() / "fnirs-representation-learning"
rs_data_dir = root / "snirf_dataset_2"
output_dir = root / "outputs"
output_tables_dir = output_dir / "tables"
output_figures_dir = output_dir / "figures"

output_tables_dir.mkdir(parents=True, exist_ok=True)
output_figures_dir.mkdir(parents=True, exist_ok=True)

manual_subject_names = [
    "Subj86", "Subj91", "Subj92", "Subj94", "Subj95", "Subj96",
    "Subj97", "Subj98", "Subj99", "Subj100", "Subj101",
    "Subj102", "Subj103", "Subj104",
]
auto_select_additional_subjects = False

file_labels = ["no_hrf", "hrf_20"]
output_prefix = "batch_true_injected_comprehensive_v5"

save_html_figures = True
save_intermediate_after_each_subject = True
include_fir_pipeline = True
compute_empirical_null = True
empirical_null_shift_count = 8
empirical_null_min_shift_s = 20.0

short_separation_threshold_m = 0.015
long_separation_threshold_m = 0.025
local_ss_max_distance_m = 0.015

strict_sci_threshold = 0.50
loose_sci_threshold = 0.35
strict_snr_threshold = 2.0
strict_negative_fraction_threshold = 0.001

filter_low_hz = 0.01
filter_high_hz = 0.20

epoch_tmin = -5.0
epoch_tmax = 30.0
baseline_window = (-5.0, 0.0)
response_window = (4.0, 8.0)

stim_duration_s = 1.0
drift_high_pass_hz = 0.01
fir_delays_s = list(range(26))
ppf_value = 0.1
canonical_noise_model = "auto"

n_pooled_ss_components = 2
n_ss_aux_components = 3

plot_condition_order = ["no_hrf", "hrf_20"]
plot_target_status_order = ["true_target", "true_non_target"]
plot_score_type_order = ["canonical_beta", "block_average_score"]
plot_chromophore_order = ["hbo", "hbr"]

subject_name = None
subject_dir = None
file_label = None


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------


def get_file_path(subject_dir, file_label):
    if file_label == "no_hrf":
        return subject_dir / "resting_clean.snirf"
    if file_label == "hrf_20":
        return subject_dir / "resting_hrf_20.snirf"
    if file_label == "hrf_50":
        return subject_dir / "resting_hrf_50.snirf"
    if file_label == "hrf_100":
        return subject_dir / "resting_hrf_100.snirf"
    raise ValueError(f"Unknown file_label: {file_label}")


def get_annotation_source_file_path(subject_dir):
    return subject_dir / "resting_hrf_20.snirf"


def get_amplitude_value(file_label):
    if file_label == "no_hrf":
        return 0
    if file_label == "hrf_20":
        return 20
    if file_label == "hrf_50":
        return 50
    if file_label == "hrf_100":
        return 100
    return np.nan


def resolve_subject_names():
    resolved_subjects = []
    for subject in manual_subject_names:
        if subject not in resolved_subjects:
            resolved_subjects.append(subject)

    if not auto_select_additional_subjects:
        return resolved_subjects

    if not rs_data_dir.exists():
        print(f"Subject directory root not found: {rs_data_dir}")
        return resolved_subjects

    available_subjects = sorted([
        path.name for path in rs_data_dir.iterdir()
        if path.is_dir() and path.name.startswith("Subj")
    ])

    for subject in available_subjects:
        if subject not in resolved_subjects:
            resolved_subjects.append(subject)

    return resolved_subjects


def copy_valid_annotations(raw_target_cw, annotation_source_file_path):
    raw_annotation_source_cw = mne.io.read_raw_snirf(annotation_source_file_path, preload=False, verbose=False)
    source_annotations = raw_annotation_source_cw.annotations
    target_duration_s = raw_target_cw.times[-1]

    kept_onsets = []
    kept_durations = []
    kept_descriptions = []

    for onset_s, duration_s, description in zip(
        source_annotations.onset,
        source_annotations.duration,
        source_annotations.description,
    ):
        if onset_s >= target_duration_s:
            continue

        clipped_duration_s = min(duration_s, max(0.0, target_duration_s - onset_s))
        kept_onsets.append(onset_s)
        kept_durations.append(clipped_duration_s)
        kept_descriptions.append(description)

    raw_target_cw.set_annotations(
        mne.Annotations(
            onset=kept_onsets,
            duration=kept_durations,
            description=kept_descriptions,
        )
    )

    return raw_target_cw


# -----------------------------------------------------------------------------
# Error logging / small utilities
# -----------------------------------------------------------------------------


def make_error_row(subject, file_label_value, pipeline_label, stage, error_message, channel_name=None):
    return {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "subject": subject,
        "file_label": file_label_value,
        "pipeline_label": pipeline_label,
        "stage": stage,
        "channel_name": channel_name,
        "error_message": str(error_message),
    }


def safe_standardize_rows(data_matrix):
    data_matrix = np.asarray(data_matrix, dtype=float)
    if data_matrix.ndim == 1:
        data_matrix = data_matrix[None, :]
    centered = data_matrix - data_matrix.mean(axis=1, keepdims=True)
    stds = data_matrix.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return centered / stds


def extract_principal_components(data_matrix, n_components):
    data_matrix = safe_standardize_rows(data_matrix)
    if data_matrix.shape[0] == 0:
        return None
    if data_matrix.shape[0] == 1:
        return data_matrix.T
    u_matrix, singular_values, _ = np.linalg.svd(data_matrix.T, full_matrices=False)
    kept_components = min(n_components, u_matrix.shape[1], data_matrix.shape[0])
    if kept_components <= 0:
        return None
    return u_matrix[:, :kept_components] * singular_values[:kept_components]


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(len(p_values), np.nan)
    valid_mask = np.isfinite(p_values)
    if valid_mask.sum() == 0:
        return q_values

    valid_p = p_values[valid_mask]
    order = np.argsort(valid_p)
    ordered_p = valid_p[order]
    n = len(ordered_p)
    ordered_q = ordered_p * n / np.arange(1, n + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    ordered_q = np.clip(ordered_q, 0.0, 1.0)

    valid_q = np.empty_like(ordered_q)
    valid_q[order] = ordered_q
    q_values[valid_mask] = valid_q
    return q_values


# -----------------------------------------------------------------------------
# SNIRF truth-label helpers
# -----------------------------------------------------------------------------


def read_measurement_data_type_labels(snirf_file_path):
    snirf_file_path = Path(snirf_file_path)
    if not snirf_file_path.exists():
        return None

    with h5py.File(snirf_file_path, "r") as h5_file:
        if "nirs" not in h5_file:
            return None

        nirs_group = h5_file["nirs"]
        data_group_name = None
        if "data1" in nirs_group:
            data_group_name = "data1"
        else:
            for key in nirs_group.keys():
                if key.startswith("data"):
                    data_group_name = key
                    break

        if data_group_name is None:
            return None

        data_group = nirs_group[data_group_name]
        measurement_entries = []

        for key in data_group.keys():
            if not key.startswith("measurementList"):
                continue

            suffix = key.replace("measurementList", "")
            try:
                measurement_index = int(suffix)
            except ValueError:
                continue

            measurement_group = data_group[key]
            if "dataTypeLabel" in measurement_group:
                label_value = measurement_group["dataTypeLabel"][()]
            else:
                label_value = 0

            if isinstance(label_value, np.ndarray):
                if label_value.size == 0:
                    label_value = 0
                else:
                    label_value = label_value.reshape(-1)[0]

            try:
                label_value = int(label_value)
            except Exception:
                label_value = 0

            measurement_entries.append((measurement_index, label_value))

    if len(measurement_entries) == 0:
        return None

    measurement_entries = sorted(measurement_entries, key=lambda item: item[0])
    return np.array([label_value for _, label_value in measurement_entries], dtype=int)


def get_cw_channel_indices(raw_snirf):
    picks_fnirs = mne.pick_types(raw_snirf.info, fnirs=True)
    channel_types = np.array(raw_snirf.get_channel_types())
    picks_cw = picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]
    return picks_cw


def get_true_injected_channel_names(reference_snirf_file_path, raw_cw):
    data_type_labels = read_measurement_data_type_labels(reference_snirf_file_path)
    if data_type_labels is None:
        return []

    picks_cw = get_cw_channel_indices(raw_cw)
    cw_channel_names = np.array(raw_cw.ch_names)[picks_cw]

    if len(data_type_labels) == len(cw_channel_names):
        aligned_channel_names = cw_channel_names
    elif len(data_type_labels) == len(raw_cw.ch_names):
        aligned_channel_names = np.array(raw_cw.ch_names)
    else:
        raise ValueError(
            f"Could not align dataTypeLabel entries ({len(data_type_labels)}) "
            f"with CW channel count ({len(cw_channel_names)}) or total channel count ({len(raw_cw.ch_names)})."
        )

    return aligned_channel_names[data_type_labels == 1].tolist()


def get_true_injected_pair_names(reference_snirf_file_path, raw_cw):
    true_injected_channel_names = get_true_injected_channel_names(reference_snirf_file_path, raw_cw)
    return sorted(set([channel_name.split(" ")[0] for channel_name in true_injected_channel_names]))


def get_channel_names_from_pairs(pair_names):
    return [pair_name + " hbo" for pair_name in pair_names] + [pair_name + " hbr" for pair_name in pair_names]


# -----------------------------------------------------------------------------
# Channel / QC helpers
# -----------------------------------------------------------------------------


def build_cw_channel_table(raw_snirf, subject_name, file_label):
    picks_cw = get_cw_channel_indices(raw_snirf)
    cw_names = np.array(raw_snirf.ch_names)[picks_cw]
    distances_m = source_detector_distances(raw_snirf.info, picks=picks_cw)
    short_mask_all = short_channels(raw_snirf.info, threshold=short_separation_threshold_m)
    short_mask = short_mask_all[picks_cw]
    long_mask = distances_m >= long_separation_threshold_m
    pair_names = np.array([channel_name.split(" ")[0] for channel_name in cw_names])

    channel_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "amplitude_value": get_amplitude_value(file_label),
        "channel_name": cw_names,
        "pair_name": pair_names,
        "distance_m": distances_m,
        "is_ss": short_mask,
        "is_ls": long_mask,
    })

    channel_table["group"] = np.select(
        [channel_table["is_ss"], channel_table["is_ls"]],
        ["SS", "LS"],
        default="MID",
    )

    return channel_table


def build_hb_channel_table(raw_hb, subject_name, file_label):
    picks_hbo = mne.pick_types(raw_hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(raw_hb.info, fnirs="hbr")
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))

    hb_names = np.array(raw_hb.ch_names)[picks_hb]
    hb_types = np.array(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.array([channel_name.split(" ")[0] for channel_name in hb_names])

    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]

    short_mask_all = short_channels(raw_hb.info, threshold=short_separation_threshold_m)
    short_mask = short_mask_all[picks_hb]
    long_mask = distances_hb >= long_separation_threshold_m

    hb_channel_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "amplitude_value": get_amplitude_value(file_label),
        "channel_name": hb_names,
        "pair_name": pair_names,
        "chromophore": hb_types,
        "distance_m": distances_hb,
        "is_ss": short_mask,
        "is_ls": long_mask,
    })

    hb_channel_table["group"] = np.select(
        [hb_channel_table["is_ss"], hb_channel_table["is_ls"]],
        ["SS", "LS"],
        default="MID",
    )

    return hb_channel_table


def build_quality_table(raw_cw, subject_name, file_label):
    raw_od = optical_density(raw_cw.copy())

    picks_cw = get_cw_channel_indices(raw_cw)
    cw_channel_table = build_cw_channel_table(raw_cw, subject_name, file_label)
    cw_channel_names = np.array(raw_cw.ch_names)[picks_cw]
    cw_data = raw_cw.get_data(picks=picks_cw)

    sci_values = scalp_coupling_index(raw_od)
    mean_abs_signal = np.mean(np.abs(cw_data), axis=1)
    signal_std = np.std(cw_data, axis=1)
    snr_values = mean_abs_signal / np.maximum(signal_std, 1e-12)
    negative_fraction_values = np.mean(cw_data <= 0, axis=1)

    channel_quality_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "amplitude_value": get_amplitude_value(file_label),
        "channel_name": cw_channel_names,
        "pair_name": cw_channel_table["pair_name"].values,
        "distance_m": cw_channel_table["distance_m"].values,
        "group": cw_channel_table["group"].values,
        "sci": sci_values,
        "snr": snr_values,
        "negative_fraction": negative_fraction_values,
    })

    pair_quality_table = (
        channel_quality_table
        .groupby(["subject", "file_label", "amplitude_value", "pair_name"], as_index=False)
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

    return channel_quality_table, pair_quality_table


def get_bad_pair_names(pair_quality_table, pruning_style):
    pair_quality_table = pair_quality_table.copy()

    if pruning_style == "strict_combined":
        bad_mask = (
            (pair_quality_table["sci_min"] < strict_sci_threshold) |
            (pair_quality_table["snr_min"] < strict_snr_threshold) |
            (pair_quality_table["negative_fraction_max"] > strict_negative_fraction_threshold)
        )
    elif pruning_style == "loose_sci":
        bad_mask = pair_quality_table["sci_min"] < loose_sci_threshold
    else:
        raise ValueError(f"Unknown pruning_style: {pruning_style}")

    return pair_quality_table.loc[bad_mask, "pair_name"].tolist()


def apply_bad_pairs_to_hb(raw_hb, bad_pair_names):
    raw_hb = raw_hb.copy()
    hb_bad_channel_names = []

    for channel_name in raw_hb.ch_names:
        if channel_name.split(" ")[0] in bad_pair_names:
            hb_bad_channel_names.append(channel_name)

    raw_hb.info["bads"] = hb_bad_channel_names
    return raw_hb


def get_available_long_channel_names(raw_hb):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)
    long_channel_names = hb_channel_table.loc[hb_channel_table["group"] == "LS", "channel_name"].tolist()
    return sorted([channel_name for channel_name in long_channel_names if channel_name not in raw_hb.info["bads"]])


def get_available_short_channel_names(raw_hb, chromophore):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)
    short_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "SS") &
        (hb_channel_table["chromophore"] == chromophore),
        "channel_name",
    ].tolist()
    return [channel_name for channel_name in short_channel_names if channel_name not in raw_hb.info["bads"]]


def get_target_status_from_pair_name(pair_name, target_pair_names):
    return "true_target" if pair_name in target_pair_names else "true_non_target"


def get_pipeline_spec(pipeline_label):
    if pipeline_label == "PrefLocalSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "nuisance_method": "local_nearest"}
    if pipeline_label == "PnoSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "nuisance_method": "none"}
    if pipeline_label == "PPooledPCA2":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "nuisance_method": "pooled_pca2"}
    if pipeline_label == "PSSAuxPCA":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "nuisance_method": "ss_aux_pca"}
    if pipeline_label == "PBlockAvg":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "block_average", "nuisance_method": "local_nearest"}
    if pipeline_label == "PLooseQC":
        return {"pruning_style": "loose_sci", "do_tddr": True, "estimation_method": "canonical_glm", "nuisance_method": "local_nearest"}
    if pipeline_label == "PFIRLocalSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "fir_glm", "nuisance_method": "local_nearest"}
    raise ValueError(f"Unknown pipeline_label: {pipeline_label}")


pipeline_labels = [
    "PrefLocalSS",
    "PnoSS",
    "PPooledPCA2",
    "PSSAuxPCA",
    "PBlockAvg",
    "PLooseQC",
]
if include_fir_pipeline:
    pipeline_labels.append("PFIRLocalSS")

core_variability_pipelines = ["PrefLocalSS", "PnoSS", "PPooledPCA2", "PSSAuxPCA"]


# -----------------------------------------------------------------------------
# Nuisance-regressor helpers
# -----------------------------------------------------------------------------


def get_channel_position(raw, channel_name):
    channel_index = raw.ch_names.index(channel_name)
    return np.array(raw.info["chs"][channel_index]["loc"][:3], dtype=float)


def get_auxiliary_channel_names(raw_cw):
    channel_types = raw_cw.get_channel_types()
    auxiliary_channel_names = []
    excluded_types = {"fnirs_cw_amplitude", "fnirs_od", "hbo", "hbr", "stim"}
    for channel_name, channel_type in zip(raw_cw.ch_names, channel_types):
        if channel_type not in excluded_types:
            auxiliary_channel_names.append(channel_name)
    return auxiliary_channel_names


def get_auxiliary_signal_matrix(raw_cw, raw_hb):
    auxiliary_channel_names = get_auxiliary_channel_names(raw_cw)
    if len(auxiliary_channel_names) == 0:
        return None, []

    aux_raw = raw_cw.copy().pick(auxiliary_channel_names)
    auxiliary_data = aux_raw.get_data()

    if auxiliary_data.shape[1] != len(raw_hb.times) or not np.allclose(aux_raw.times, raw_hb.times):
        resampled_rows = []
        for row in auxiliary_data:
            resampled_rows.append(np.interp(raw_hb.times, aux_raw.times, row))
        auxiliary_data = np.vstack(resampled_rows)

    return auxiliary_data, auxiliary_channel_names


def build_short_channel_data_matrix(raw_hb, chromophore):
    short_channel_names = get_available_short_channel_names(raw_hb, chromophore)
    if len(short_channel_names) == 0:
        return None, []
    short_channel_data = raw_hb.copy().pick(short_channel_names).get_data()
    return short_channel_data, short_channel_names


def build_local_short_regressor(raw_hb, long_channel_name, chromophore):
    short_channel_names = get_available_short_channel_names(raw_hb, chromophore)
    if len(short_channel_names) == 0:
        return None, None, np.nan, "no_short_channels"

    long_position = get_channel_position(raw_hb, long_channel_name)
    distance_rows = []
    for short_channel_name in short_channel_names:
        short_position = get_channel_position(raw_hb, short_channel_name)
        midpoint_distance = float(np.linalg.norm(long_position - short_position))
        distance_rows.append((short_channel_name, midpoint_distance))

    distance_rows = sorted(distance_rows, key=lambda item: item[1])
    nearest_short_channel_name, nearest_distance_m = distance_rows[0]

    if nearest_distance_m <= local_ss_max_distance_m:
        regressor_signal = raw_hb.copy().pick([nearest_short_channel_name]).get_data()[0]
        return regressor_signal, nearest_short_channel_name, nearest_distance_m, "nearest_short_channel"

    short_channel_data = raw_hb.copy().pick(short_channel_names).get_data()
    regressor_signal = short_channel_data.mean(axis=0)
    return regressor_signal, "pooled_fallback_average", nearest_distance_m, "pooled_fallback_average"


def build_nuisance_dataframe(raw_hb, raw_cw, chromophore, nuisance_method, long_channel_name=None):
    if nuisance_method == "none":
        return None, {}

    nuisance_metadata = {}

    if nuisance_method == "pooled_average":
        short_channel_data, short_channel_names = build_short_channel_data_matrix(raw_hb, chromophore)
        if short_channel_data is None:
            return None, {"nuisance_method_used": "none_no_short_channels"}
        regressor_values = short_channel_data.mean(axis=0)
        nuisance_metadata["nuisance_method_used"] = "pooled_average"
        nuisance_metadata["n_short_channels_used"] = len(short_channel_names)
        return pd.DataFrame({f"ss_avg_{chromophore}": regressor_values}), nuisance_metadata

    if nuisance_method == "local_nearest":
        regressor_values, regressor_label, nearest_distance_m, mode_label = build_local_short_regressor(
            raw_hb=raw_hb,
            long_channel_name=long_channel_name,
            chromophore=chromophore,
        )
        if regressor_values is None:
            return None, {"nuisance_method_used": "none_no_short_channels"}
        nuisance_metadata["nuisance_method_used"] = mode_label
        nuisance_metadata["nuisance_regressor_label"] = regressor_label
        nuisance_metadata["nearest_short_distance_m"] = nearest_distance_m
        return pd.DataFrame({f"ss_local_{chromophore}": regressor_values}), nuisance_metadata

    if nuisance_method == "pooled_pca2":
        short_channel_data, short_channel_names = build_short_channel_data_matrix(raw_hb, chromophore)
        if short_channel_data is None:
            return None, {"nuisance_method_used": "none_no_short_channels"}
        component_matrix = extract_principal_components(short_channel_data, n_pooled_ss_components)
        if component_matrix is None:
            return None, {"nuisance_method_used": "none_failed_pca"}
        nuisance_df = pd.DataFrame(component_matrix, columns=[f"ss_pc{i + 1}_{chromophore}" for i in range(component_matrix.shape[1])])
        nuisance_metadata["nuisance_method_used"] = "pooled_pca2"
        nuisance_metadata["n_short_channels_used"] = len(short_channel_names)
        nuisance_metadata["n_nuisance_components"] = component_matrix.shape[1]
        return nuisance_df, nuisance_metadata

    if nuisance_method == "ss_aux_pca":
        block_rows = []
        short_channel_data, short_channel_names = build_short_channel_data_matrix(raw_hb, chromophore)
        if short_channel_data is not None:
            block_rows.append(short_channel_data)
        auxiliary_data, auxiliary_channel_names = get_auxiliary_signal_matrix(raw_cw, raw_hb)
        if auxiliary_data is not None:
            block_rows.append(auxiliary_data)
        if len(block_rows) == 0:
            return None, {"nuisance_method_used": "none_no_short_or_aux"}
        combined_matrix = np.vstack(block_rows)
        component_matrix = extract_principal_components(combined_matrix, n_ss_aux_components)
        if component_matrix is None:
            return None, {"nuisance_method_used": "none_failed_pca"}
        nuisance_df = pd.DataFrame(component_matrix, columns=[f"aux_pc{i + 1}_{chromophore}" for i in range(component_matrix.shape[1])])
        nuisance_metadata["nuisance_method_used"] = "ss_aux_pca"
        nuisance_metadata["n_short_channels_used"] = 0 if short_channel_data is None else short_channel_data.shape[0]
        nuisance_metadata["n_aux_channels_used"] = 0 if auxiliary_data is None else auxiliary_data.shape[0]
        nuisance_metadata["n_nuisance_components"] = component_matrix.shape[1]
        return nuisance_df, nuisance_metadata

    raise ValueError(f"Unknown nuisance_method: {nuisance_method}")


def apply_channelwise_nuisance_regression(raw_hb, raw_cw, nuisance_method):
    if nuisance_method == "none":
        return raw_hb.copy(), pd.DataFrame()

    raw_hb = raw_hb.copy().load_data()
    long_channel_names = get_available_long_channel_names(raw_hb)
    nuisance_meta_rows = []

    for channel_name in long_channel_names:
        chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"
        nuisance_df, nuisance_metadata = build_nuisance_dataframe(
            raw_hb=raw_hb,
            raw_cw=raw_cw,
            chromophore=chromophore,
            nuisance_method=nuisance_method,
            long_channel_name=channel_name,
        )
        if nuisance_df is None or nuisance_df.shape[1] == 0:
            nuisance_meta_rows.append({
                "subject": subject_name,
                "file_label": file_label,
                "pipeline_label": "PBlockAvg",
                "channel_name": channel_name,
                "chromophore": chromophore,
                **nuisance_metadata,
            })
            continue

        design_matrix = np.column_stack([np.ones(len(raw_hb.times)), nuisance_df.to_numpy()])
        channel_signal = raw_hb.copy().pick([channel_name]).get_data()[0]
        coefficients, _, _, _ = np.linalg.lstsq(design_matrix, channel_signal, rcond=None)
        fitted_signal = design_matrix @ coefficients
        cleaned_signal = channel_signal - fitted_signal + coefficients[0]
        raw_hb._data[raw_hb.ch_names.index(channel_name), :] = cleaned_signal

        nuisance_meta_rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "channel_name": channel_name,
            "chromophore": chromophore,
            **nuisance_metadata,
        })

    nuisance_meta_df = pd.DataFrame(nuisance_meta_rows)
    return raw_hb, nuisance_meta_df


def preprocess_raw_to_hb(raw_cw, pipeline_label):
    pipeline_spec = get_pipeline_spec(pipeline_label)
    pruning_style = pipeline_spec["pruning_style"]
    do_tddr = pipeline_spec["do_tddr"]

    channel_quality_table, pair_quality_table = build_quality_table(raw_cw, subject_name, file_label)
    bad_pair_names = get_bad_pair_names(pair_quality_table, pruning_style)

    raw_od = optical_density(raw_cw.copy())
    processed_od = tddr(raw_od.copy()) if do_tddr else raw_od.copy()
    processed_od = processed_od.copy().filter(filter_low_hz, filter_high_hz, verbose=False)
    raw_hb = beer_lambert_law(processed_od, ppf=ppf_value)
    raw_hb = apply_bad_pairs_to_hb(raw_hb, bad_pair_names)

    nuisance_meta_df = pd.DataFrame()
    if pipeline_spec["estimation_method"] == "block_average":
        raw_hb, nuisance_meta_df = apply_channelwise_nuisance_regression(
            raw_hb=raw_hb,
            raw_cw=raw_cw,
            nuisance_method=pipeline_spec["nuisance_method"],
        )

    return {
        "pipeline_label": pipeline_label,
        "raw_hb": raw_hb,
        "raw_cw": raw_cw,
        "channel_quality_table": channel_quality_table,
        "pair_quality_table": pair_quality_table,
        "bad_pair_names": bad_pair_names,
        "pruning_style": pruning_style,
        "do_tddr": do_tddr,
        "nuisance_meta_df": nuisance_meta_df,
    }


# -----------------------------------------------------------------------------
# Estimation helpers
# -----------------------------------------------------------------------------


def make_epochs_from_raw_hb(raw_hb):
    events, event_id = mne.events_from_annotations(raw_hb, verbose=False)
    epochs_hb = mne.Epochs(
        raw_hb,
        events=events,
        event_id=event_id,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        baseline=baseline_window,
        preload=True,
        detrend=None,
        reject_by_annotation=False,
        verbose=False,
    )
    return epochs_hb, events, event_id


def build_mean_epoch_plot_table(epochs_hb, channel_names, pipeline_label, file_label, trace_group_label):
    available_channel_names = [channel_name for channel_name in channel_names if channel_name in epochs_hb.ch_names]
    hbo_channel_names = [channel_name for channel_name in available_channel_names if channel_name.endswith("hbo")]
    hbr_channel_names = [channel_name for channel_name in available_channel_names if channel_name.endswith("hbr")]

    plot_rows = []
    if len(hbo_channel_names) > 0:
        hbo_data = epochs_hb.copy().pick(hbo_channel_names).get_data()
        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hb.times,
            "signal": hbo_data.mean(axis=(0, 1)),
            "chromophore": "HbO",
            "pipeline_label": pipeline_label,
            "file_label": file_label,
            "trace_group_label": trace_group_label,
        }))
    if len(hbr_channel_names) > 0:
        hbr_data = epochs_hb.copy().pick(hbr_channel_names).get_data()
        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hb.times,
            "signal": hbr_data.mean(axis=(0, 1)),
            "chromophore": "HbR",
            "pipeline_label": pipeline_label,
            "file_label": file_label,
            "trace_group_label": trace_group_label,
        }))

    if len(plot_rows) == 0:
        return pd.DataFrame(columns=["time_s", "signal", "chromophore", "pipeline_label", "file_label", "trace_group_label"])

    return pd.concat(plot_rows, ignore_index=True)


def find_first_matching_column(column_names, candidate_names):
    for candidate_name in candidate_names:
        for column_name in column_names:
            if column_name == candidate_name:
                return column_name
    return None


def standardize_glm_dataframe(glm_df):
    glm_df = glm_df.copy()
    glm_df.columns = [str(column_name).strip().lower() for column_name in glm_df.columns]
    return glm_df


def get_task_regressor_names(design_matrix):
    task_regressor_names = []
    for column_name in design_matrix.columns:
        if column_name == "constant":
            continue
        if str(column_name).startswith("drift"):
            continue
        if str(column_name).startswith("ss_"):
            continue
        if str(column_name).startswith("aux_"):
            continue
        task_regressor_names.append(column_name)
    return task_regressor_names


def build_design_matrix_for_channel(raw_hb, raw_cw, channel_name, model_type, nuisance_method):
    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    if model_type == "canonical_glm":
        hrf_model = "glover"
        fir_delays = None
    elif model_type == "fir_glm":
        hrf_model = "fir"
        fir_delays = fir_delays_s
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    nuisance_df, nuisance_metadata = build_nuisance_dataframe(
        raw_hb=raw_hb,
        raw_cw=raw_cw,
        chromophore=chromophore,
        nuisance_method=nuisance_method,
        long_channel_name=channel_name,
    )

    design_matrix = make_first_level_design_matrix(
        raw_hb,
        stim_dur=stim_duration_s,
        drift_model="cosine",
        high_pass=drift_high_pass_hz,
        hrf_model=hrf_model,
        fir_delays=fir_delays,
        add_regs=nuisance_df,
        add_reg_names=None if nuisance_df is None else list(nuisance_df.columns),
    )

    return design_matrix, nuisance_metadata


def extract_canonical_glm_metrics(glm_df, task_regressor_name, channel_name, pipeline_label, file_label, target_pair_names, nuisance_metadata):
    glm_df = standardize_glm_dataframe(glm_df)
    condition_column_name = find_first_matching_column(glm_df.columns, ["condition", "regressor", "variable", "name"])
    beta_column_name = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])
    t_column_name = find_first_matching_column(glm_df.columns, ["t", "t_value", "tstat", "t_stat"])
    p_column_name = find_first_matching_column(glm_df.columns, ["p_value", "pvalue", "p"])

    canonical_rows = glm_df.loc[glm_df[condition_column_name].astype(str) == str(task_regressor_name)]
    if len(canonical_rows) == 0:
        canonical_rows = glm_df.loc[glm_df[condition_column_name].astype(str).str.contains(str(task_regressor_name), regex=False)]
    if len(canonical_rows) == 0:
        raise ValueError(f"No canonical GLM row matched task regressor {task_regressor_name} for {channel_name}")

    canonical_row = canonical_rows.iloc[0]
    pair_name = channel_name.split(" ")[0]
    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    result_row = {
        "subject": subject_name,
        "file_label": file_label,
        "amplitude_value": get_amplitude_value(file_label),
        "pipeline_label": pipeline_label,
        "channel_name": channel_name,
        "pair_name": pair_name,
        "chromophore": chromophore,
        "target_status": get_target_status_from_pair_name(pair_name, target_pair_names),
        "task_regressor": str(task_regressor_name),
        "beta": float(canonical_row[beta_column_name]),
        "t_value": float(canonical_row[t_column_name]),
        **nuisance_metadata,
    }
    if p_column_name is not None:
        result_row["p_value"] = float(canonical_row[p_column_name])

    return result_row


def extract_fir_glm_rows(glm_df, fir_regressor_names, channel_name, pipeline_label, file_label, target_pair_names, nuisance_metadata):
    glm_df = standardize_glm_dataframe(glm_df)
    condition_column_name = find_first_matching_column(glm_df.columns, ["condition", "regressor", "variable", "name"])
    beta_column_name = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])
    pair_name = channel_name.split(" ")[0]
    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    fir_rows = []
    for fir_regressor_name in fir_regressor_names:
        matching_rows = glm_df.loc[glm_df[condition_column_name].astype(str) == str(fir_regressor_name)]
        if len(matching_rows) == 0:
            matching_rows = glm_df.loc[glm_df[condition_column_name].astype(str).str.contains(str(fir_regressor_name), regex=False)]
        if len(matching_rows) == 0:
            continue

        fir_row = matching_rows.iloc[0]
        delay_string = str(fir_regressor_name).split("_")[-1]
        try:
            delay_value_s = float(delay_string)
        except ValueError:
            continue

        fir_rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "amplitude_value": get_amplitude_value(file_label),
            "pipeline_label": pipeline_label,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": chromophore,
            "target_status": get_target_status_from_pair_name(pair_name, target_pair_names),
            "fir_regressor": str(fir_regressor_name),
            "delay_s": delay_value_s,
            "beta": float(fir_row[beta_column_name]),
            **nuisance_metadata,
        })

    return pd.DataFrame(fir_rows)


def run_canonical_glm_for_channel(raw_hb, raw_cw, channel_name, pipeline_label, file_label, target_pair_names, nuisance_method):
    design_matrix, nuisance_metadata = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        raw_cw=raw_cw,
        channel_name=channel_name,
        model_type="canonical_glm",
        nuisance_method=nuisance_method,
    )
    task_regressor_names = get_task_regressor_names(design_matrix)
    if len(task_regressor_names) == 0:
        raise ValueError(f"No task regressors found for canonical GLM on {channel_name}")
    task_regressor_name = task_regressor_names[0]
    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model=canonical_noise_model)
    glm_df = glm_result.to_dataframe()
    return extract_canonical_glm_metrics(
        glm_df=glm_df,
        task_regressor_name=task_regressor_name,
        channel_name=channel_name,
        pipeline_label=pipeline_label,
        file_label=file_label,
        target_pair_names=target_pair_names,
        nuisance_metadata=nuisance_metadata,
    )


def run_fir_glm_for_channel(raw_hb, raw_cw, channel_name, pipeline_label, file_label, target_pair_names, nuisance_method):
    design_matrix, nuisance_metadata = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        raw_cw=raw_cw,
        channel_name=channel_name,
        model_type="fir_glm",
        nuisance_method=nuisance_method,
    )
    fir_regressor_names = get_task_regressor_names(design_matrix)
    if len(fir_regressor_names) == 0:
        raise ValueError(f"No task regressors found for FIR GLM on {channel_name}")
    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model=canonical_noise_model)
    glm_df = glm_result.to_dataframe()
    return extract_fir_glm_rows(
        glm_df=glm_df,
        fir_regressor_names=fir_regressor_names,
        channel_name=channel_name,
        pipeline_label=pipeline_label,
        file_label=file_label,
        target_pair_names=target_pair_names,
        nuisance_metadata=nuisance_metadata,
    )


def compute_block_average_channel_metrics(epochs_hb, channel_names, pipeline_label, file_label, target_pair_names):
    baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
    response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])

    rows = []
    for channel_name in channel_names:
        if channel_name not in epochs_hb.ch_names:
            continue

        channel_data = epochs_hb.copy().pick([channel_name]).get_data()[:, 0, :]
        baseline_values = channel_data[:, baseline_time_mask].mean(axis=1)
        response_values = channel_data[:, response_time_mask].mean(axis=1)
        score_values = response_values - baseline_values
        pair_name = channel_name.split(" ")[0]

        rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "amplitude_value": get_amplitude_value(file_label),
            "pipeline_label": pipeline_label,
            "channel_name": channel_name,
            "pair_name": pair_name,
            "chromophore": "hbo" if channel_name.endswith("hbo") else "hbr",
            "target_status": get_target_status_from_pair_name(pair_name, target_pair_names),
            "score": float(score_values.mean()),
            "score_std": float(score_values.std()),
        })

    return pd.DataFrame(rows)


def compute_fwhm_from_fir(delays_s, beta_values, chromophore):
    delays_s = np.array(delays_s, dtype=float)
    beta_values = np.array(beta_values, dtype=float)

    if len(beta_values) == 0:
        return np.nan

    if chromophore == "hbo":
        peak_index = int(np.argmax(beta_values))
        peak_value = beta_values[peak_index]
        if peak_value <= 0:
            return np.nan
        half_height = peak_value / 2.0
        above_half_mask = beta_values >= half_height
    else:
        peak_index = int(np.argmin(beta_values))
        peak_value = beta_values[peak_index]
        if peak_value >= 0:
            return np.nan
        half_height = peak_value / 2.0
        above_half_mask = beta_values <= half_height

    if above_half_mask.sum() < 2:
        return np.nan

    return delays_s[above_half_mask].max() - delays_s[above_half_mask].min()


# -----------------------------------------------------------------------------
# Null-shift helpers
# -----------------------------------------------------------------------------


def make_shifted_annotations(source_annotations, total_duration_s, shift_s):
    shifted_onsets = []
    shifted_durations = []
    shifted_descriptions = []

    for onset_s, duration_s, description in zip(
        source_annotations.onset,
        source_annotations.duration,
        source_annotations.description,
    ):
        shifted_onset = onset_s + shift_s
        while shifted_onset >= total_duration_s:
            shifted_onset -= total_duration_s
        shifted_onsets.append(shifted_onset)
        shifted_durations.append(duration_s)
        shifted_descriptions.append(description)

    order = np.argsort(shifted_onsets)
    return mne.Annotations(
        onset=np.array(shifted_onsets)[order],
        duration=np.array(shifted_durations)[order],
        description=np.array(shifted_descriptions)[order].tolist(),
    )


def get_shift_values(total_duration_s):
    if empirical_null_shift_count <= 0:
        return []
    max_shift_s = max(empirical_null_min_shift_s + 1.0, total_duration_s - empirical_null_min_shift_s)
    shift_values = np.linspace(empirical_null_min_shift_s, max_shift_s, empirical_null_shift_count + 2)[1:-1]
    return [float(shift_s) for shift_s in shift_values]


def summarize_canonical_target_separation(canonical_rows):
    canonical_df = pd.DataFrame(canonical_rows)
    if len(canonical_df) == 0:
        return pd.DataFrame()

    summary_df = (
        canonical_df
        .groupby(["chromophore", "target_status"], as_index=False)
        .agg(mean_score=("beta", "mean"))
    )
    wide_df = (
        summary_df
        .pivot_table(index=["chromophore"], columns="target_status", values="mean_score")
        .reset_index()
        .rename(columns={
            "true_target": "mean_target_score",
            "true_non_target": "mean_non_target_score",
        })
    )
    if "mean_target_score" not in wide_df.columns:
        wide_df["mean_target_score"] = np.nan
    if "mean_non_target_score" not in wide_df.columns:
        wide_df["mean_non_target_score"] = np.nan
    wide_df["target_minus_non_target_score"] = wide_df["mean_target_score"] - wide_df["mean_non_target_score"]
    return wide_df


def compute_empirical_null_shift_rows(raw_hb, raw_cw, pipeline_label, target_pair_names, available_long_channel_names, nuisance_method):
    if not compute_empirical_null:
        return []

    total_duration_s = raw_hb.times[-1]
    source_annotations = raw_hb.annotations
    shift_rows = []

    for shift_index, shift_s in enumerate(get_shift_values(total_duration_s), start=1):
        shifted_raw = raw_hb.copy()
        shifted_raw.set_annotations(make_shifted_annotations(source_annotations, total_duration_s, shift_s))

        shifted_canonical_rows = []
        for channel_name in available_long_channel_names:
            try:
                shifted_canonical_rows.append(
                    run_canonical_glm_for_channel(
                        raw_hb=shifted_raw,
                        raw_cw=raw_cw,
                        channel_name=channel_name,
                        pipeline_label=pipeline_label,
                        file_label=file_label,
                        target_pair_names=target_pair_names,
                        nuisance_method=nuisance_method,
                    )
                )
            except Exception:
                continue

        separation_df = summarize_canonical_target_separation(shifted_canonical_rows)
        if len(separation_df) == 0:
            continue

        for _, row in separation_df.iterrows():
            shift_rows.append({
                "subject": subject_name,
                "file_label": file_label,
                "pipeline_label": pipeline_label,
                "shift_index": shift_index,
                "shift_s": shift_s,
                "chromophore": row["chromophore"],
                "mean_target_score": row["mean_target_score"],
                "mean_non_target_score": row["mean_non_target_score"],
                "target_minus_non_target_score": row["target_minus_non_target_score"],
            })

    return shift_rows


# -----------------------------------------------------------------------------
# Subject-level probe
# -----------------------------------------------------------------------------


def build_subject_probe(subject_name_value):
    global subject_name
    global subject_dir
    global file_label

    subject_name = subject_name_value
    subject_dir = rs_data_dir / subject_name
    error_rows = []

    reference_file_label = "hrf_20"
    reference_file_path = get_file_path(subject_dir, reference_file_label)
    if not reference_file_path.exists():
        return None, [make_error_row(subject_name, reference_file_label, None, "reference_file_missing", str(reference_file_path))]

    try:
        file_label = reference_file_label
        reference_raw_cw = mne.io.read_raw_snirf(reference_file_path, preload=True, verbose=False)
        target_pair_names = get_true_injected_pair_names(reference_file_path, reference_raw_cw)
        target_channel_names = get_channel_names_from_pairs(target_pair_names)

        cw_channel_table = build_cw_channel_table(reference_raw_cw, subject_name, reference_file_label)
        long_pair_names = sorted(cw_channel_table.loc[cw_channel_table["group"] == "LS", "pair_name"].unique())
        non_target_pair_names = [pair_name for pair_name in long_pair_names if pair_name not in target_pair_names]
    except Exception as exc:
        return None, [make_error_row(subject_name, reference_file_label, None, "reference_load_or_truth", exc)]

    truth_summary_row = {
        "subject": subject_name,
        "n_true_target_pairs": len(target_pair_names),
        "n_true_non_target_pairs": len(non_target_pair_names),
        "target_pair_names": "|".join(target_pair_names),
    }

    quality_summary_rows = []
    channel_availability_rows = []
    epoch_plot_tables = []
    canonical_channel_rows = []
    block_average_channel_tables = []
    fir_channel_tables = []
    empirical_null_shift_rows = []
    nuisance_detail_rows = []

    for file_label_value in file_labels:
        file_label = file_label_value
        file_path = get_file_path(subject_dir, file_label)
        if not file_path.exists():
            error_rows.append(make_error_row(subject_name, file_label, None, "file_missing", str(file_path)))
            continue

        try:
            raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
            if file_label == "no_hrf":
                raw_cw = copy_valid_annotations(raw_cw, get_annotation_source_file_path(subject_dir))
        except Exception as exc:
            error_rows.append(make_error_row(subject_name, file_label, None, "file_read_or_annotations", exc))
            continue

        for pipeline_label in pipeline_labels:
            try:
                pipeline_spec = get_pipeline_spec(pipeline_label)
                pipeline_result = preprocess_raw_to_hb(raw_cw, pipeline_label)
                raw_hb = pipeline_result["raw_hb"]
                raw_cw_for_pipeline = pipeline_result["raw_cw"]
            except Exception as exc:
                error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "preprocess", exc))
                continue

            try:
                pair_quality_table = pipeline_result["pair_quality_table"].copy()
                pair_quality_table["pipeline_label"] = pipeline_label
                pair_quality_table["pruning_style"] = pipeline_spec["pruning_style"]
                pair_quality_table["do_tddr"] = pipeline_spec["do_tddr"]
                pair_quality_table["target_status"] = np.where(
                    pair_quality_table["pair_name"].isin(target_pair_names),
                    "true_target",
                    "true_non_target",
                )
                quality_summary_rows.append(pair_quality_table)

                if len(pipeline_result["nuisance_meta_df"]) > 0:
                    nuisance_meta_df = pipeline_result["nuisance_meta_df"].copy()
                    nuisance_meta_df["pipeline_label"] = pipeline_label
                    nuisance_detail_rows.append(nuisance_meta_df)

                epochs_hb, _, _ = make_epochs_from_raw_hb(raw_hb)
                if pipeline_label in ["PrefLocalSS", "PSSAuxPCA", "PBlockAvg"]:
                    target_epoch_plot_table = build_mean_epoch_plot_table(
                        epochs_hb=epochs_hb,
                        channel_names=target_channel_names,
                        pipeline_label=pipeline_label,
                        file_label=file_label,
                        trace_group_label="true_target_only",
                    )
                    if len(target_epoch_plot_table) > 0:
                        epoch_plot_tables.append(target_epoch_plot_table)

                all_available_long_channel_names = get_available_long_channel_names(raw_hb)
                available_target_pair_names = sorted(set([
                    channel_name.split(" ")[0]
                    for channel_name in all_available_long_channel_names
                    if channel_name.split(" ")[0] in target_pair_names
                ]))
                available_non_target_pair_names = sorted(set([
                    channel_name.split(" ")[0]
                    for channel_name in all_available_long_channel_names
                    if channel_name.split(" ")[0] in non_target_pair_names
                ]))

                channel_availability_rows.append({
                    "subject": subject_name,
                    "file_label": file_label,
                    "amplitude_value": get_amplitude_value(file_label),
                    "pipeline_label": pipeline_label,
                    "n_total_true_target_pairs": len(target_pair_names),
                    "n_available_true_target_pairs": len(available_target_pair_names),
                    "n_total_true_non_target_pairs": len(non_target_pair_names),
                    "n_available_true_non_target_pairs": len(available_non_target_pair_names),
                    "n_available_long_channels": len(all_available_long_channel_names),
                    "target_pair_retention_fraction": (len(available_target_pair_names) / len(target_pair_names)) if len(target_pair_names) > 0 else np.nan,
                    "non_target_pair_retention_fraction": (len(available_non_target_pair_names) / len(non_target_pair_names)) if len(non_target_pair_names) > 0 else np.nan,
                })
            except Exception as exc:
                error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "epoch_or_channel_setup", exc))
                continue

            if len(all_available_long_channel_names) == 0:
                error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "no_available_long_channels", "No surviving long-separation channels"))
                continue

            if pipeline_spec["estimation_method"] == "canonical_glm":
                for channel_name in all_available_long_channel_names:
                    try:
                        canonical_channel_rows.append(
                            run_canonical_glm_for_channel(
                                raw_hb=raw_hb,
                                raw_cw=raw_cw_for_pipeline,
                                channel_name=channel_name,
                                pipeline_label=pipeline_label,
                                file_label=file_label,
                                target_pair_names=target_pair_names,
                                nuisance_method=pipeline_spec["nuisance_method"],
                            )
                        )
                    except Exception as exc:
                        error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "canonical_channel_glm", exc, channel_name=channel_name))

                if file_label == "no_hrf" and pipeline_label in core_variability_pipelines:
                    try:
                        empirical_null_shift_rows.extend(
                            compute_empirical_null_shift_rows(
                                raw_hb=raw_hb,
                                raw_cw=raw_cw_for_pipeline,
                                pipeline_label=pipeline_label,
                                target_pair_names=target_pair_names,
                                available_long_channel_names=all_available_long_channel_names,
                                nuisance_method=pipeline_spec["nuisance_method"],
                            )
                        )
                    except Exception as exc:
                        error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "empirical_null", exc))

            elif pipeline_spec["estimation_method"] == "block_average":
                try:
                    block_average_channel_df = compute_block_average_channel_metrics(
                        epochs_hb=epochs_hb,
                        channel_names=all_available_long_channel_names,
                        pipeline_label=pipeline_label,
                        file_label=file_label,
                        target_pair_names=target_pair_names,
                    )
                    if len(block_average_channel_df) > 0:
                        block_average_channel_tables.append(block_average_channel_df)
                except Exception as exc:
                    error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "block_average", exc))

            elif pipeline_spec["estimation_method"] == "fir_glm":
                for channel_name in all_available_long_channel_names:
                    try:
                        fir_channel_df = run_fir_glm_for_channel(
                            raw_hb=raw_hb,
                            raw_cw=raw_cw_for_pipeline,
                            channel_name=channel_name,
                            pipeline_label=pipeline_label,
                            file_label=file_label,
                            target_pair_names=target_pair_names,
                            nuisance_method=pipeline_spec["nuisance_method"],
                        )
                        if len(fir_channel_df) > 0:
                            fir_channel_tables.append(fir_channel_df)
                    except Exception as exc:
                        error_rows.append(make_error_row(subject_name, file_label, pipeline_label, "fir_channel_glm", exc, channel_name=channel_name))

    quality_summary_df = pd.concat(quality_summary_rows, ignore_index=True) if len(quality_summary_rows) > 0 else pd.DataFrame()
    channel_availability_df = pd.DataFrame(channel_availability_rows) if len(channel_availability_rows) > 0 else pd.DataFrame()
    canonical_channel_df = pd.DataFrame(canonical_channel_rows) if len(canonical_channel_rows) > 0 else pd.DataFrame()
    block_average_channel_df = pd.concat(block_average_channel_tables, ignore_index=True) if len(block_average_channel_tables) > 0 else pd.DataFrame()
    fir_channel_df = pd.concat(fir_channel_tables, ignore_index=True) if len(fir_channel_tables) > 0 else pd.DataFrame()
    epoch_plot_df = pd.concat(epoch_plot_tables, ignore_index=True) if len(epoch_plot_tables) > 0 else pd.DataFrame()
    empirical_null_shift_df = pd.DataFrame(empirical_null_shift_rows) if len(empirical_null_shift_rows) > 0 else pd.DataFrame()
    nuisance_detail_df = pd.concat(nuisance_detail_rows, ignore_index=True) if len(nuisance_detail_rows) > 0 else pd.DataFrame()

    subject_result = {
        "truth_summary_row": truth_summary_row,
        "quality_summary_df": quality_summary_df,
        "channel_availability_df": channel_availability_df,
        "canonical_channel_df": canonical_channel_df,
        "block_average_channel_df": block_average_channel_df,
        "fir_channel_df": fir_channel_df,
        "epoch_plot_df": epoch_plot_df,
        "empirical_null_shift_df": empirical_null_shift_df,
        "nuisance_detail_df": nuisance_detail_df,
    }
    return subject_result, error_rows


# -----------------------------------------------------------------------------
# Summary / saving helpers
# -----------------------------------------------------------------------------


def add_fdr_q_values(canonical_channel_df):
    if len(canonical_channel_df) == 0 or "p_value" not in canonical_channel_df.columns:
        return canonical_channel_df

    canonical_channel_df = canonical_channel_df.copy()
    canonical_channel_df["q_value_bh"] = np.nan

    group_columns = ["subject", "file_label", "pipeline_label", "chromophore"]
    for _, group_df in canonical_channel_df.groupby(group_columns):
        group_indices = group_df.index.to_list()
        q_values = benjamini_hochberg(group_df["p_value"].to_numpy())
        canonical_channel_df.loc[group_indices, "q_value_bh"] = q_values

    return canonical_channel_df


def build_roi_score_df(canonical_channel_df, block_average_channel_df):
    roi_summary_tables = []

    if len(canonical_channel_df) > 0:
        canonical_roi_df = (
            canonical_channel_df
            .groupby(["subject", "file_label", "amplitude_value", "pipeline_label", "chromophore", "target_status"], as_index=False)
            .agg(
                roi_mean_score=("beta", "mean"),
                roi_std_score=("beta", "std"),
                n_channels=("channel_name", "count"),
            )
        )
        canonical_roi_df["score_type"] = "canonical_beta"
        roi_summary_tables.append(canonical_roi_df)

    if len(block_average_channel_df) > 0:
        block_roi_df = (
            block_average_channel_df
            .groupby(["subject", "file_label", "amplitude_value", "pipeline_label", "chromophore", "target_status"], as_index=False)
            .agg(
                roi_mean_score=("score", "mean"),
                roi_std_score=("score", "std"),
                n_channels=("channel_name", "count"),
            )
        )
        block_roi_df["score_type"] = "block_average_score"
        roi_summary_tables.append(block_roi_df)

    return pd.concat(roi_summary_tables, ignore_index=True) if len(roi_summary_tables) > 0 else pd.DataFrame()


def build_target_vs_nontarget_summary_df(canonical_channel_df, block_average_channel_df):
    separation_summary_tables = []

    if len(canonical_channel_df) > 0:
        canonical_separation_df = (
            canonical_channel_df
            .groupby(["subject", "file_label", "pipeline_label", "chromophore", "target_status"], as_index=False)
            .agg(mean_score=("beta", "mean"))
        )
        canonical_separation_wide_df = (
            canonical_separation_df
            .pivot_table(
                index=["subject", "file_label", "pipeline_label", "chromophore"],
                columns="target_status",
                values="mean_score",
            )
            .reset_index()
            .rename(columns={
                "true_non_target": "mean_non_target_score",
                "true_target": "mean_target_score",
            })
        )
        if "mean_target_score" not in canonical_separation_wide_df.columns:
            canonical_separation_wide_df["mean_target_score"] = np.nan
        if "mean_non_target_score" not in canonical_separation_wide_df.columns:
            canonical_separation_wide_df["mean_non_target_score"] = np.nan
        canonical_separation_wide_df["target_minus_non_target_score"] = (
            canonical_separation_wide_df["mean_target_score"] - canonical_separation_wide_df["mean_non_target_score"]
        )
        canonical_separation_wide_df["score_type"] = "canonical_beta"
        separation_summary_tables.append(canonical_separation_wide_df)

    if len(block_average_channel_df) > 0:
        block_separation_df = (
            block_average_channel_df
            .groupby(["subject", "file_label", "pipeline_label", "chromophore", "target_status"], as_index=False)
            .agg(mean_score=("score", "mean"))
        )
        block_separation_wide_df = (
            block_separation_df
            .pivot_table(
                index=["subject", "file_label", "pipeline_label", "chromophore"],
                columns="target_status",
                values="mean_score",
            )
            .reset_index()
            .rename(columns={
                "true_non_target": "mean_non_target_score",
                "true_target": "mean_target_score",
            })
        )
        if "mean_target_score" not in block_separation_wide_df.columns:
            block_separation_wide_df["mean_target_score"] = np.nan
        if "mean_non_target_score" not in block_separation_wide_df.columns:
            block_separation_wide_df["mean_non_target_score"] = np.nan
        block_separation_wide_df["target_minus_non_target_score"] = (
            block_separation_wide_df["mean_target_score"] - block_separation_wide_df["mean_non_target_score"]
        )
        block_separation_wide_df["score_type"] = "block_average_score"
        separation_summary_tables.append(block_separation_wide_df)

    return pd.concat(separation_summary_tables, ignore_index=True) if len(separation_summary_tables) > 0 else pd.DataFrame()


def build_parametric_null_summary_df(canonical_channel_df):
    if len(canonical_channel_df) == 0 or "p_value" not in canonical_channel_df.columns:
        return pd.DataFrame()

    canonical_null_df = canonical_channel_df.loc[canonical_channel_df["file_label"] == "no_hrf"].copy()
    if len(canonical_null_df) == 0:
        return pd.DataFrame()

    canonical_null_df["is_false_positive_p_lt_0_05"] = canonical_null_df["p_value"] < 0.05
    if "q_value_bh" in canonical_null_df.columns:
        canonical_null_df["is_false_positive_q_lt_0_05"] = canonical_null_df["q_value_bh"] < 0.05
    else:
        canonical_null_df["is_false_positive_q_lt_0_05"] = np.nan

    return (
        canonical_null_df
        .groupby(["subject", "pipeline_label", "chromophore", "target_status"], as_index=False)
        .agg(
            false_positive_rate_p_lt_0_05=("is_false_positive_p_lt_0_05", "mean"),
            false_positive_rate_q_lt_0_05=("is_false_positive_q_lt_0_05", "mean"),
            mean_abs_beta=("beta", lambda values: np.mean(np.abs(values))),
            n_channels=("channel_name", "count"),
        )
    )


def build_empirical_null_pvalue_df(target_vs_nontarget_summary_df, empirical_null_shift_df):
    if len(target_vs_nontarget_summary_df) == 0 or len(empirical_null_shift_df) == 0:
        return pd.DataFrame()

    observed_df = target_vs_nontarget_summary_df.loc[
        (target_vs_nontarget_summary_df["score_type"] == "canonical_beta") &
        (target_vs_nontarget_summary_df["file_label"].isin(["no_hrf", "hrf_20"])) &
        (target_vs_nontarget_summary_df["pipeline_label"].isin(core_variability_pipelines))
    ].copy()

    empirical_rows = []
    null_group_keys = ["subject", "pipeline_label", "chromophore"]
    for _, observed_row in observed_df.iterrows():
        null_rows = empirical_null_shift_df.loc[
            (empirical_null_shift_df["subject"] == observed_row["subject"]) &
            (empirical_null_shift_df["pipeline_label"] == observed_row["pipeline_label"]) &
            (empirical_null_shift_df["chromophore"] == observed_row["chromophore"])
        ]
        if len(null_rows) == 0:
            continue

        null_values = null_rows["target_minus_non_target_score"].to_numpy(dtype=float)
        observed_value = float(observed_row["target_minus_non_target_score"])
        if observed_row["chromophore"] == "hbo":
            empirical_p_value = (np.sum(null_values >= observed_value) + 1) / (len(null_values) + 1)
        else:
            empirical_p_value = (np.sum(null_values <= observed_value) + 1) / (len(null_values) + 1)

        empirical_rows.append({
            "subject": observed_row["subject"],
            "file_label": observed_row["file_label"],
            "pipeline_label": observed_row["pipeline_label"],
            "chromophore": observed_row["chromophore"],
            "observed_target_minus_non_target_score": observed_value,
            "null_shift_mean": float(np.mean(null_values)),
            "null_shift_std": float(np.std(null_values)),
            "null_shift_min": float(np.min(null_values)),
            "null_shift_max": float(np.max(null_values)),
            "empirical_p_value": empirical_p_value,
            "n_null_shifts": len(null_values),
        })

    return pd.DataFrame(empirical_rows)


def build_fir_shape_summary_df(fir_channel_df):
    if len(fir_channel_df) == 0:
        return pd.DataFrame()

    fir_shape_rows = []
    grouped_fir_df = fir_channel_df.groupby(["subject", "file_label", "pipeline_label", "chromophore", "target_status"])
    for (subject_name_value, file_label_value, pipeline_label_value, chromophore, target_status), group_df in grouped_fir_df:
        fir_roi_timecourse_df = (
            group_df.groupby(["delay_s"], as_index=False)
            .agg(roi_mean_beta=("beta", "mean"))
            .sort_values("delay_s")
        )
        delays_s = fir_roi_timecourse_df["delay_s"].to_numpy()
        beta_values = fir_roi_timecourse_df["roi_mean_beta"].to_numpy()
        if len(beta_values) == 0:
            continue

        peak_index = int(np.argmax(beta_values)) if chromophore == "hbo" else int(np.argmin(beta_values))
        fir_shape_rows.append({
            "subject": subject_name_value,
            "file_label": file_label_value,
            "amplitude_value": get_amplitude_value(file_label_value),
            "pipeline_label": pipeline_label_value,
            "chromophore": chromophore,
            "target_status": target_status,
            "peak_delay_s": float(delays_s[peak_index]),
            "peak_beta": float(beta_values[peak_index]),
            "fwhm_s": compute_fwhm_from_fir(delays_s, beta_values, chromophore),
        })
    return pd.DataFrame(fir_shape_rows)


def build_variability_summary_df(target_vs_nontarget_summary_df):
    if len(target_vs_nontarget_summary_df) == 0:
        return pd.DataFrame()

    variability_df = target_vs_nontarget_summary_df.loc[
        (target_vs_nontarget_summary_df["score_type"] == "canonical_beta") &
        (target_vs_nontarget_summary_df["pipeline_label"].isin(core_variability_pipelines)) &
        (target_vs_nontarget_summary_df["file_label"] == "hrf_20")
    ].copy()

    if len(variability_df) == 0:
        return pd.DataFrame()

    return (
        variability_df
        .groupby(["subject", "chromophore"], as_index=False)
        .agg(
            mean_target_minus_non_target=("target_minus_non_target_score", "mean"),
            std_across_pipelines=("target_minus_non_target_score", "std"),
            min_across_pipelines=("target_minus_non_target_score", "min"),
            max_across_pipelines=("target_minus_non_target_score", "max"),
            n_pipelines=("pipeline_label", "count"),
        )
        .assign(range_across_pipelines=lambda df: df["max_across_pipelines"] - df["min_across_pipelines"])
    )


def build_pairwise_pipeline_delta_df(target_vs_nontarget_summary_df):
    if len(target_vs_nontarget_summary_df) == 0:
        return pd.DataFrame()

    canonical_df = target_vs_nontarget_summary_df.loc[
        (target_vs_nontarget_summary_df["score_type"] == "canonical_beta") &
        (target_vs_nontarget_summary_df["file_label"] == "hrf_20") &
        (target_vs_nontarget_summary_df["pipeline_label"].isin(core_variability_pipelines))
    ].copy()
    if len(canonical_df) == 0:
        return pd.DataFrame()

    pivot_df = canonical_df.pivot_table(
        index=["subject", "chromophore"],
        columns="pipeline_label",
        values="target_minus_non_target_score",
    ).reset_index()

    pairwise_rows = []
    for _, row in pivot_df.iterrows():
        for left_pipeline in core_variability_pipelines:
            for right_pipeline in core_variability_pipelines:
                if left_pipeline >= right_pipeline:
                    continue
                if left_pipeline not in row.index or right_pipeline not in row.index:
                    continue
                left_value = row[left_pipeline]
                right_value = row[right_pipeline]
                pairwise_rows.append({
                    "subject": row["subject"],
                    "chromophore": row["chromophore"],
                    "left_pipeline": left_pipeline,
                    "right_pipeline": right_pipeline,
                    "left_minus_right": left_value - right_value,
                    "abs_left_minus_right": abs(left_value - right_value),
                })
    return pd.DataFrame(pairwise_rows)


def save_epoch_trace_figure(epoch_plot_df):
    if len(epoch_plot_df) == 0 or not save_html_figures:
        return

    epoch_figure = px.line(
        epoch_plot_df,
        x="time_s",
        y="signal",
        color="pipeline_label",
        facet_row="chromophore",
        facet_col="file_label",
        category_orders={"file_label": plot_condition_order},
        title="Pre-estimation event-locked traces on true target channels (context only)",
    )
    epoch_figure.update_layout(width=1450, height=900)
    epoch_figure.write_html(output_figures_dir / f"{output_prefix}_epoch_traces_targets_only.html")


def save_roi_score_figure(roi_score_df):
    if len(roi_score_df) == 0 or not save_html_figures:
        return

    roi_figure = px.scatter(
        roi_score_df.sort_values(["score_type", "chromophore", "pipeline_label", "file_label", "target_status"]),
        x="file_label",
        y="roi_mean_score",
        color="pipeline_label",
        symbol="target_status",
        facet_row="chromophore",
        facet_col="score_type",
        category_orders={
            "file_label": plot_condition_order,
            "target_status": plot_target_status_order,
            "score_type": plot_score_type_order,
            "chromophore": plot_chromophore_order,
        },
        hover_data=["subject", "n_channels", "roi_std_score"],
        title="ROI scores by pipeline, condition, and target status",
    )
    roi_figure.update_layout(width=1500, height=850)
    roi_figure.write_html(output_figures_dir / f"{output_prefix}_roi_scores_by_target_status.html")


def save_target_minus_non_target_figure(target_vs_nontarget_summary_df):
    if len(target_vs_nontarget_summary_df) == 0 or not save_html_figures:
        return

    separation_figure = px.bar(
        target_vs_nontarget_summary_df.sort_values(["score_type", "chromophore", "file_label", "pipeline_label", "subject"]),
        x="pipeline_label",
        y="target_minus_non_target_score",
        color="file_label",
        pattern_shape="subject",
        barmode="group",
        facet_row="chromophore",
        facet_col="score_type",
        category_orders={
            "file_label": plot_condition_order,
            "score_type": plot_score_type_order,
            "chromophore": plot_chromophore_order,
        },
        hover_data=["subject", "mean_target_score", "mean_non_target_score"],
        title="Target minus non-target separation by pipeline",
    )
    separation_figure.update_layout(width=1650, height=900)
    separation_figure.write_html(output_figures_dir / f"{output_prefix}_target_minus_non_target_scores.html")


def save_channel_distribution_figure(channel_df, value_column, score_label, figure_suffix):
    if len(channel_df) == 0 or not save_html_figures:
        return

    figure_df = channel_df.loc[channel_df["file_label"].isin(plot_condition_order)].copy()
    distribution_figure = px.strip(
        figure_df,
        x="pipeline_label",
        y=value_column,
        color="target_status",
        facet_row="chromophore",
        facet_col="file_label",
        category_orders={
            "file_label": plot_condition_order,
            "target_status": plot_target_status_order,
            "chromophore": plot_chromophore_order,
        },
        hover_data=["subject", "channel_name", "pair_name"],
        title=f"Channel-level {score_label}: true targets vs true non-targets",
    )
    distribution_figure.update_layout(width=1650, height=900)
    distribution_figure.write_html(output_figures_dir / f"{output_prefix}_{figure_suffix}.html")


def save_variability_figure(variability_summary_df):
    if len(variability_summary_df) == 0 or not save_html_figures:
        return

    variability_figure = px.bar(
        variability_summary_df,
        x="subject",
        y="range_across_pipelines",
        color="chromophore",
        barmode="group",
        hover_data=["std_across_pipelines", "mean_target_minus_non_target", "n_pipelines"],
        title="Within-subject variability across core nuisance-model pipelines (hrf_20)",
    )
    variability_figure.update_layout(width=1400, height=650)
    variability_figure.write_html(output_figures_dir / f"{output_prefix}_core_pipeline_variability.html")


def save_empirical_null_figure(empirical_null_pvalue_df):
    if len(empirical_null_pvalue_df) == 0 or not save_html_figures:
        return

    empirical_figure = px.scatter(
        empirical_null_pvalue_df,
        x="null_shift_mean",
        y="observed_target_minus_non_target_score",
        color="pipeline_label",
        symbol="file_label",
        facet_row="chromophore",
        hover_data=["subject", "empirical_p_value", "null_shift_std", "n_null_shifts"],
        title="Observed separation vs empirical no-HRF shift null",
    )
    empirical_figure.update_layout(width=1450, height=800)
    empirical_figure.write_html(output_figures_dir / f"{output_prefix}_empirical_null_comparison.html")


def save_outputs(truth_summary_rows, all_quality_tables, all_channel_availability_tables,
                 all_canonical_channel_tables, all_block_average_channel_tables,
                 all_fir_channel_tables, all_epoch_plot_tables, all_error_rows, run_log_rows,
                 all_empirical_null_shift_tables, all_nuisance_detail_tables):
    truth_summary_df = pd.DataFrame(truth_summary_rows)
    quality_summary_df = pd.concat(all_quality_tables, ignore_index=True) if len(all_quality_tables) > 0 else pd.DataFrame()
    channel_availability_df = pd.concat(all_channel_availability_tables, ignore_index=True) if len(all_channel_availability_tables) > 0 else pd.DataFrame()
    canonical_channel_df = pd.concat(all_canonical_channel_tables, ignore_index=True) if len(all_canonical_channel_tables) > 0 else pd.DataFrame()
    canonical_channel_df = add_fdr_q_values(canonical_channel_df)
    block_average_channel_df = pd.concat(all_block_average_channel_tables, ignore_index=True) if len(all_block_average_channel_tables) > 0 else pd.DataFrame()
    fir_channel_df = pd.concat(all_fir_channel_tables, ignore_index=True) if len(all_fir_channel_tables) > 0 else pd.DataFrame()
    epoch_plot_df = pd.concat(all_epoch_plot_tables, ignore_index=True) if len(all_epoch_plot_tables) > 0 else pd.DataFrame()
    empirical_null_shift_df = pd.concat(all_empirical_null_shift_tables, ignore_index=True) if len(all_empirical_null_shift_tables) > 0 else pd.DataFrame()
    nuisance_detail_df = pd.concat(all_nuisance_detail_tables, ignore_index=True) if len(all_nuisance_detail_tables) > 0 else pd.DataFrame()
    error_log_df = pd.DataFrame(all_error_rows)
    run_log_df = pd.DataFrame(run_log_rows)

    truth_summary_df.to_csv(output_tables_dir / f"{output_prefix}_truth_summary.csv", index=False)
    quality_summary_df.to_csv(output_tables_dir / f"{output_prefix}_quality_summary.csv", index=False)
    channel_availability_df.to_csv(output_tables_dir / f"{output_prefix}_channel_availability_summary.csv", index=False)
    if len(canonical_channel_df) > 0:
        canonical_channel_df.to_csv(output_tables_dir / f"{output_prefix}_canonical_channel_metrics.csv", index=False)
    if len(block_average_channel_df) > 0:
        block_average_channel_df.to_csv(output_tables_dir / f"{output_prefix}_block_average_channel_metrics.csv", index=False)
    if len(fir_channel_df) > 0:
        fir_channel_df.to_csv(output_tables_dir / f"{output_prefix}_fir_channel_metrics.csv", index=False)
    if len(empirical_null_shift_df) > 0:
        empirical_null_shift_df.to_csv(output_tables_dir / f"{output_prefix}_empirical_null_shift_summary.csv", index=False)
    if len(nuisance_detail_df) > 0:
        nuisance_detail_df.to_csv(output_tables_dir / f"{output_prefix}_nuisance_detail_summary.csv", index=False)
    if len(error_log_df) > 0:
        error_log_df.to_csv(output_tables_dir / f"{output_prefix}_error_log.csv", index=False)
    if len(run_log_df) > 0:
        run_log_df.to_csv(output_tables_dir / f"{output_prefix}_run_log.csv", index=False)

    roi_score_df = build_roi_score_df(canonical_channel_df, block_average_channel_df)
    if len(roi_score_df) > 0:
        roi_score_df.to_csv(output_tables_dir / f"{output_prefix}_roi_scores.csv", index=False)

    target_vs_nontarget_summary_df = build_target_vs_nontarget_summary_df(canonical_channel_df, block_average_channel_df)
    if len(target_vs_nontarget_summary_df) > 0:
        target_vs_nontarget_summary_df.to_csv(output_tables_dir / f"{output_prefix}_target_vs_nontarget_summary.csv", index=False)

    parametric_null_summary_df = build_parametric_null_summary_df(canonical_channel_df)
    if len(parametric_null_summary_df) > 0:
        parametric_null_summary_df.to_csv(output_tables_dir / f"{output_prefix}_parametric_null_summary.csv", index=False)

    empirical_null_pvalue_df = build_empirical_null_pvalue_df(target_vs_nontarget_summary_df, empirical_null_shift_df)
    if len(empirical_null_pvalue_df) > 0:
        empirical_null_pvalue_df.to_csv(output_tables_dir / f"{output_prefix}_empirical_null_pvalues.csv", index=False)

    variability_summary_df = build_variability_summary_df(target_vs_nontarget_summary_df)
    if len(variability_summary_df) > 0:
        variability_summary_df.to_csv(output_tables_dir / f"{output_prefix}_variability_summary.csv", index=False)

    pairwise_pipeline_delta_df = build_pairwise_pipeline_delta_df(target_vs_nontarget_summary_df)
    if len(pairwise_pipeline_delta_df) > 0:
        pairwise_pipeline_delta_df.to_csv(output_tables_dir / f"{output_prefix}_pairwise_pipeline_deltas.csv", index=False)

    fir_shape_summary_df = build_fir_shape_summary_df(fir_channel_df)
    if len(fir_shape_summary_df) > 0:
        fir_shape_summary_df.to_csv(output_tables_dir / f"{output_prefix}_fir_shape_summary.csv", index=False)

    save_epoch_trace_figure(epoch_plot_df)
    save_roi_score_figure(roi_score_df)
    save_target_minus_non_target_figure(target_vs_nontarget_summary_df)
    save_channel_distribution_figure(canonical_channel_df, "beta", "canonical beta", "canonical_channel_distributions")
    save_channel_distribution_figure(block_average_channel_df, "score", "block-average score", "block_average_channel_distributions")
    save_variability_figure(variability_summary_df)
    save_empirical_null_figure(empirical_null_pvalue_df)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    subject_names_to_run = resolve_subject_names()
    print("Subjects to run:", subject_names_to_run)

    truth_summary_rows = []
    all_quality_tables = []
    all_channel_availability_tables = []
    all_canonical_channel_tables = []
    all_block_average_channel_tables = []
    all_fir_channel_tables = []
    all_epoch_plot_tables = []
    all_error_rows = []
    run_log_rows = []
    all_empirical_null_shift_tables = []
    all_nuisance_detail_tables = []

    for subject_name_value in subject_names_to_run:
        start_time = datetime.utcnow()
        try:
            subject_result, error_rows = build_subject_probe(subject_name_value)
            all_error_rows.extend(error_rows)

            if subject_result is None:
                print(f"Skipping {subject_name_value}: subject probe returned None")
                run_log_rows.append({
                    "timestamp_utc": datetime.utcnow().isoformat(),
                    "subject": subject_name_value,
                    "status": "skipped",
                })
                if save_intermediate_after_each_subject:
                    save_outputs(
                        truth_summary_rows,
                        all_quality_tables,
                        all_channel_availability_tables,
                        all_canonical_channel_tables,
                        all_block_average_channel_tables,
                        all_fir_channel_tables,
                        all_epoch_plot_tables,
                        all_error_rows,
                        run_log_rows,
                        all_empirical_null_shift_tables,
                        all_nuisance_detail_tables,
                    )
                continue

            truth_summary_rows.append(subject_result["truth_summary_row"])

            if len(subject_result["quality_summary_df"]) > 0:
                all_quality_tables.append(subject_result["quality_summary_df"])
            if len(subject_result["channel_availability_df"]) > 0:
                all_channel_availability_tables.append(subject_result["channel_availability_df"])
            if len(subject_result["canonical_channel_df"]) > 0:
                all_canonical_channel_tables.append(subject_result["canonical_channel_df"])
            if len(subject_result["block_average_channel_df"]) > 0:
                all_block_average_channel_tables.append(subject_result["block_average_channel_df"])
            if len(subject_result["fir_channel_df"]) > 0:
                all_fir_channel_tables.append(subject_result["fir_channel_df"])
            if len(subject_result["epoch_plot_df"]) > 0:
                all_epoch_plot_tables.append(subject_result["epoch_plot_df"])
            if len(subject_result["empirical_null_shift_df"]) > 0:
                all_empirical_null_shift_tables.append(subject_result["empirical_null_shift_df"])
            if len(subject_result["nuisance_detail_df"]) > 0:
                all_nuisance_detail_tables.append(subject_result["nuisance_detail_df"])

            elapsed_seconds = (datetime.utcnow() - start_time).total_seconds()
            run_log_rows.append({
                "timestamp_utc": datetime.utcnow().isoformat(),
                "subject": subject_name_value,
                "status": "finished",
                "elapsed_seconds": elapsed_seconds,
                "n_errors_for_subject": len(error_rows),
            })
            print(f"Finished {subject_name_value} in {elapsed_seconds:.1f} s")

        except Exception as exc:
            all_error_rows.append(make_error_row(subject_name_value, None, None, "subject_level_crash", exc))
            run_log_rows.append({
                "timestamp_utc": datetime.utcnow().isoformat(),
                "subject": subject_name_value,
                "status": "crashed",
                "elapsed_seconds": (datetime.utcnow() - start_time).total_seconds(),
            })
            print(f"Crashed on {subject_name_value}: {exc}")

        if save_intermediate_after_each_subject:
            save_outputs(
                truth_summary_rows,
                all_quality_tables,
                all_channel_availability_tables,
                all_canonical_channel_tables,
                all_block_average_channel_tables,
                all_fir_channel_tables,
                all_epoch_plot_tables,
                all_error_rows,
                run_log_rows,
                all_empirical_null_shift_tables,
                all_nuisance_detail_tables,
            )

    save_outputs(
        truth_summary_rows,
        all_quality_tables,
        all_channel_availability_tables,
        all_canonical_channel_tables,
        all_block_average_channel_tables,
        all_fir_channel_tables,
        all_epoch_plot_tables,
        all_error_rows,
        run_log_rows,
        all_empirical_null_shift_tables,
        all_nuisance_detail_tables,
    )
    print(f"Saved {output_prefix} outputs.")


if __name__ == "__main__":
    main()
