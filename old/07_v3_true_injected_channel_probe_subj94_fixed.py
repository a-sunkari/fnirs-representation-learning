import pandas as pd
import numpy as np
import mne
import mne_nirs
import h5py
import plotly.express as px

from pathlib import Path

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

# Keep this as a hard-subject debugging script first.
subject_names_to_run = ["Subj86", "Subj91", "Subj92", "Subj94", "Subj95", "Subj96",
    "Subj97", "Subj98", "Subj99", "Subj100", "Subj101",
    "Subj102", "Subj103", "Subj104"]
file_labels = ["no_hrf", "hrf_20"]

# Keep outputs separate from earlier runs.
output_prefix = "v4_overnight_true_injected_low_amp_v3"

include_motion_stress_pipeline = False
pipeline_labels = [
    "Pref",
    "PnoSS",
    "PBlockAvg",
    "PFIR",
    "PLooseQC",
]
if include_motion_stress_pipeline:
    pipeline_labels.append("PnoMC")

short_separation_threshold_m = 0.015
long_separation_threshold_m = 0.025

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
# SNIRF truth-label helpers
# -----------------------------------------------------------------------------


def read_measurement_data_type_labels(snirf_file_path):
    """
    Read measurementList*/dataTypeLabel from a semisynthetic SNIRF file.
    The uploaded MATLAB script snirf_add_hrf_aux_labels.m writes:
        1 = HRF added
        0 = no HRF
    for each measurement channel.
    """
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
    data_type_labels = np.array([label_value for _, label_value in measurement_entries], dtype=int)
    return data_type_labels


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

    true_injected_channel_names = aligned_channel_names[data_type_labels == 1].tolist()
    return true_injected_channel_names


def get_true_injected_pair_names(reference_snirf_file_path, raw_cw):
    true_injected_channel_names = get_true_injected_channel_names(reference_snirf_file_path, raw_cw)
    true_injected_pair_names = sorted(set([channel_name.split(" ")[0] for channel_name in true_injected_channel_names]))
    return true_injected_pair_names


def get_channel_names_from_pairs(pair_names):
    hbo_channel_names = [pair_name + " hbo" for pair_name in pair_names]
    hbr_channel_names = [pair_name + " hbr" for pair_name in pair_names]
    return hbo_channel_names + hbr_channel_names


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

    bad_pair_names = pair_quality_table.loc[bad_mask, "pair_name"].tolist()
    return bad_pair_names


def apply_bad_pairs_to_hb(raw_hb, bad_pair_names):
    raw_hb = raw_hb.copy()
    hb_bad_channel_names = []

    for channel_name in raw_hb.ch_names:
        pair_name = channel_name.split(" ")[0]
        if pair_name in bad_pair_names:
            hb_bad_channel_names.append(channel_name)

    raw_hb.info["bads"] = hb_bad_channel_names
    return raw_hb


def get_available_short_channel_names(raw_hb, chromophore):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)
    short_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "SS") &
        (hb_channel_table["chromophore"] == chromophore),
        "channel_name",
    ].tolist()
    short_channel_names = [
        channel_name for channel_name in short_channel_names
        if channel_name not in raw_hb.info["bads"]
    ]
    return short_channel_names


def build_short_regressor_signal(raw_hb, chromophore, ss_method):
    short_channel_names = get_available_short_channel_names(raw_hb, chromophore)
    if len(short_channel_names) == 0:
        return None

    short_channel_data = raw_hb.copy().pick(short_channel_names).get_data()

    if ss_method == "average":
        return short_channel_data.mean(axis=0)

    if ss_method == "pca":
        centered_data = short_channel_data - short_channel_data.mean(axis=1, keepdims=True)
        data_matrix = centered_data.T
        _, _, right_singular_vectors = np.linalg.svd(data_matrix, full_matrices=False)
        first_component_weights = right_singular_vectors[0]
        first_component_timecourse = data_matrix @ first_component_weights
        return first_component_timecourse

    raise ValueError(f"Unknown ss_method: {ss_method}")


def apply_short_regression_to_hb(raw_hb, ss_method):
    raw_hb = raw_hb.copy().load_data()
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    for chromophore in ["hbo", "hbr"]:
        short_regressor_signal = build_short_regressor_signal(raw_hb, chromophore, ss_method)
        if short_regressor_signal is None:
            continue

        regressor_matrix = np.column_stack([
            np.ones(len(short_regressor_signal)),
            short_regressor_signal,
        ])

        long_channel_names = hb_channel_table.loc[
            (hb_channel_table["group"] == "LS") &
            (hb_channel_table["chromophore"] == chromophore),
            "channel_name",
        ].tolist()

        for channel_name in long_channel_names:
            if channel_name in raw_hb.info["bads"]:
                continue

            channel_index = raw_hb.ch_names.index(channel_name)
            channel_signal = raw_hb.get_data(picks=[channel_name])[0]
            regression_coefficients, _, _, _ = np.linalg.lstsq(regressor_matrix, channel_signal, rcond=None)
            fitted_signal = regressor_matrix @ regression_coefficients
            cleaned_signal = channel_signal - fitted_signal + regression_coefficients[0]
            raw_hb._data[channel_index, :] = cleaned_signal

    return raw_hb


def get_available_long_channel_names(raw_hb):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)
    long_channel_names = hb_channel_table.loc[
        hb_channel_table["group"] == "LS",
        "channel_name",
    ].tolist()
    long_channel_names = [
        channel_name for channel_name in long_channel_names
        if channel_name not in raw_hb.info["bads"]
    ]
    return sorted(long_channel_names)


def get_target_status_from_pair_name(pair_name, target_pair_names):
    return "true_target" if pair_name in target_pair_names else "true_non_target"


def get_pipeline_spec(pipeline_label):
    if pipeline_label == "Pref":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "average"}
    if pipeline_label == "PnoSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "none"}
    if pipeline_label == "PBlockAvg":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "block_average", "ss_method": "average"}
    if pipeline_label == "PFIR":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "fir_glm", "ss_method": "average"}
    if pipeline_label == "PLooseQC":
        return {"pruning_style": "loose_sci", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "average"}
    if pipeline_label == "PnoMC":
        return {"pruning_style": "strict_combined", "do_tddr": False, "estimation_method": "canonical_glm", "ss_method": "average"}
    raise ValueError(f"Unknown pipeline_label: {pipeline_label}")


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

    if pipeline_label == "PBlockAvg":
        raw_hb = apply_short_regression_to_hb(raw_hb, "average")

    return {
        "pipeline_label": pipeline_label,
        "raw_od": processed_od,
        "raw_hb": raw_hb,
        "channel_quality_table": channel_quality_table,
        "pair_quality_table": pair_quality_table,
        "bad_pair_names": bad_pair_names,
        "pruning_style": pruning_style,
        "do_tddr": do_tddr,
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
        task_regressor_names.append(column_name)
    return task_regressor_names


def build_design_matrix_for_channel(raw_hb, channel_name, model_type, ss_method):
    if model_type == "canonical_glm":
        hrf_model = "glover"
        fir_delays = None
    elif model_type == "fir_glm":
        hrf_model = "fir"
        fir_delays = fir_delays_s
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    add_regs = None
    add_reg_names = None
    if ss_method != "none":
        chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"
        short_regressor_signal = build_short_regressor_signal(raw_hb, chromophore, ss_method)
        if short_regressor_signal is not None:
            add_regs = pd.DataFrame({f"ss_{chromophore}": short_regressor_signal})
            add_reg_names = [f"ss_{chromophore}"]

    design_matrix = make_first_level_design_matrix(
        raw_hb,
        stim_dur=stim_duration_s,
        drift_model="cosine",
        high_pass=drift_high_pass_hz,
        hrf_model=hrf_model,
        fir_delays=fir_delays,
        add_regs=add_regs,
        add_reg_names=add_reg_names,
    )

    return design_matrix


def extract_canonical_glm_metrics(glm_df, task_regressor_name, channel_name, pipeline_label, file_label, target_pair_names):
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
    }
    if p_column_name is not None:
        result_row["p_value"] = float(canonical_row[p_column_name])

    return result_row


def extract_fir_glm_rows(glm_df, fir_regressor_names, channel_name, pipeline_label, file_label, target_pair_names):
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
        })

    return pd.DataFrame(fir_rows)


def run_canonical_glm_for_channel(raw_hb, channel_name, pipeline_label, file_label, ss_method, target_pair_names):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        channel_name=channel_name,
        model_type="canonical_glm",
        ss_method=ss_method,
    )
    task_regressor_names = get_task_regressor_names(design_matrix)
    if len(task_regressor_names) == 0:
        raise ValueError(f"No task regressors found for canonical GLM on {channel_name}")
    task_regressor_name = task_regressor_names[0]
    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model="ar1")
    glm_df = glm_result.to_dataframe()
    return extract_canonical_glm_metrics(glm_df, task_regressor_name, channel_name, pipeline_label, file_label, target_pair_names)


def run_fir_glm_for_channel(raw_hb, channel_name, pipeline_label, file_label, ss_method, target_pair_names):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        channel_name=channel_name,
        model_type="fir_glm",
        ss_method=ss_method,
    )
    fir_regressor_names = get_task_regressor_names(design_matrix)
    if len(fir_regressor_names) == 0:
        raise ValueError(f"No task regressors found for FIR GLM on {channel_name}")
    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model="ar1")
    glm_df = glm_result.to_dataframe()
    return extract_fir_glm_rows(glm_df, fir_regressor_names, channel_name, pipeline_label, file_label, target_pair_names)


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
# Subject-level probe
# -----------------------------------------------------------------------------


def build_subject_probe(subject_name_value):
    global subject_name
    global subject_dir
    global file_label

    subject_name = subject_name_value
    subject_dir = rs_data_dir / subject_name

    reference_file_label = "hrf_20"
    reference_file_path = get_file_path(subject_dir, reference_file_label)
    if not reference_file_path.exists():
        return None

    file_label = reference_file_label
    reference_raw_cw = mne.io.read_raw_snirf(reference_file_path, preload=True, verbose=False)

    target_pair_names = get_true_injected_pair_names(reference_file_path, reference_raw_cw)
    target_channel_names = get_channel_names_from_pairs(target_pair_names)

    cw_channel_table = build_cw_channel_table(reference_raw_cw, subject_name, reference_file_label)
    long_pair_names = sorted(cw_channel_table.loc[cw_channel_table["group"] == "LS", "pair_name"].unique())
    non_target_pair_names = [pair_name for pair_name in long_pair_names if pair_name not in target_pair_names]

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

    for file_label in file_labels:
        file_path = get_file_path(subject_dir, file_label)
        if not file_path.exists():
            continue

        raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
        if file_label == "no_hrf":
            raw_cw = copy_valid_annotations(raw_cw, get_annotation_source_file_path(subject_dir))

        for pipeline_label in pipeline_labels:
            pipeline_spec = get_pipeline_spec(pipeline_label)
            pipeline_result = preprocess_raw_to_hb(raw_cw, pipeline_label)
            raw_hb = pipeline_result["raw_hb"]

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

            epochs_hb, _, _ = make_epochs_from_raw_hb(raw_hb)
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

            if pipeline_spec["estimation_method"] == "canonical_glm":
                for channel_name in all_available_long_channel_names:
                    canonical_metrics = run_canonical_glm_for_channel(
                        raw_hb=raw_hb,
                        channel_name=channel_name,
                        pipeline_label=pipeline_label,
                        file_label=file_label,
                        ss_method=pipeline_spec["ss_method"],
                        target_pair_names=target_pair_names,
                    )
                    canonical_channel_rows.append(canonical_metrics)

            elif pipeline_spec["estimation_method"] == "block_average":
                block_average_channel_df = compute_block_average_channel_metrics(
                    epochs_hb=epochs_hb,
                    channel_names=all_available_long_channel_names,
                    pipeline_label=pipeline_label,
                    file_label=file_label,
                    target_pair_names=target_pair_names,
                )
                if len(block_average_channel_df) > 0:
                    block_average_channel_tables.append(block_average_channel_df)

            elif pipeline_spec["estimation_method"] == "fir_glm":
                for channel_name in all_available_long_channel_names:
                    fir_channel_df = run_fir_glm_for_channel(
                        raw_hb=raw_hb,
                        channel_name=channel_name,
                        pipeline_label=pipeline_label,
                        file_label=file_label,
                        ss_method=pipeline_spec["ss_method"],
                        target_pair_names=target_pair_names,
                    )
                    if len(fir_channel_df) > 0:
                        fir_channel_tables.append(fir_channel_df)

    quality_summary_df = pd.concat(quality_summary_rows, ignore_index=True) if len(quality_summary_rows) > 0 else pd.DataFrame()
    channel_availability_df = pd.DataFrame(channel_availability_rows) if len(channel_availability_rows) > 0 else pd.DataFrame()
    canonical_channel_df = pd.DataFrame(canonical_channel_rows) if len(canonical_channel_rows) > 0 else pd.DataFrame()
    block_average_channel_df = pd.concat(block_average_channel_tables, ignore_index=True) if len(block_average_channel_tables) > 0 else pd.DataFrame()
    fir_channel_df = pd.concat(fir_channel_tables, ignore_index=True) if len(fir_channel_tables) > 0 else pd.DataFrame()
    epoch_plot_df = pd.concat(epoch_plot_tables, ignore_index=True) if len(epoch_plot_tables) > 0 else pd.DataFrame()

    return {
        "truth_summary_row": truth_summary_row,
        "quality_summary_df": quality_summary_df,
        "channel_availability_df": channel_availability_df,
        "canonical_channel_df": canonical_channel_df,
        "block_average_channel_df": block_average_channel_df,
        "fir_channel_df": fir_channel_df,
        "epoch_plot_df": epoch_plot_df,
    }


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------


def save_epoch_trace_figure(epoch_plot_df):
    if len(epoch_plot_df) == 0:
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
    epoch_figure.update_layout(width=1400, height=850)
    epoch_figure.write_html(output_figures_dir / f"{output_prefix}_epoch_traces_targets_only.html")



def save_roi_score_figure(roi_score_df):
    if len(roi_score_df) == 0:
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
        hover_data=["n_channels", "roi_std_score"],
        title="ROI scores by pipeline, condition, and target status",
    )
    roi_figure.update_layout(width=1500, height=800)
    roi_figure.write_html(output_figures_dir / f"{output_prefix}_roi_scores_by_target_status.html")



def save_target_minus_non_target_figure(target_vs_nontarget_summary_df):
    if len(target_vs_nontarget_summary_df) == 0:
        return

    separation_figure = px.bar(
        target_vs_nontarget_summary_df.sort_values(["score_type", "chromophore", "file_label", "pipeline_label"]),
        x="pipeline_label",
        y="target_minus_non_target_score",
        color="file_label",
        barmode="group",
        facet_row="chromophore",
        facet_col="score_type",
        category_orders={
            "file_label": plot_condition_order,
            "score_type": plot_score_type_order,
            "chromophore": plot_chromophore_order,
        },
        hover_data=["mean_target_score", "mean_non_target_score"],
        title="Target minus non-target separation by pipeline",
    )
    separation_figure.update_layout(width=1500, height=800)
    separation_figure.write_html(output_figures_dir / f"{output_prefix}_target_minus_non_target_scores.html")



def save_channel_distribution_figure(channel_df, value_column, score_label, figure_suffix):
    if len(channel_df) == 0:
        return

    figure_df = channel_df.copy()
    figure_df = figure_df.loc[figure_df["file_label"].isin(plot_condition_order)].copy()

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
        hover_data=["channel_name", "pair_name"],
        title=f"Channel-level {score_label}: true targets vs true non-targets",
    )
    distribution_figure.update_layout(width=1500, height=850)
    distribution_figure.write_html(output_figures_dir / f"{output_prefix}_{figure_suffix}.html")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    truth_summary_rows = []
    all_quality_tables = []
    all_channel_availability_tables = []
    all_canonical_channel_tables = []
    all_block_average_channel_tables = []
    all_fir_channel_tables = []
    all_epoch_plot_tables = []

    for subject_name_value in subject_names_to_run:
        subject_result = build_subject_probe(subject_name_value)
        if subject_result is None:
            print(f"Skipping {subject_name_value}: missing reference file")
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

        print(f"Finished {subject_name_value}")

    truth_summary_df = pd.DataFrame(truth_summary_rows)
    quality_summary_df = pd.concat(all_quality_tables, ignore_index=True) if len(all_quality_tables) > 0 else pd.DataFrame()
    channel_availability_df = pd.concat(all_channel_availability_tables, ignore_index=True) if len(all_channel_availability_tables) > 0 else pd.DataFrame()
    canonical_channel_df = pd.concat(all_canonical_channel_tables, ignore_index=True) if len(all_canonical_channel_tables) > 0 else pd.DataFrame()
    block_average_channel_df = pd.concat(all_block_average_channel_tables, ignore_index=True) if len(all_block_average_channel_tables) > 0 else pd.DataFrame()
    fir_channel_df = pd.concat(all_fir_channel_tables, ignore_index=True) if len(all_fir_channel_tables) > 0 else pd.DataFrame()
    epoch_plot_df = pd.concat(all_epoch_plot_tables, ignore_index=True) if len(all_epoch_plot_tables) > 0 else pd.DataFrame()

    truth_summary_df.to_csv(output_tables_dir / f"{output_prefix}_truth_summary.csv", index=False)
    quality_summary_df.to_csv(output_tables_dir / f"{output_prefix}_quality_summary.csv", index=False)
    if len(channel_availability_df) > 0:
        channel_availability_df.to_csv(output_tables_dir / f"{output_prefix}_channel_availability_summary.csv", index=False)
    if len(canonical_channel_df) > 0:
        canonical_channel_df.to_csv(output_tables_dir / f"{output_prefix}_canonical_channel_metrics.csv", index=False)
    if len(block_average_channel_df) > 0:
        block_average_channel_df.to_csv(output_tables_dir / f"{output_prefix}_block_average_channel_metrics.csv", index=False)
    if len(fir_channel_df) > 0:
        fir_channel_df.to_csv(output_tables_dir / f"{output_prefix}_fir_channel_metrics.csv", index=False)

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

    roi_score_df = pd.concat(roi_summary_tables, ignore_index=True) if len(roi_summary_tables) > 0 else pd.DataFrame()
    if len(roi_score_df) > 0:
        roi_score_df.to_csv(output_tables_dir / f"{output_prefix}_roi_scores.csv", index=False)

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

    target_vs_nontarget_summary_df = pd.concat(separation_summary_tables, ignore_index=True) if len(separation_summary_tables) > 0 else pd.DataFrame()
    if len(target_vs_nontarget_summary_df) > 0:
        target_vs_nontarget_summary_df.to_csv(output_tables_dir / f"{output_prefix}_target_vs_nontarget_summary.csv", index=False)

    if len(canonical_channel_df) > 0 and "p_value" in canonical_channel_df.columns:
        canonical_null_df = canonical_channel_df.loc[canonical_channel_df["file_label"] == "no_hrf"].copy()
        if len(canonical_null_df) > 0:
            canonical_null_df["is_false_positive_p_lt_0_05"] = canonical_null_df["p_value"] < 0.05
            null_false_positive_summary_df = (
                canonical_null_df
                .groupby(["subject", "pipeline_label", "chromophore", "target_status"], as_index=False)
                .agg(
                    false_positive_rate_p_lt_0_05=("is_false_positive_p_lt_0_05", "mean"),
                    mean_abs_beta=("beta", lambda values: np.mean(np.abs(values))),
                    n_channels=("channel_name", "count"),
                )
            )
            null_false_positive_summary_df.to_csv(output_tables_dir / f"{output_prefix}_canonical_null_false_positive_summary.csv", index=False)

    if len(fir_channel_df) > 0:
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
        fir_shape_summary_df = pd.DataFrame(fir_shape_rows)
        fir_shape_summary_df.to_csv(output_tables_dir / f"{output_prefix}_fir_shape_summary.csv", index=False)

    save_epoch_trace_figure(epoch_plot_df)
    save_roi_score_figure(roi_score_df)
    save_target_minus_non_target_figure(target_vs_nontarget_summary_df)
    save_channel_distribution_figure(canonical_channel_df, "beta", "canonical beta", "canonical_channel_distributions")
    save_channel_distribution_figure(block_average_channel_df, "score", "block-average score", "block_average_channel_distributions")

    print(f"Saved {output_prefix} outputs.")


if __name__ == "__main__":
    main()
