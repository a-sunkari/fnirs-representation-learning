import pandas as pd
import numpy as np
import mne
import mne_nirs
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


root = Path.home() / "fnirs-representation-learning"
rs_data_dir = root / "snirf_dataset_2"
output_dir = root / "outputs"
output_tables_dir = output_dir / "tables"
output_figures_dir = output_dir / "figures"

output_tables_dir.mkdir(parents=True, exist_ok=True)
output_figures_dir.mkdir(parents=True, exist_ok=True)

subject_name = "Subj100"
subject_dir = rs_data_dir / subject_name

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

task_duration_s = 1.0
drift_high_pass_hz = 0.01

fir_delays_s = list(range(26))
ppf_value = 0.1

top_m_pairs = 5

file_labels = ["null", "hrf_20", "hrf_50", "hrf_100"]

pipeline_labels = [
    "Pref",
    "PnoSS",
    "PPcaSS",
    "PLooseQC",
    "PBlockAvg",
    "PFIR",
    "PnoMC",
]


def get_file_path(subject_dir, file_label):
    if file_label == "null":
        return subject_dir / "resting_clean.snirf"
    if file_label == "hrf_20":
        return subject_dir / "resting_hrf_20.snirf"
    if file_label == "hrf_50":
        return subject_dir / "resting_hrf_50.snirf"
    if file_label == "hrf_100":
        return subject_dir / "resting_hrf_100.snirf"
    raise ValueError(f"Unknown file_label: {file_label}")


def get_annotation_source_file_path(subject_dir):
    return subject_dir / "resting_hrf_100.snirf"


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


def get_cw_channel_indices(raw_snirf):
    picks_fnirs = mne.pick_types(raw_snirf.info, fnirs=True)
    channel_types = np.array(raw_snirf.get_channel_types())
    picks_cw = picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]
    return picks_cw


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


def build_short_regressor_signal(raw_hb, chromophore, ss_method):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    short_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "SS") &
        (hb_channel_table["chromophore"] == chromophore),
        "channel_name",
    ].tolist()

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


def preprocess_raw_to_hb(raw_cw, pipeline_label):
    if pipeline_label in ["Pref", "PnoSS", "PPcaSS", "PBlockAvg", "PFIR"]:
        pruning_style = "strict_combined"
        do_tddr = True
    elif pipeline_label == "PLooseQC":
        pruning_style = "loose_sci"
        do_tddr = True
    elif pipeline_label == "PnoMC":
        pruning_style = "strict_combined"
        do_tddr = False
    else:
        raise ValueError(f"Unknown pipeline_label: {pipeline_label}")

    channel_quality_table, pair_quality_table = build_quality_table(raw_cw, subject_name, file_label)
    bad_pair_names = get_bad_pair_names(pair_quality_table, pruning_style)

    raw_od = optical_density(raw_cw.copy())

    if do_tddr:
        processed_od = tddr(raw_od.copy())
    else:
        processed_od = raw_od.copy()

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


def compute_channel_effect_table(epochs_hb, raw_hb, subject_name, file_label, pipeline_label):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    ls_hbo_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "LS") &
        (hb_channel_table["chromophore"] == "hbo"),
        "channel_name",
    ].tolist()

    baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
    response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])

    channel_effect_rows = []

    for channel_name in ls_hbo_channel_names:
        if channel_name in raw_hb.info["bads"]:
            continue

        single_channel_epochs = epochs_hb.copy().pick([channel_name])
        single_channel_data = single_channel_epochs.get_data()[:, 0, :]

        baseline_values = single_channel_data[:, baseline_time_mask].mean(axis=1)
        response_values = single_channel_data[:, response_time_mask].mean(axis=1)
        effect_values = response_values - baseline_values

        channel_effect_rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "pipeline_label": pipeline_label,
            "channel_name": channel_name,
            "pair_name": channel_name.split(" ")[0],
            "mean_baseline": baseline_values.mean(),
            "mean_response": response_values.mean(),
            "mean_effect_size": effect_values.mean(),
            "std_effect_size": effect_values.std(),
        })

    channel_effect_df = pd.DataFrame(channel_effect_rows)
    channel_effect_df = channel_effect_df.sort_values("mean_effect_size", ascending=False).reset_index(drop=True)
    return channel_effect_df


def build_mean_epoch_plot_table(epochs_hb, selected_channel_names, pipeline_label):
    available_channel_names = [channel_name for channel_name in selected_channel_names if channel_name in epochs_hb.ch_names]

    hbo_channel_names = [channel_name for channel_name in available_channel_names if channel_name.endswith("hbo")]
    hbr_channel_names = [channel_name for channel_name in available_channel_names if channel_name.endswith("hbr")]

    plot_rows = []

    if len(hbo_channel_names) > 0:
        hbo_data = epochs_hb.copy().pick(hbo_channel_names).get_data()
        mean_hbo_time_course = hbo_data.mean(axis=(0, 1))
        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hb.times,
            "signal": mean_hbo_time_course,
            "chromophore": "HbO",
            "pipeline_label": pipeline_label,
        }))

    if len(hbr_channel_names) > 0:
        hbr_data = epochs_hb.copy().pick(hbr_channel_names).get_data()
        mean_hbr_time_course = hbr_data.mean(axis=(0, 1))
        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hb.times,
            "signal": mean_hbr_time_course,
            "chromophore": "HbR",
            "pipeline_label": pipeline_label,
        }))

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

    design_matrix = make_first_level_design_matrix(
        raw_hb,
        stim_dur=task_duration_s,
        drift_model="cosine",
        high_pass=drift_high_pass_hz,
        hrf_model=hrf_model,
        fir_delays=fir_delays,
    )

    if ss_method != "none":
        chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"
        short_regressor_signal = build_short_regressor_signal(raw_hb, chromophore, ss_method)
        if short_regressor_signal is not None:
            design_matrix[f"ss_{chromophore}"] = short_regressor_signal

    return design_matrix


def extract_canonical_glm_metrics(glm_df, task_regressor_name, channel_name, pipeline_label, file_label):
    glm_df = standardize_glm_dataframe(glm_df)

    condition_column_name = find_first_matching_column(glm_df.columns, ["condition", "regressor", "variable", "name"])
    beta_column_name = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])
    t_column_name = find_first_matching_column(glm_df.columns, ["t", "t_value", "tstat", "t_stat"])
    p_column_name = find_first_matching_column(glm_df.columns, ["p_value", "pvalue", "p"])

    canonical_rows = glm_df.loc[glm_df[condition_column_name].astype(str) == str(task_regressor_name)]
    if len(canonical_rows) == 0:
        canonical_rows = glm_df.loc[glm_df[condition_column_name].astype(str).str.contains(str(task_regressor_name), regex=False)]

    canonical_row = canonical_rows.iloc[0]
    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    result_row = {
        "subject": subject_name,
        "file_label": file_label,
        "pipeline_label": pipeline_label,
        "channel_name": channel_name,
        "pair_name": channel_name.split(" ")[0],
        "chromophore": chromophore,
        "task_regressor": str(task_regressor_name),
        "beta": float(canonical_row[beta_column_name]),
        "t_value": float(canonical_row[t_column_name]),
    }

    if p_column_name is not None:
        result_row["p_value"] = float(canonical_row[p_column_name])

    return result_row


def extract_fir_glm_rows(glm_df, fir_regressor_names, channel_name, pipeline_label, file_label):
    glm_df = standardize_glm_dataframe(glm_df)

    condition_column_name = find_first_matching_column(glm_df.columns, ["condition", "regressor", "variable", "name"])
    beta_column_name = find_first_matching_column(glm_df.columns, ["theta", "beta", "coef", "estimate", "effect"])

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
            "pipeline_label": pipeline_label,
            "channel_name": channel_name,
            "pair_name": channel_name.split(" ")[0],
            "chromophore": chromophore,
            "fir_regressor": str(fir_regressor_name),
            "delay_s": delay_value_s,
            "beta": float(fir_row[beta_column_name]),
        })

    return pd.DataFrame(fir_rows)


def run_canonical_glm_for_channel(raw_hb, channel_name, pipeline_label, file_label, ss_method):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        channel_name=channel_name,
        model_type="canonical_glm",
        ss_method=ss_method,
    )

    task_regressor_names = get_task_regressor_names(design_matrix)
    task_regressor_name = task_regressor_names[0]

    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model="ar1")
    glm_df = glm_result.to_dataframe()

    canonical_metrics = extract_canonical_glm_metrics(
        glm_df=glm_df,
        task_regressor_name=task_regressor_name,
        channel_name=channel_name,
        pipeline_label=pipeline_label,
        file_label=file_label,
    )
    return canonical_metrics


def run_fir_glm_for_channel(raw_hb, channel_name, pipeline_label, file_label, ss_method):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        channel_name=channel_name,
        model_type="fir_glm",
        ss_method=ss_method,
    )

    fir_regressor_names = get_task_regressor_names(design_matrix)
    single_channel_raw = raw_hb.copy().pick([channel_name])
    glm_result = run_glm(single_channel_raw, design_matrix, noise_model="ar1")
    glm_df = glm_result.to_dataframe()

    fir_df = extract_fir_glm_rows(
        glm_df=glm_df,
        fir_regressor_names=fir_regressor_names,
        channel_name=channel_name,
        pipeline_label=pipeline_label,
        file_label=file_label,
    )
    return fir_df


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


def compute_block_average_metrics(epochs_hb, selected_channel_names, pipeline_label, file_label):
    available_hbo_channel_names = [channel_name for channel_name in selected_channel_names if channel_name.endswith("hbo") and channel_name in epochs_hb.ch_names]
    available_hbr_channel_names = [channel_name for channel_name in selected_channel_names if channel_name.endswith("hbr") and channel_name in epochs_hb.ch_names]

    baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
    response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])

    metrics_row = {
        "subject": subject_name,
        "file_label": file_label,
        "pipeline_label": pipeline_label,
    }

    if len(available_hbo_channel_names) > 0:
        hbo_data = epochs_hb.copy().pick(available_hbo_channel_names).get_data()
        hbo_baseline = hbo_data[:, :, baseline_time_mask].mean()
        hbo_response = hbo_data[:, :, response_time_mask].mean()
        metrics_row["hbo_score"] = float(hbo_response - hbo_baseline)
    else:
        metrics_row["hbo_score"] = np.nan

    if len(available_hbr_channel_names) > 0:
        hbr_data = epochs_hb.copy().pick(available_hbr_channel_names).get_data()
        hbr_baseline = hbr_data[:, :, baseline_time_mask].mean()
        hbr_response = hbr_data[:, :, response_time_mask].mean()
        metrics_row["hbr_score"] = float(hbr_response - hbr_baseline)
    else:
        metrics_row["hbr_score"] = np.nan

    return metrics_row


def get_pipeline_spec(pipeline_label):
    if pipeline_label == "Pref":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "average"}
    if pipeline_label == "PnoSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "none"}
    if pipeline_label == "PPcaSS":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "pca"}
    if pipeline_label == "PLooseQC":
        return {"pruning_style": "loose_sci", "do_tddr": True, "estimation_method": "canonical_glm", "ss_method": "average"}
    if pipeline_label == "PBlockAvg":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "block_average", "ss_method": "average"}
    if pipeline_label == "PFIR":
        return {"pruning_style": "strict_combined", "do_tddr": True, "estimation_method": "fir_glm", "ss_method": "average"}
    if pipeline_label == "PnoMC":
        return {"pruning_style": "strict_combined", "do_tddr": False, "estimation_method": "canonical_glm", "ss_method": "average"}
    raise ValueError(f"Unknown pipeline_label: {pipeline_label}")


def get_amplitude_value(file_label):
    if file_label == "null":
        return 0
    if file_label == "hrf_20":
        return 20
    if file_label == "hrf_50":
        return 50
    if file_label == "hrf_100":
        return 100
    return np.nan


quality_summary_rows = []
roi_score_rows = []
canonical_channel_rows = []
fir_channel_rows = []
epoch_plot_tables = []


reference_file_label = "hrf_100"
reference_file_path = get_file_path(subject_dir, reference_file_label)
reference_raw_cw = mne.io.read_raw_snirf(reference_file_path, preload=True, verbose=False)
reference_pipeline_result = preprocess_raw_to_hb(reference_raw_cw, "Pref")
reference_raw_hb = reference_pipeline_result["raw_hb"]
reference_epochs_hb, reference_events, reference_event_id = make_epochs_from_raw_hb(reference_raw_hb)
reference_channel_effect_df = compute_channel_effect_table(
    epochs_hb=reference_epochs_hb,
    raw_hb=reference_raw_hb,
    subject_name=subject_name,
    file_label=reference_file_label,
    pipeline_label="Pref",
)

top_hbo_channel_names = reference_channel_effect_df.head(top_m_pairs)["channel_name"].tolist()
target_pair_names = [channel_name.replace(" hbo", "") for channel_name in top_hbo_channel_names]
top_hbr_channel_names = [pair_name + " hbr" for pair_name in target_pair_names]
selected_channel_names = top_hbo_channel_names + top_hbr_channel_names

print("Target pair names:")
print(target_pair_names)

for file_label in file_labels:
    file_path = get_file_path(subject_dir, file_label)
    raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)

    if file_label == "null":
        annotation_source_file_path = get_annotation_source_file_path(subject_dir)
        raw_cw = copy_valid_annotations(raw_cw, annotation_source_file_path)

    for pipeline_label in pipeline_labels:
        pipeline_spec = get_pipeline_spec(pipeline_label)
        pipeline_result = preprocess_raw_to_hb(raw_cw, pipeline_label)
        raw_hb = pipeline_result["raw_hb"]

        pair_quality_table = pipeline_result["pair_quality_table"].copy()
        pair_quality_table["pipeline_label"] = pipeline_label
        pair_quality_table["pruning_style"] = pipeline_spec["pruning_style"]
        pair_quality_table["do_tddr"] = pipeline_spec["do_tddr"]
        quality_summary_rows.append(pair_quality_table)

        epochs_hb, events, event_id = make_epochs_from_raw_hb(raw_hb)

        epoch_plot_table = build_mean_epoch_plot_table(
            epochs_hb=epochs_hb,
            selected_channel_names=selected_channel_names,
            pipeline_label=f"{pipeline_label}_{file_label}",
        )
        epoch_plot_tables.append(epoch_plot_table)

        if pipeline_spec["estimation_method"] == "block_average":
            block_average_row = compute_block_average_metrics(
                epochs_hb=epochs_hb,
                selected_channel_names=selected_channel_names,
                pipeline_label=pipeline_label,
                file_label=file_label,
            )
            block_average_row["amplitude_value"] = get_amplitude_value(file_label)
            block_average_row["estimation_method"] = pipeline_spec["estimation_method"]
            block_average_row["ss_method"] = pipeline_spec["ss_method"]
            roi_score_rows.append(block_average_row)
            continue

        channel_rows_for_pipeline_file = []

        for channel_name in selected_channel_names:
            if channel_name not in raw_hb.ch_names:
                continue
            if channel_name in raw_hb.info["bads"]:
                continue

            if pipeline_spec["estimation_method"] == "canonical_glm":
                canonical_metrics = run_canonical_glm_for_channel(
                    raw_hb=raw_hb,
                    channel_name=channel_name,
                    pipeline_label=pipeline_label,
                    file_label=file_label,
                    ss_method=pipeline_spec["ss_method"],
                )
                canonical_channel_rows.append(canonical_metrics)
                channel_rows_for_pipeline_file.append(canonical_metrics)

            elif pipeline_spec["estimation_method"] == "fir_glm":
                fir_df = run_fir_glm_for_channel(
                    raw_hb=raw_hb,
                    channel_name=channel_name,
                    pipeline_label=pipeline_label,
                    file_label=file_label,
                    ss_method=pipeline_spec["ss_method"],
                )
                if len(fir_df) > 0:
                    fir_channel_rows.append(fir_df)

        if pipeline_spec["estimation_method"] == "canonical_glm" and len(channel_rows_for_pipeline_file) > 0:
            channel_df = pd.DataFrame(channel_rows_for_pipeline_file)
            roi_row = {
                "subject": subject_name,
                "file_label": file_label,
                "pipeline_label": pipeline_label,
                "amplitude_value": get_amplitude_value(file_label),
                "estimation_method": pipeline_spec["estimation_method"],
                "ss_method": pipeline_spec["ss_method"],
                "hbo_score": channel_df.loc[channel_df["chromophore"] == "hbo", "beta"].mean(),
                "hbr_score": channel_df.loc[channel_df["chromophore"] == "hbr", "beta"].mean(),
                "hbo_mean_t": channel_df.loc[channel_df["chromophore"] == "hbo", "t_value"].mean(),
                "hbr_mean_t": channel_df.loc[channel_df["chromophore"] == "hbr", "t_value"].mean(),
            }
            roi_score_rows.append(roi_row)


quality_summary_df = pd.concat(quality_summary_rows, ignore_index=True)
roi_score_df = pd.DataFrame(roi_score_rows)

if len(canonical_channel_rows) > 0:
    canonical_channel_df = pd.DataFrame(canonical_channel_rows)
else:
    canonical_channel_df = pd.DataFrame()

if len(fir_channel_rows) > 0:
    fir_channel_df = pd.concat(fir_channel_rows, ignore_index=True)
else:
    fir_channel_df = pd.DataFrame()

pipeline_epoch_plot_df = pd.concat(epoch_plot_tables, ignore_index=True)

if len(fir_channel_df) > 0:
    fir_roi_timecourse_df = (
        fir_channel_df
        .groupby(["subject", "file_label", "pipeline_label", "chromophore", "delay_s"], as_index=False)
        .agg(roi_mean_beta=("beta", "mean"))
    )

    fir_shape_rows = []

    for file_label in sorted(fir_roi_timecourse_df["file_label"].unique()):
        for chromophore in ["hbo", "hbr"]:
            chromophore_df = fir_roi_timecourse_df.loc[
                (fir_roi_timecourse_df["file_label"] == file_label) &
                (fir_roi_timecourse_df["chromophore"] == chromophore)
            ].sort_values("delay_s")

            delays_s = chromophore_df["delay_s"].to_numpy()
            beta_values = chromophore_df["roi_mean_beta"].to_numpy()

            if len(beta_values) == 0:
                continue

            if chromophore == "hbo":
                peak_index = int(np.argmax(beta_values))
            else:
                peak_index = int(np.argmin(beta_values))

            peak_delay_s = float(delays_s[peak_index])
            peak_beta = float(beta_values[peak_index])
            fwhm_s = compute_fwhm_from_fir(delays_s, beta_values, chromophore)

            fir_shape_rows.append({
                "subject": subject_name,
                "file_label": file_label,
                "pipeline_label": "PFIR",
                "amplitude_value": get_amplitude_value(file_label),
                "chromophore": chromophore,
                "peak_delay_s": peak_delay_s,
                "peak_beta": peak_beta,
                "fwhm_s": fwhm_s,
            })

    fir_shape_summary_df = pd.DataFrame(fir_shape_rows)
else:
    fir_roi_timecourse_df = pd.DataFrame()
    fir_shape_summary_df = pd.DataFrame()


quality_summary_path = output_tables_dir / f"{subject_name.lower()}_fresh_aligned_quality_summary.csv"
roi_score_path = output_tables_dir / f"{subject_name.lower()}_fresh_aligned_roi_scores.csv"
canonical_channel_path = output_tables_dir / f"{subject_name.lower()}_fresh_aligned_canonical_channel_metrics.csv"
fir_channel_path = output_tables_dir / f"{subject_name.lower()}_fresh_aligned_fir_channel_metrics.csv"
fir_shape_path = output_tables_dir / f"{subject_name.lower()}_fresh_aligned_fir_shape_summary.csv"

quality_summary_df.to_csv(quality_summary_path, index=False)
roi_score_df.to_csv(roi_score_path, index=False)
reference_channel_effect_df.to_csv(output_tables_dir / f"{subject_name.lower()}_fresh_aligned_reference_channel_effects.csv", index=False)

if len(canonical_channel_df) > 0:
    canonical_channel_df.to_csv(canonical_channel_path, index=False)

if len(fir_channel_df) > 0:
    fir_channel_df.to_csv(fir_channel_path, index=False)

if len(fir_shape_summary_df) > 0:
    fir_shape_summary_df.to_csv(fir_shape_path, index=False)


quality_figure = px.bar(
    quality_summary_df.groupby(["file_label", "pipeline_label"], as_index=False).agg(
        n_bad_pairs=("pair_name", lambda values: int(((quality_summary_df.loc[values.index, "sci_min"] < strict_sci_threshold)).sum())),
        mean_sci=("sci_mean", "mean"),
        mean_snr=("snr_mean", "mean"),
    ),
    x="pipeline_label",
    y="mean_sci",
    color="file_label",
    barmode="group",
    title=f"{subject_name}: mean SCI across FRESH-aligned pipelines",
)
quality_figure.update_layout(width=1100, height=500)
quality_figure.write_html(output_figures_dir / f"{subject_name.lower()}_fresh_aligned_quality_summary.html")

roi_score_figure = px.line(
    roi_score_df.sort_values("amplitude_value"),
    x="amplitude_value",
    y="hbo_score",
    color="pipeline_label",
    markers=True,
    title=f"{subject_name}: HbO ROI score across FRESH-aligned pipelines",
)
roi_score_figure.update_layout(width=1100, height=500)
roi_score_figure.write_html(output_figures_dir / f"{subject_name.lower()}_fresh_aligned_hbo_scores.html")

pipeline_epoch_figure = px.line(
    pipeline_epoch_plot_df,
    x="time_s",
    y="signal",
    color="pipeline_label",
    facet_row="chromophore",
    title=f"{subject_name}: event-locked Hb traces across FRESH-aligned pipelines and file labels",
)
pipeline_epoch_figure.update_layout(width=1200, height=800)
pipeline_epoch_figure.write_html(output_figures_dir / f"{subject_name.lower()}_fresh_aligned_epoch_traces.html")

if len(fir_roi_timecourse_df) > 0:
    fir_figure = px.line(
        fir_roi_timecourse_df,
        x="delay_s",
        y="roi_mean_beta",
        color="file_label",
        facet_row="chromophore",
        title=f"{subject_name}: PFIR ROI time course across amplitudes",
    )
    fir_figure.update_layout(width=1100, height=650)
    fir_figure.write_html(output_figures_dir / f"{subject_name.lower()}_fresh_aligned_fir_timecourse.html")

print("Saved FRESH-aligned Subj100 probe outputs.")
print(f"ROI score table: {roi_score_path}")
print(f"Quality summary table: {quality_summary_path}")
