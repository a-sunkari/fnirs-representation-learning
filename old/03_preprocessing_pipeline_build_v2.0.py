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

file_label = "hrf_50"
file_name = "resting_hrf_50.snirf"
file_path = subject_dir / file_name

short_separation_threshold_m = 0.015
long_separation_threshold_m = 0.025
sci_threshold = 0.50

filter_low_hz = 0.01
filter_high_hz = 0.20

epoch_tmin = -5.0
epoch_tmax = 30.0
baseline_window = (-5.0, 0.0)
response_window = (4.0, 8.0)

task_duration_s = 1.0
drift_high_pass_hz = 0.01

top_m_pairs = 5
fir_delays_s = list(range(26))

ppf_value = 0.1

pipeline_labels = ["Pref", "PnoSS", "PnoMC", "PFIR"]

file_name_map = {
    "null": "resting_clean.snirf",
    "hrf_20": "resting_hrf_20.snirf",
    "hrf_50": "resting_hrf_50.snirf",
    "hrf_100": "resting_hrf_100.snirf",
}

annotation_source_file_label = "hrf_50"
amplitude_value_map = {
    "null": 0.0,
    "hrf_20": 20.0,
    "hrf_50": 50.0,
    "hrf_100": 100.0,
}


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


def apply_bad_pairs_to_hb(raw_hb, bad_pair_names):
    raw_hb = raw_hb.copy()
    hb_bad_channel_names = []

    for channel_name in raw_hb.ch_names:
        pair_name = channel_name.split(" ")[0]
        if pair_name in bad_pair_names:
            hb_bad_channel_names.append(channel_name)

    raw_hb.info["bads"] = hb_bad_channel_names
    return raw_hb


def compute_channel_midpoint(raw_hb, channel_name):
    channel_index = raw_hb.ch_names.index(channel_name)
    channel_loc = raw_hb.info["chs"][channel_index]["loc"]

    source_xyz = channel_loc[3:6]
    detector_xyz = channel_loc[6:9]
    midpoint_xyz = (source_xyz + detector_xyz) / 2.0

    return midpoint_xyz


def find_nearest_short_channel(raw_hb, long_channel_name):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    long_channel_row = hb_channel_table.loc[
        hb_channel_table["channel_name"] == long_channel_name
    ].iloc[0]

    long_chromophore = long_channel_row["chromophore"]

    short_channel_candidates = hb_channel_table.loc[
        (hb_channel_table["group"] == "SS") &
        (hb_channel_table["chromophore"] == long_chromophore),
        "channel_name",
    ].tolist()

    if len(short_channel_candidates) == 0:
        return None

    long_midpoint = compute_channel_midpoint(raw_hb, long_channel_name)

    nearest_channel_name = None
    nearest_distance = np.inf

    for short_channel_name in short_channel_candidates:
        short_midpoint = compute_channel_midpoint(raw_hb, short_channel_name)
        midpoint_distance = np.linalg.norm(long_midpoint - short_midpoint)

        if midpoint_distance < nearest_distance:
            nearest_distance = midpoint_distance
            nearest_channel_name = short_channel_name

    return nearest_channel_name


def build_average_short_regressor(raw_hb, chromophore):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    short_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "SS") &
        (hb_channel_table["chromophore"] == chromophore),
        "channel_name",
    ].tolist()

    if len(short_channel_names) == 0:
        return None

    short_channel_data = raw_hb.copy().pick(short_channel_names).get_data()
    mean_short_signal = short_channel_data.mean(axis=0)

    return mean_short_signal


def add_reference_annotations(raw_target_cw, raw_annotation_source_cw):
    raw_target_cw = raw_target_cw.copy()
    raw_target_cw.set_annotations(raw_annotation_source_cw.annotations.copy())
    return raw_target_cw


def get_task_duration_from_annotations(raw_hb):
    annotation_durations = np.array(raw_hb.annotations.duration, dtype=float)
    positive_durations = annotation_durations[annotation_durations > 0]

    if len(positive_durations) == 0:
        return task_duration_s

    return float(np.median(positive_durations))


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
    if len(channel_effect_df) == 0:
        return channel_effect_df

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

    if len(plot_rows) == 0:
        return pd.DataFrame(columns=["time_s", "signal", "chromophore", "pipeline_label"])

    mean_epoch_plot_table = pd.concat(plot_rows, ignore_index=True)
    return mean_epoch_plot_table


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


def build_design_matrix_for_channel(raw_hb, long_channel_name, hrf_model, include_short_regressor):
    current_task_duration_s = get_task_duration_from_annotations(raw_hb)

    design_matrix = make_first_level_design_matrix(
        raw_hb,
        stim_dur=current_task_duration_s,
        drift_model="cosine",
        high_pass=drift_high_pass_hz,
        hrf_model=hrf_model,
        fir_delays=fir_delays_s if hrf_model == "fir" else None,
    )

    if include_short_regressor:
        chromophore = "hbo" if long_channel_name.endswith("hbo") else "hbr"
        nearest_short_channel_name = find_nearest_short_channel(raw_hb, long_channel_name)

        if nearest_short_channel_name is not None:
            nearest_short_signal = raw_hb.copy().pick([nearest_short_channel_name]).get_data()[0]
            design_matrix[f"ss_{chromophore}"] = nearest_short_signal
        else:
            average_short_signal = build_average_short_regressor(raw_hb, chromophore)
            if average_short_signal is not None:
                design_matrix[f"ss_{chromophore}"] = average_short_signal

    return design_matrix


def extract_canonical_glm_metrics(glm_df, task_regressor_name, channel_name, pipeline_label):
    glm_df = standardize_glm_dataframe(glm_df)

    condition_column_name = find_first_matching_column(
        glm_df.columns,
        ["condition", "regressor", "variable", "name"]
    )

    beta_column_name = find_first_matching_column(
        glm_df.columns,
        ["theta", "beta", "coef", "estimate", "effect"]
    )

    t_column_name = find_first_matching_column(
        glm_df.columns,
        ["t", "t_value", "tstat", "t_stat"]
    )

    p_column_name = find_first_matching_column(
        glm_df.columns,
        ["p_value", "pvalue", "p"]
    )

    if condition_column_name is None or beta_column_name is None or t_column_name is None:
        raise ValueError(f"Could not find expected GLM columns. Found columns: {glm_df.columns.tolist()}")

    canonical_rows = glm_df.loc[glm_df[condition_column_name].astype(str) == str(task_regressor_name)]

    if len(canonical_rows) == 0:
        canonical_rows = glm_df.loc[
            glm_df[condition_column_name].astype(str).str.contains(str(task_regressor_name), regex=False)
        ]

    if len(canonical_rows) == 0:
        raise ValueError(f"Could not find canonical task regressor row for {task_regressor_name}")

    canonical_row = canonical_rows.iloc[0]

    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    result_row = {
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


def extract_fir_glm_rows(glm_df, fir_regressor_names, channel_name, pipeline_label):
    glm_df = standardize_glm_dataframe(glm_df)

    condition_column_name = find_first_matching_column(
        glm_df.columns,
        ["condition", "regressor", "variable", "name"]
    )

    beta_column_name = find_first_matching_column(
        glm_df.columns,
        ["theta", "beta", "coef", "estimate", "effect"]
    )

    if condition_column_name is None or beta_column_name is None:
        raise ValueError(f"Could not find expected FIR GLM columns. Found columns: {glm_df.columns.tolist()}")

    chromophore = "hbo" if channel_name.endswith("hbo") else "hbr"

    fir_rows = []

    for fir_regressor_name in fir_regressor_names:
        matching_rows = glm_df.loc[
            glm_df[condition_column_name].astype(str) == str(fir_regressor_name)
        ]

        if len(matching_rows) == 0:
            matching_rows = glm_df.loc[
                glm_df[condition_column_name].astype(str).str.contains(str(fir_regressor_name), regex=False)
            ]

        if len(matching_rows) == 0:
            continue

        fir_row = matching_rows.iloc[0]
        delay_string = str(fir_regressor_name).split("_")[-1]

        try:
            delay_value_s = float(delay_string)
        except ValueError:
            continue

        fir_rows.append({
            "pipeline_label": pipeline_label,
            "channel_name": channel_name,
            "pair_name": channel_name.split(" ")[0],
            "chromophore": chromophore,
            "fir_regressor": str(fir_regressor_name),
            "delay_s": delay_value_s,
            "beta": float(fir_row[beta_column_name]),
        })

    fir_df = pd.DataFrame(fir_rows)
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

    width_s = delays_s[above_half_mask].max() - delays_s[above_half_mask].min()
    return width_s


def preprocess_raw_to_hb(raw_cw, pipeline_label):
    raw_od = optical_density(raw_cw.copy())

    sci_values = scalp_coupling_index(raw_od)
    cw_channel_table = build_cw_channel_table(raw_cw, subject_name, file_label)

    sci_table = pd.DataFrame({
        "channel_name": np.array(raw_cw.ch_names)[get_cw_channel_indices(raw_cw)],
        "pair_name": cw_channel_table["pair_name"].values,
        "sci": sci_values,
    })

    bad_pair_names = sci_table.loc[sci_table["sci"] < sci_threshold, "pair_name"].unique().tolist()

    if pipeline_label == "PnoMC":
        processed_od = raw_od.copy()
    else:
        processed_od = tddr(raw_od.copy())

    raw_hb = beer_lambert_law(processed_od, ppf=ppf_value)
    raw_hb = raw_hb.copy().filter(filter_low_hz, filter_high_hz, verbose=False)
    raw_hb = apply_bad_pairs_to_hb(raw_hb, bad_pair_names)

    return {
        "pipeline_label": pipeline_label,
        "raw_od": processed_od,
        "raw_hb": raw_hb,
        "sci_table": sci_table,
        "bad_pair_names": bad_pair_names,
    }


def run_canonical_glm_for_channel(raw_hb, channel_name, pipeline_label, include_short_regressor):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        long_channel_name=channel_name,
        hrf_model="glover",
        include_short_regressor=include_short_regressor,
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
    )

    return canonical_metrics


def run_fir_glm_for_channel(raw_hb, channel_name, pipeline_label, include_short_regressor):
    design_matrix = build_design_matrix_for_channel(
        raw_hb=raw_hb,
        long_channel_name=channel_name,
        hrf_model="fir",
        include_short_regressor=include_short_regressor,
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
    )

    return fir_df


def build_subject_reference_channels(subject_name):
    global file_label
    file_label = "hrf_100"

    subject_dir = rs_data_dir / subject_name
    file_path = subject_dir / file_name_map[file_label]

    if not file_path.exists():
        return pd.DataFrame(), [], []

    raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
    reference_pipeline_result = preprocess_raw_to_hb(raw_cw, pipeline_label="Pref")
    reference_raw_hb = reference_pipeline_result["raw_hb"]
    reference_epochs_hb, _, _ = make_epochs_from_raw_hb(reference_raw_hb)

    reference_channel_effect_df = compute_channel_effect_table(
        epochs_hb=reference_epochs_hb,
        raw_hb=reference_raw_hb,
        subject_name=subject_name,
        file_label=file_label,
        pipeline_label="Pref",
    )

    if len(reference_channel_effect_df) == 0:
        return reference_channel_effect_df, [], []

    top_hbo_channel_names = reference_channel_effect_df.head(top_m_pairs)["channel_name"].tolist()
    target_pair_names = [channel_name.replace(" hbo", "") for channel_name in top_hbo_channel_names]
    top_hbr_channel_names = [pair_name + " hbr" for pair_name in target_pair_names]

    return reference_channel_effect_df, top_hbo_channel_names, top_hbr_channel_names


def run_subject_file_pipeline_set(subject_name, file_label, top_hbo_channel_names, top_hbr_channel_names):
    subject_dir = rs_data_dir / subject_name
    file_name = file_name_map[file_label]
    file_path = subject_dir / file_name

    if not file_path.exists():
        return None

    raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)

    if file_label == "null":
        annotation_source_file_path = subject_dir / file_name_map[annotation_source_file_label]
        if not annotation_source_file_path.exists():
            return None

        raw_annotation_source_cw = mne.io.read_raw_snirf(annotation_source_file_path, preload=False, verbose=False)
        raw_cw = add_reference_annotations(raw_cw, raw_annotation_source_cw)

    pipeline_results = []

    for pipeline_label in pipeline_labels:
        pipeline_result = preprocess_raw_to_hb(raw_cw, pipeline_label)
        pipeline_results.append(pipeline_result)

    selected_channel_names = top_hbo_channel_names + top_hbr_channel_names

    epoch_rows = []
    canonical_rows = []
    fir_rows = []
    pipeline_plot_tables = []

    for pipeline_result in pipeline_results:
        pipeline_label = pipeline_result["pipeline_label"]
        raw_hb = pipeline_result["raw_hb"]

        epochs_hb, events, event_id = make_epochs_from_raw_hb(raw_hb)

        available_hbo_channel_names = [channel_name for channel_name in top_hbo_channel_names if channel_name in epochs_hb.ch_names and channel_name not in raw_hb.info["bads"]]
        available_hbr_channel_names = [channel_name for channel_name in top_hbr_channel_names if channel_name in epochs_hb.ch_names and channel_name not in raw_hb.info["bads"]]

        if len(available_hbo_channel_names) > 0:
            hbo_data = epochs_hb.copy().pick(available_hbo_channel_names).get_data()
            baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
            response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])
            hbo_baseline = hbo_data[:, :, baseline_time_mask].mean()
            hbo_response = hbo_data[:, :, response_time_mask].mean()
            hbo_mean_effect = hbo_response - hbo_baseline
        else:
            hbo_mean_effect = np.nan

        if len(available_hbr_channel_names) > 0:
            hbr_data = epochs_hb.copy().pick(available_hbr_channel_names).get_data()
            baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
            response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])
            hbr_baseline = hbr_data[:, :, baseline_time_mask].mean()
            hbr_response = hbr_data[:, :, response_time_mask].mean()
            hbr_mean_effect = hbr_response - hbr_baseline
        else:
            hbr_mean_effect = np.nan

        epoch_rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "amplitude_value": amplitude_value_map[file_label],
            "pipeline_label": pipeline_label,
            "hbo_mean_effect": hbo_mean_effect,
            "hbr_mean_effect": hbr_mean_effect,
            "n_events": len(epochs_hb),
            "n_hbo_channels": len(available_hbo_channel_names),
            "n_hbr_channels": len(available_hbr_channel_names),
            "n_pairs_below_threshold": len(pipeline_result["bad_pair_names"]),
        })

        mean_epoch_plot_table = build_mean_epoch_plot_table(
            epochs_hb=epochs_hb,
            selected_channel_names=selected_channel_names,
            pipeline_label=pipeline_label,
        )
        if len(mean_epoch_plot_table) > 0:
            mean_epoch_plot_table["subject"] = subject_name
            mean_epoch_plot_table["file_label"] = file_label
            mean_epoch_plot_table["amplitude_value"] = amplitude_value_map[file_label]
            pipeline_plot_tables.append(mean_epoch_plot_table)

        for channel_name in selected_channel_names:
            if channel_name not in raw_hb.ch_names:
                continue
            if channel_name in raw_hb.info["bads"]:
                continue

            include_short_regressor = pipeline_label != "PnoSS"

            canonical_metrics = run_canonical_glm_for_channel(
                raw_hb=raw_hb,
                channel_name=channel_name,
                pipeline_label=pipeline_label,
                include_short_regressor=include_short_regressor,
            )
            canonical_metrics["subject"] = subject_name
            canonical_metrics["file_label"] = file_label
            canonical_metrics["amplitude_value"] = amplitude_value_map[file_label]
            canonical_rows.append(canonical_metrics)

            if pipeline_label == "PFIR":
                fir_channel_df = run_fir_glm_for_channel(
                    raw_hb=raw_hb,
                    channel_name=channel_name,
                    pipeline_label=pipeline_label,
                    include_short_regressor=True,
                )

                if len(fir_channel_df) > 0:
                    fir_channel_df["subject"] = subject_name
                    fir_channel_df["file_label"] = file_label
                    fir_channel_df["amplitude_value"] = amplitude_value_map[file_label]
                    fir_rows.append(fir_channel_df)

    canonical_channel_df = pd.DataFrame(canonical_rows)

    if len(fir_rows) > 0:
        fir_channel_df = pd.concat(fir_rows, ignore_index=True)
    else:
        fir_channel_df = pd.DataFrame()

    if len(pipeline_plot_tables) > 0:
        pipeline_plot_df = pd.concat(pipeline_plot_tables, ignore_index=True)
    else:
        pipeline_plot_df = pd.DataFrame()

    pipeline_epoch_metric_df = pd.DataFrame(epoch_rows)

    return {
        "pipeline_epoch_metric_df": pipeline_epoch_metric_df,
        "canonical_channel_df": canonical_channel_df,
        "fir_channel_df": fir_channel_df,
        "pipeline_plot_df": pipeline_plot_df,
    }


def compute_fir_shape_summary(fir_channel_df):
    if len(fir_channel_df) == 0:
        return pd.DataFrame()

    fir_shape_rows = []

    grouped_fir_df = fir_channel_df.groupby(["subject", "file_label", "pipeline_label", "chromophore"])

    for (subject_name_value, file_label_value, pipeline_label_value, chromophore), group_df in grouped_fir_df:
        fir_roi_timecourse_df = (
            group_df.groupby(["delay_s"], as_index=False)
            .agg(roi_mean_beta=("beta", "mean"))
            .sort_values("delay_s")
        )

        delays_s = fir_roi_timecourse_df["delay_s"].to_numpy()
        beta_values = fir_roi_timecourse_df["roi_mean_beta"].to_numpy()

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
            "subject": subject_name_value,
            "file_label": file_label_value,
            "amplitude_value": amplitude_value_map[file_label_value],
            "pipeline_label": pipeline_label_value,
            "chromophore": chromophore,
            "peak_delay_s": peak_delay_s,
            "peak_beta": peak_beta,
            "fwhm_s": fwhm_s,
        })

    fir_shape_summary_df = pd.DataFrame(fir_shape_rows)
    return fir_shape_summary_df


def compute_amplitude_recovery_summary(canonical_roi_summary_df):
    if len(canonical_roi_summary_df) == 0:
        return pd.DataFrame()

    amplitude_rows = []

    grouped_df = canonical_roi_summary_df.loc[
        canonical_roi_summary_df["chromophore"] == "hbo"
    ].groupby(["subject", "pipeline_label"])

    for (subject_name_value, pipeline_label), group_df in grouped_df:
        group_df = group_df.sort_values("amplitude_value")
        x = group_df["amplitude_value"].to_numpy(dtype=float)
        y = group_df["roi_mean_beta"].to_numpy(dtype=float)

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) < 2:
            continue

        slope, intercept = np.polyfit(x, y, deg=1)
        y_hat = intercept + slope * x
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan

        amplitude_rows.append({
            "subject": subject_name_value,
            "pipeline_label": pipeline_label,
            "beta_amplitude_slope": float(slope),
            "beta_amplitude_intercept": float(intercept),
            "beta_amplitude_rmse": rmse,
            "beta_amplitude_correlation": correlation,
        })

    amplitude_recovery_summary_df = pd.DataFrame(amplitude_rows)
    return amplitude_recovery_summary_df


def compute_null_false_positive_summary(canonical_channel_df):
    if len(canonical_channel_df) == 0:
        return pd.DataFrame()

    null_df = canonical_channel_df.loc[canonical_channel_df["file_label"] == "null"].copy()
    if len(null_df) == 0:
        return pd.DataFrame()

    if "p_value" not in null_df.columns:
        return pd.DataFrame()

    grouped_null_df = null_df.groupby(["subject", "pipeline_label", "chromophore"], as_index=False).agg(
        false_positive_rate=("p_value", lambda p_values: float(np.mean(np.array(p_values) < 0.05))),
        mean_null_beta=("beta", "mean"),
        mean_null_t=("t_value", "mean"),
        n_channels=("channel_name", "count"),
    )

    return grouped_null_df


def save_summary_figures(canonical_roi_summary_df, null_false_positive_summary_df, amplitude_recovery_summary_df):
    if len(canonical_roi_summary_df) > 0:
        beta_amplitude_figure_df = (
            canonical_roi_summary_df
            .groupby(["pipeline_label", "file_label", "amplitude_value", "chromophore"], as_index=False)
            .agg(
                roi_mean_beta=("roi_mean_beta", "mean"),
                roi_mean_t=("roi_mean_t", "mean"),
            )
            .sort_values(["chromophore", "amplitude_value", "pipeline_label"])
        )

        beta_amplitude_figure = px.line(
            beta_amplitude_figure_df,
            x="amplitude_value",
            y="roi_mean_beta",
            color="pipeline_label",
            facet_row="chromophore",
            markers=True,
            title="ROI mean beta across amplitude levels and pipelines",
        )
        beta_amplitude_figure.update_layout(width=1100, height=700)
        beta_amplitude_figure.write_html(output_figures_dir / "all_subjects_beta_amplitude_pipeline_comparison.html")

        t_amplitude_figure = px.line(
            beta_amplitude_figure_df,
            x="amplitude_value",
            y="roi_mean_t",
            color="pipeline_label",
            facet_row="chromophore",
            markers=True,
            title="ROI mean t across amplitude levels and pipelines",
        )
        t_amplitude_figure.update_layout(width=1100, height=700)
        t_amplitude_figure.write_html(output_figures_dir / "all_subjects_t_amplitude_pipeline_comparison.html")

    if len(null_false_positive_summary_df) > 0:
        null_fpr_plot_df = (
            null_false_positive_summary_df
            .groupby(["pipeline_label", "chromophore"], as_index=False)
            .agg(false_positive_rate=("false_positive_rate", "mean"))
        )

        null_fpr_figure = px.bar(
            null_fpr_plot_df,
            x="pipeline_label",
            y="false_positive_rate",
            color="chromophore",
            barmode="group",
            title="Null false positive rate across pipelines",
        )
        null_fpr_figure.update_layout(width=1000, height=500)
        null_fpr_figure.write_html(output_figures_dir / "all_subjects_null_false_positive_rate.html")

    if len(amplitude_recovery_summary_df) > 0:
        amplitude_slope_plot_df = (
            amplitude_recovery_summary_df
            .groupby(["pipeline_label"], as_index=False)
            .agg(
                beta_amplitude_slope=("beta_amplitude_slope", "mean"),
                beta_amplitude_rmse=("beta_amplitude_rmse", "mean"),
                beta_amplitude_correlation=("beta_amplitude_correlation", "mean"),
            )
        )

        amplitude_slope_figure = px.bar(
            amplitude_slope_plot_df,
            x="pipeline_label",
            y="beta_amplitude_slope",
            title="HbO beta-amplitude slope across pipelines",
        )
        amplitude_slope_figure.update_layout(width=900, height=500)
        amplitude_slope_figure.write_html(output_figures_dir / "all_subjects_beta_amplitude_slope.html")

        amplitude_rmse_figure = px.bar(
            amplitude_slope_plot_df,
            x="pipeline_label",
            y="beta_amplitude_rmse",
            title="HbO beta-amplitude RMSE across pipelines",
        )
        amplitude_rmse_figure.update_layout(width=900, height=500)
        amplitude_rmse_figure.write_html(output_figures_dir / "all_subjects_beta_amplitude_rmse.html")


def main():
    global subject_name
    global subject_dir
    global file_label
    global file_name
    global file_path

    subject_directories = sorted([subject_dir for subject_dir in rs_data_dir.glob("Subj*") if subject_dir.is_dir()])

    reference_channel_effect_tables = []
    target_pair_rows = []
    pipeline_epoch_metric_tables = []
    canonical_channel_tables = []
    fir_channel_tables = []
    pipeline_plot_tables = []

    for subject_dir in subject_directories:
        subject_name = subject_dir.name

        reference_channel_effect_df, top_hbo_channel_names, top_hbr_channel_names = build_subject_reference_channels(subject_name)

        if len(reference_channel_effect_df) == 0:
            print(f"Skipping {subject_name}: could not build reference channels")
            continue

        reference_channel_effect_tables.append(reference_channel_effect_df)

        target_pair_names = [channel_name.replace(" hbo", "") for channel_name in top_hbo_channel_names]
        for pair_order, pair_name in enumerate(target_pair_names, start=1):
            target_pair_rows.append({
                "subject": subject_name,
                "pair_order": pair_order,
                "pair_name": pair_name,
            })

        for file_label in ["null", "hrf_20", "hrf_50", "hrf_100"]:
            file_name = file_name_map[file_label]
            file_path = subject_dir / file_name

            subject_file_result = run_subject_file_pipeline_set(
                subject_name=subject_name,
                file_label=file_label,
                top_hbo_channel_names=top_hbo_channel_names,
                top_hbr_channel_names=top_hbr_channel_names,
            )

            if subject_file_result is None:
                print(f"Skipping {subject_name} {file_label}: missing or unusable file")
                continue

            pipeline_epoch_metric_tables.append(subject_file_result["pipeline_epoch_metric_df"])
            canonical_channel_tables.append(subject_file_result["canonical_channel_df"])

            if len(subject_file_result["fir_channel_df"]) > 0:
                fir_channel_tables.append(subject_file_result["fir_channel_df"])

            if len(subject_file_result["pipeline_plot_df"]) > 0:
                pipeline_plot_tables.append(subject_file_result["pipeline_plot_df"])

            print(f"Finished {subject_name} {file_label}")

    reference_channel_effect_df = pd.concat(reference_channel_effect_tables, ignore_index=True)
    target_pair_df = pd.DataFrame(target_pair_rows)
    pipeline_epoch_metric_df = pd.concat(pipeline_epoch_metric_tables, ignore_index=True)
    canonical_channel_df = pd.concat(canonical_channel_tables, ignore_index=True)

    if len(fir_channel_tables) > 0:
        fir_channel_df = pd.concat(fir_channel_tables, ignore_index=True)
    else:
        fir_channel_df = pd.DataFrame()

    if len(pipeline_plot_tables) > 0:
        pipeline_plot_df = pd.concat(pipeline_plot_tables, ignore_index=True)
    else:
        pipeline_plot_df = pd.DataFrame()

    canonical_roi_summary_df = (
        canonical_channel_df
        .groupby(["subject", "file_label", "amplitude_value", "pipeline_label", "chromophore"], as_index=False)
        .agg(
            roi_mean_beta=("beta", "mean"),
            roi_std_beta=("beta", "std"),
            roi_mean_t=("t_value", "mean"),
            roi_max_t=("t_value", "max"),
            roi_mean_p=("p_value", "mean") if "p_value" in canonical_channel_df.columns else ("beta", lambda x: np.nan),
            n_channels=("channel_name", "count"),
        )
    )

    fir_shape_summary_df = compute_fir_shape_summary(fir_channel_df)
    amplitude_recovery_summary_df = compute_amplitude_recovery_summary(canonical_roi_summary_df)
    null_false_positive_summary_df = compute_null_false_positive_summary(canonical_channel_df)

    pipeline_final_summary_df = (
        canonical_roi_summary_df
        .groupby(["file_label", "amplitude_value", "pipeline_label", "chromophore"], as_index=False)
        .agg(
            roi_mean_beta=("roi_mean_beta", "mean"),
            roi_std_beta=("roi_mean_beta", "std"),
            roi_mean_t=("roi_mean_t", "mean"),
            roi_std_t=("roi_mean_t", "std"),
            n_subjects=("subject", "nunique"),
        )
    )

    pipeline_epoch_summary_df = (
        pipeline_epoch_metric_df
        .groupby(["file_label", "amplitude_value", "pipeline_label"], as_index=False)
        .agg(
            hbo_mean_effect=("hbo_mean_effect", "mean"),
            hbr_mean_effect=("hbr_mean_effect", "mean"),
            n_subjects=("subject", "nunique"),
            mean_bad_pairs=("n_pairs_below_threshold", "mean"),
        )
    )

    save_summary_figures(
        canonical_roi_summary_df=canonical_roi_summary_df,
        null_false_positive_summary_df=null_false_positive_summary_df,
        amplitude_recovery_summary_df=amplitude_recovery_summary_df,
    )

    reference_channel_effect_df.to_csv(
        output_tables_dir / "all_subjects_reference_channel_effects_true_pipeline.csv",
        index=False,
    )
    target_pair_df.to_csv(
        output_tables_dir / "all_subjects_target_pairs_true_pipeline.csv",
        index=False,
    )
    pipeline_epoch_metric_df.to_csv(
        output_tables_dir / "all_subjects_pipeline_epoch_metrics_true_pipeline.csv",
        index=False,
    )
    pipeline_epoch_summary_df.to_csv(
        output_tables_dir / "all_subjects_pipeline_epoch_summary_true_pipeline.csv",
        index=False,
    )
    canonical_channel_df.to_csv(
        output_tables_dir / "all_subjects_canonical_channel_metrics_true_pipeline.csv",
        index=False,
    )
    canonical_roi_summary_df.to_csv(
        output_tables_dir / "all_subjects_canonical_roi_summary_true_pipeline.csv",
        index=False,
    )
    pipeline_final_summary_df.to_csv(
        output_tables_dir / "all_subjects_pipeline_final_summary_true_pipeline.csv",
        index=False,
    )
    amplitude_recovery_summary_df.to_csv(
        output_tables_dir / "all_subjects_amplitude_recovery_summary_true_pipeline.csv",
        index=False,
    )
    null_false_positive_summary_df.to_csv(
        output_tables_dir / "all_subjects_null_false_positive_summary_true_pipeline.csv",
        index=False,
    )

    if len(fir_channel_df) > 0:
        fir_channel_df.to_csv(
            output_tables_dir / "all_subjects_pfir_channel_timecourse_true_pipeline.csv",
            index=False,
        )
        fir_shape_summary_df.to_csv(
            output_tables_dir / "all_subjects_pfir_shape_summary_true_pipeline.csv",
            index=False,
        )

    if len(pipeline_plot_df) > 0:
        pipeline_plot_df.to_csv(
            output_tables_dir / "all_subjects_pipeline_plot_table_true_pipeline.csv",
            index=False,
        )

    print("Saved all-subject true pipeline outputs.")


if __name__ == "__main__":
    main()
