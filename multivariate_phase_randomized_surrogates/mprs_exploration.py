# %%
import math
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import welch

from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index,
    source_detector_distances,
    short_channels,
    temporal_derivative_distribution_repair,
)

px.defaults.template = "plotly_white"

# %%
# ---- project paths ----
ROOT = Path.home() / "fnirs-representation-learning"
DATASET_DIR = ROOT / "snirf_dataset_2"

# ---- choose one case to explore first ----
SUBJECT = "Subj100"

# best first choice for surrogate exploration: pure resting-state background
FILE_LABEL = "no_hrf"
FILE_NAME = "resting_clean.snirf"
ANNOTATION_SOURCE = "resting_hrf_20.snirf"   # only used for no_hrf

# You can switch to:
# FILE_LABEL = "hrf_20"
# FILE_NAME = "resting_hrf_20.snirf"

# ---- benchmark-matching preprocessing settings ----
CONFIG = {
    "short_separation_threshold_m": 0.015,
    "long_separation_threshold_m": 0.025,
    "strict_sci_threshold": 0.50,
    "strict_snr_threshold": 2.0,
    "strict_negative_fraction_threshold": 0.001,
    "filter_low_hz": 0.01,
    "filter_high_hz": 0.20,
    "filter_highpass_only_hz": 0.01,
    "ppf_value": 0.1,
    "wavelet_iqr_multiplier": 1.5,
    "wavelet_name": "db2",
    "wavelet_padding_mode": "periodization",
}

PIPELINE_LIKE = {
    "pruning_style": "strict_combined",
    "motion_method": "tddr",
    "filter_mode": "bandpass",
}

N_SURROGATES = 5
SEED = 42
N_TRACE_CHANNELS_TO_PLOT = 4

# %%
def get_cw_channel_indices(raw):
    picks_fnirs = mne.pick_types(raw.info, fnirs=True)
    channel_types = np.asarray(raw.get_channel_types())
    return picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]


def sanitize_annotations_to_single_task(raw, default_description="task"):
    raw = raw.copy()
    kept_onsets, kept_durations, kept_descriptions = [], [], []
    for onset_s, duration_s, description in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        if str(description).lower().startswith("bad"):
            continue
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(duration_s) if float(duration_s) > 0 else 1.0)
        kept_descriptions.append(default_description)

    raw.set_annotations(
        mne.Annotations(
            onset=kept_onsets,
            duration=kept_durations,
            description=kept_descriptions,
        )
    )
    return raw


def copy_valid_annotations(raw_target_cw, annotation_source_file_path):
    raw_source = mne.io.read_raw_snirf(annotation_source_file_path, preload=False, verbose=False)
    source_annotations = raw_source.annotations
    target_duration_s = float(raw_target_cw.times[-1])

    kept_onsets, kept_durations, kept_descriptions = [], [], []
    for onset_s, duration_s, description in zip(
        source_annotations.onset,
        source_annotations.duration,
        source_annotations.description,
    ):
        if str(description).lower().startswith("bad"):
            continue
        if float(onset_s) >= target_duration_s:
            continue

        clipped_duration = min(
            float(duration_s) if float(duration_s) > 0 else 1.0,
            max(0.0, target_duration_s - float(onset_s)),
        )
        kept_onsets.append(float(onset_s))
        kept_durations.append(float(clipped_duration))
        kept_descriptions.append("task")

    raw_target_cw = raw_target_cw.copy()
    raw_target_cw.set_annotations(
        mne.Annotations(
            onset=kept_onsets,
            duration=kept_durations,
            description=kept_descriptions,
        )
    )
    return raw_target_cw


def build_cw_channel_table(raw_cw, config):
    picks_cw = get_cw_channel_indices(raw_cw)
    cw_names = np.asarray(raw_cw.ch_names)[picks_cw]
    distances_m = source_detector_distances(raw_cw.info, picks=picks_cw)

    short_mask_all = short_channels(raw_cw.info, threshold=config["short_separation_threshold_m"])
    short_mask = short_mask_all[picks_cw]
    long_mask = distances_m >= config["long_separation_threshold_m"]

    pair_names = np.asarray([name.split(" ")[0] for name in cw_names])
    positions = np.asarray([raw_cw.info["chs"][int(idx)]["loc"][:3] for idx in picks_cw], dtype=float)

    df = pd.DataFrame({
        "channel_name": cw_names,
        "pair_name": pair_names,
        "distance_m": distances_m,
        "midpoint_x": positions[:, 0],
        "midpoint_y": positions[:, 1],
        "midpoint_z": positions[:, 2],
        "is_ss": short_mask,
        "is_ls": long_mask,
    })
    df["group"] = np.select([df["is_ss"], df["is_ls"]], ["SS", "LS"], default="MID")
    return df


def build_hb_channel_table(raw_hb, config):
    picks_hbo = mne.pick_types(raw_hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(raw_hb.info, fnirs="hbr")
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))

    hb_names = np.asarray(raw_hb.ch_names)[picks_hb]
    hb_types = np.asarray(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.asarray([name.split(" ")[0] for name in hb_names])

    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]

    short_mask_all = short_channels(raw_hb.info, threshold=config["short_separation_threshold_m"])
    short_mask = short_mask_all[picks_hb]
    long_mask = distances_hb >= config["long_separation_threshold_m"]

    positions = np.asarray([raw_hb.info["chs"][int(idx)]["loc"][:3] for idx in picks_hb], dtype=float)

    df = pd.DataFrame({
        "channel_name": hb_names,
        "pair_name": pair_names,
        "chromophore": hb_types,
        "distance_m": distances_hb,
        "midpoint_x": positions[:, 0],
        "midpoint_y": positions[:, 1],
        "midpoint_z": positions[:, 2],
        "is_ss": short_mask,
        "is_ls": long_mask,
    })
    df["group"] = np.select([df["is_ss"], df["is_ls"]], ["SS", "LS"], default="MID")
    return df


def build_quality_tables(raw_cw, config):
    raw_od = optical_density(raw_cw.copy())
    picks_cw = get_cw_channel_indices(raw_cw)
    cw_channel_table = build_cw_channel_table(raw_cw, config)

    cw_names = np.asarray(raw_cw.ch_names)[picks_cw]
    cw_data = raw_cw.get_data(picks=picks_cw)

    sci_values = scalp_coupling_index(raw_od)
    mean_abs_signal = np.mean(np.abs(cw_data), axis=1)
    signal_std = np.std(cw_data, axis=1)
    snr_values = mean_abs_signal / np.maximum(signal_std, 1e-12)
    negative_fraction_values = np.mean(cw_data <= 0, axis=1)

    channel_quality = pd.DataFrame({
        "channel_name": cw_names,
        "pair_name": cw_channel_table["pair_name"].to_numpy(),
        "distance_m": cw_channel_table["distance_m"].to_numpy(),
        "group": cw_channel_table["group"].to_numpy(),
        "sci": sci_values,
        "snr": snr_values,
        "negative_fraction": negative_fraction_values,
    })

    pair_geometry = (
        cw_channel_table.groupby("pair_name", as_index=False)
        .agg(
            distance_m=("distance_m", "first"),
            group=("group", "first"),
            midpoint_x=("midpoint_x", "mean"),
            midpoint_y=("midpoint_y", "mean"),
            midpoint_z=("midpoint_z", "mean"),
        )
    )

    pair_quality = (
        channel_quality.groupby("pair_name", as_index=False)
        .agg(
            sci_min=("sci", "min"),
            sci_mean=("sci", "mean"),
            snr_min=("snr", "min"),
            snr_mean=("snr", "mean"),
            negative_fraction_max=("negative_fraction", "max"),
        )
        .merge(pair_geometry, on="pair_name", how="left")
    )

    return channel_quality, pair_quality


def get_bad_pair_names(pair_quality, pruning_style, config):
    if pruning_style == "strict_combined":
        bad_mask = (
            (pair_quality["sci_min"] < config["strict_sci_threshold"]) |
            (pair_quality["snr_min"] < config["strict_snr_threshold"]) |
            (pair_quality["negative_fraction_max"] > config["strict_negative_fraction_threshold"])
        )
    else:
        raise ValueError(f"Unsupported pruning_style: {pruning_style}")

    return pair_quality.loc[bad_mask, "pair_name"].astype(str).tolist()


def apply_bad_pairs_to_hb(raw_hb, bad_pair_names):
    raw_hb = raw_hb.copy()
    bad_pairs = set(bad_pair_names)
    raw_hb.info["bads"] = [
        name for name in raw_hb.ch_names
        if name.split(" ")[0] in bad_pairs
    ]
    return raw_hb


def apply_motion_correction_od(raw_od, motion_method):
    motion_method = str(motion_method).lower()
    if motion_method == "tddr":
        return temporal_derivative_distribution_repair(raw_od.copy())
    if motion_method == "none":
        return raw_od.copy()
    raise ValueError(f"Unsupported motion_method: {motion_method}")


def apply_filter_mode(raw_obj, filter_mode, config):
    filter_mode = str(filter_mode).lower()
    filtered = raw_obj.copy()

    if filter_mode == "bandpass":
        return filtered.filter(
            config["filter_low_hz"],
            config["filter_high_hz"],
            verbose=False,
        )
    if filter_mode == "highpass_only":
        return filtered.filter(
            config["filter_highpass_only_hz"],
            None,
            verbose=False,
        )
    raise ValueError(f"Unsupported filter_mode: {filter_mode}")


def preprocess_raw_to_hb(raw_cw, pair_quality, pipeline_like, config):
    bad_pair_names = get_bad_pair_names(pair_quality, pipeline_like["pruning_style"], config)

    raw_od = optical_density(raw_cw.copy())
    processed_od = apply_motion_correction_od(raw_od, pipeline_like["motion_method"])
    processed_od = apply_filter_mode(processed_od, pipeline_like["filter_mode"], config)

    raw_hb = beer_lambert_law(processed_od, ppf=config["ppf_value"])
    raw_hb = apply_bad_pairs_to_hb(raw_hb, bad_pair_names)

    return raw_hb, bad_pair_names


def get_available_long_channel_names(raw_hb, config):
    hb_table = build_hb_channel_table(raw_hb, config)
    long_names = hb_table.loc[hb_table["group"] == "LS", "channel_name"].astype(str).tolist()
    return sorted([name for name in long_names if name not in raw_hb.info["bads"]])


def raw_hb_long_matrix(raw_hb, config):
    ls_names = get_available_long_channel_names(raw_hb, config)
    X = raw_hb.copy().pick(ls_names).get_data().T  # time x channels
    return X, ls_names


def make_surrogate_raw_hb(raw_hb, surrogate_matrix, selected_channel_names):
    surrogate_raw = raw_hb.copy().load_data()
    surrogate_raw._data[:] = np.nan

    selected_idx = [surrogate_raw.ch_names.index(ch) for ch in selected_channel_names]
    surrogate_raw._data[selected_idx, :] = surrogate_matrix.T

    # keep untouched channels from original for convenience
    original_data = raw_hb.get_data()
    untouched_idx = [i for i in range(len(raw_hb.ch_names)) if i not in selected_idx]
    surrogate_raw._data[untouched_idx, :] = original_data[untouched_idx, :]

    return surrogate_raw

# %%
def multivariate_phase_randomized_surrogate(X, rng=None):
    """
    X: shape (time, channels)

    Uses the same random phase shift across channels at each frequency,
    which helps preserve cross-channel covariance structure.
    """
    if rng is None:
        rng = np.random.default_rng()

    X = np.asarray(X, dtype=float)
    n_time, n_channels = X.shape

    channel_means = X.mean(axis=0, keepdims=True)
    X0 = X - channel_means

    F = np.fft.rfft(X0, axis=0)
    n_freq = F.shape[0]

    phase = np.zeros(n_freq)

    if n_time % 2 == 0:
        valid = np.arange(1, n_freq - 1)   # skip DC and Nyquist
    else:
        valid = np.arange(1, n_freq)       # skip DC only

    phase[valid] = rng.uniform(0, 2 * np.pi, size=len(valid))
    phase_factors = np.exp(1j * phase)[:, None]

    Fs = F * phase_factors
    Xs = np.fft.irfft(Fs, n=n_time, axis=0)
    Xs = Xs + channel_means

    return Xs


def make_surrogates(X, n_surrogates=5, seed=42):
    rng = np.random.default_rng(seed)
    return [multivariate_phase_randomized_surrogate(X, rng=rng) for _ in range(n_surrogates)]

# %%
subject_dir = DATASET_DIR / SUBJECT
snirf_path = subject_dir / FILE_NAME
annotation_source_path = subject_dir / ANNOTATION_SOURCE

print("SNIRF:", snirf_path)
print("Exists:", snirf_path.exists())

raw_cw = mne.io.read_raw_snirf(snirf_path, preload=True, verbose=False)

if FILE_LABEL == "no_hrf":
    raw_cw = copy_valid_annotations(raw_cw, annotation_source_path)

raw_cw = sanitize_annotations_to_single_task(raw_cw)

channel_quality_df, pair_quality_df = build_quality_tables(raw_cw, CONFIG)
raw_hb, bad_pair_names = preprocess_raw_to_hb(raw_cw, pair_quality_df, PIPELINE_LIKE, CONFIG)

hb_table = build_hb_channel_table(raw_hb, CONFIG)
X_orig, ls_channel_names = raw_hb_long_matrix(raw_hb, CONFIG)

print(f"n CW channels: {len(raw_cw.ch_names)}")
print(f"n bad pairs: {len(bad_pair_names)}")
print(f"n LS Hb channels retained: {len(ls_channel_names)}")
print(f"matrix shape (time x channels): {X_orig.shape}")

display(pair_quality_df.head())
display(hb_table.head())

# %%
pair_group_counts = pair_quality_df["group"].value_counts().rename_axis("group").reset_index(name="count")
fig = px.bar(pair_group_counts, x="group", y="count", title=f"{SUBJECT} {FILE_LABEL}: pair counts by group")
fig.show()

quality_long = pair_quality_df.melt(
    id_vars=["pair_name", "group"],
    value_vars=["sci_min", "snr_min", "negative_fraction_max"],
    var_name="metric",
    value_name="value",
)
fig = px.histogram(
    quality_long,
    x="value",
    color="metric",
    facet_col="metric",
    facet_col_wrap=1,
    title=f"{SUBJECT} {FILE_LABEL}: QC metric distributions",
)
fig.update_layout(height=900)
fig.show()

# %%
X_surrogates = make_surrogates(X_orig, n_surrogates=N_SURROGATES, seed=SEED)
surrogate_raws = [
    make_surrogate_raw_hb(raw_hb, Xs, ls_channel_names)
    for Xs in X_surrogates
]

print(f"Made {len(X_surrogates)} surrogates")
print("Example surrogate shape:", X_surrogates[0].shape)

# %%
def corr_matrix(X):
    return np.corrcoef(X.T)

def channelwise_psd(X, sfreq):
    freqs, psd = welch(X, fs=sfreq, axis=0, nperseg=min(1024, X.shape[0]))
    return freqs, psd

def surrogate_summary_table(X_orig, X_surrogates, sfreq):
    orig_corr = corr_matrix(X_orig)
    orig_stds = X_orig.std(axis=0)
    f0, psd0 = channelwise_psd(X_orig, sfreq)

    rows = []
    for i, Xs in enumerate(X_surrogates, start=1):
        sur_corr = corr_matrix(Xs)
        sur_stds = Xs.std(axis=0)
        f1, psd1 = channelwise_psd(Xs, sfreq)

        rows.append({
            "surrogate_id": i,
            "mean_abs_corr_diff": float(np.mean(np.abs(orig_corr - sur_corr))),
            "mean_abs_std_diff": float(np.mean(np.abs(orig_stds - sur_stds))),
            "mean_abs_psd_diff": float(np.mean(np.abs(psd0 - psd1))),
            "orig_global_mean": float(np.mean(X_orig)),
            "sur_global_mean": float(np.mean(Xs)),
            "orig_global_std": float(np.std(X_orig)),
            "sur_global_std": float(np.std(Xs)),
        })

    return pd.DataFrame(rows)

summary_df = surrogate_summary_table(X_orig, X_surrogates, raw_hb.info["sfreq"])
display(summary_df)

fig = px.bar(
    summary_df,
    x="surrogate_id",
    y=["mean_abs_corr_diff", "mean_abs_std_diff", "mean_abs_psd_diff"],
    barmode="group",
    title="Original vs surrogate summary differences",
)
fig.show()

# %%
plot_channels = ls_channel_names[:N_TRACE_CHANNELS_TO_PLOT]
sur_idx = 0

trace_rows = []
for ch in plot_channels:
    ch_idx = ls_channel_names.index(ch)

    for t, y in zip(raw_hb.times, X_orig[:, ch_idx]):
        trace_rows.append({
            "time_s": t,
            "signal": y,
            "channel_name": ch,
            "source": "original",
        })

    for t, y in zip(raw_hb.times, X_surrogates[sur_idx][:, ch_idx]):
        trace_rows.append({
            "time_s": t,
            "signal": y,
            "channel_name": ch,
            "source": f"surrogate_{sur_idx+1}",
        })

trace_df = pd.DataFrame(trace_rows)

fig = px.line(
    trace_df,
    x="time_s",
    y="signal",
    color="source",
    facet_row="channel_name",
    title=f"{SUBJECT} {FILE_LABEL}: original vs surrogate traces",
)
fig.update_layout(height=900)
fig.show()

# %%
sur_idx = 0
sfreq = raw_hb.info["sfreq"]

freqs_orig, psd_orig = channelwise_psd(X_orig, sfreq)
freqs_sur, psd_sur = channelwise_psd(X_surrogates[sur_idx], sfreq)

psd_rows = []
for ch in plot_channels:
    ch_idx = ls_channel_names.index(ch)

    for f, p in zip(freqs_orig, psd_orig[:, ch_idx]):
        psd_rows.append({
            "freq_hz": f,
            "psd": p,
            "channel_name": ch,
            "source": "original",
        })

    for f, p in zip(freqs_sur, psd_sur[:, ch_idx]):
        psd_rows.append({
            "freq_hz": f,
            "psd": p,
            "channel_name": ch,
            "source": f"surrogate_{sur_idx+1}",
        })

psd_df = pd.DataFrame(psd_rows)
psd_df["log_psd"] = np.log10(np.maximum(psd_df["psd"], 1e-16))

fig = px.line(
    psd_df,
    x="freq_hz",
    y="log_psd",
    color="source",
    facet_row="channel_name",
    title=f"{SUBJECT} {FILE_LABEL}: original vs surrogate PSD",
)
fig.update_layout(height=900)
fig.show()

# %%
sur_idx = 0
orig_corr = corr_matrix(X_orig)
sur_corr = corr_matrix(X_surrogates[sur_idx])
corr_diff = sur_corr - orig_corr

fig = px.imshow(
    orig_corr,
    x=ls_channel_names,
    y=ls_channel_names,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="Original LS Hb channel correlation matrix",
)
fig.update_layout(height=800, width=900)
fig.show()

fig = px.imshow(
    sur_corr,
    x=ls_channel_names,
    y=ls_channel_names,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title=f"Surrogate {sur_idx+1} LS Hb channel correlation matrix",
)
fig.update_layout(height=800, width=900)
fig.show()

fig = px.imshow(
    corr_diff,
    x=ls_channel_names,
    y=ls_channel_names,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title=f"Correlation difference: surrogate {sur_idx+1} - original",
)
fig.update_layout(height=800, width=900)
fig.show()

# %%
sur_idx = 0
dist_rows = []

for ch in plot_channels:
    ch_idx = ls_channel_names.index(ch)

    for y in X_orig[:, ch_idx]:
        dist_rows.append({
            "signal": y,
            "channel_name": ch,
            "source": "original",
        })

    for y in X_surrogates[sur_idx][:, ch_idx]:
        dist_rows.append({
            "signal": y,
            "channel_name": ch,
            "source": f"surrogate_{sur_idx+1}",
        })

dist_df = pd.DataFrame(dist_rows)

fig = px.histogram(
    dist_df,
    x="signal",
    color="source",
    facet_row="channel_name",
    marginal="box",
    barmode="overlay",
    opacity=0.55,
    title=f"{SUBJECT} {FILE_LABEL}: original vs surrogate channel-value distributions",
)
fig.update_layout(height=900)
fig.show()

# %%
sur_idx = 0
sur_raw_hb = surrogate_raws[sur_idx]

print(sur_raw_hb)
print("Annotations copied over:", len(sur_raw_hb.annotations))

# sanity check that info/ch names are preserved
assert sur_raw_hb.ch_names == raw_hb.ch_names
assert sur_raw_hb.info["sfreq"] == raw_hb.info["sfreq"]


