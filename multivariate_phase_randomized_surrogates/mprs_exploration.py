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

from mne_nirs.io import write_raw_snirf

px.defaults.template = "plotly_white"

# %%
# %%
ROOT = Path.home() / "fnirs-representation-learning"
DATASET_DIR = ROOT / "snirf_dataset_2"
SANDBOX_DIR = ROOT / "mprs_od_cw_exploration"
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT = "Subj102"
FILE_LABEL = "no_hrf"
FILE_NAME = "resting_clean.snirf"
ANNOTATION_SOURCE = "resting_hrf_20.snirf"   # only used for no_hrf

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
}

PIPELINE_LIKE = {
    "pruning_style": "strict_combined",
    "motion_method": "tddr",
    "filter_mode": "bandpass",
}

N_SURROGATES = 5
SEED = 42
N_TRACE_CHANNELS_TO_PLOT = 4
SURROGATE_TO_INSPECT = 1  # 1-based

# %%
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


def corr_matrix(X):
    return np.corrcoef(X.T)


def channelwise_psd(X, sfreq):
    freqs, psd = welch(X, fs=sfreq, axis=0, nperseg=min(1024, X.shape[0]))
    return freqs, psd

# %%
# %%
def get_all_cw_matrices(raw_cw):
    cw_idx = get_cw_channel_indices(raw_cw)
    cw_names = np.asarray(raw_cw.ch_names)[cw_idx].tolist()

    cw_matrix = raw_cw.get_data(picks=cw_idx).T  # time x channels
    raw_od = optical_density(raw_cw.copy())
    od_matrix = raw_od.get_data()[cw_idx, :].T   # same channel order

    cw_channel_means = cw_matrix.mean(axis=0)
    cw_channel_means = np.maximum(cw_channel_means, 1e-12)

    return {
        "cw_idx": cw_idx,
        "cw_names": cw_names,
        "cw_matrix": cw_matrix,
        "raw_od": raw_od,
        "od_matrix": od_matrix,
        "cw_channel_means": cw_channel_means,
    }


def multivariate_phase_randomized_surrogate(X, rng=None):
    """
    X: shape (time, channels)

    Uses the same random phase shift across channels at each frequency.
    This is the conservative multivariate version.
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


def od_to_cw_matrix(od_matrix, cw_channel_means):
    """
    Approximate inverse of OD = -log(I / mean(I)).
    Reconstruct positive CW amplitudes channelwise.
    """
    cw_matrix = cw_channel_means[None, :] * np.exp(-od_matrix)
    cw_matrix = np.maximum(cw_matrix, 1e-12)
    return cw_matrix


def make_surrogate_raw_cw(raw_cw, surrogate_cw_matrix, cw_idx):
    surrogate_raw = raw_cw.copy().load_data()
    surrogate_raw_data = surrogate_raw.get_data().copy()
    surrogate_raw_data[cw_idx, :] = surrogate_cw_matrix.T
    surrogate_raw._data = surrogate_raw_data
    return surrogate_raw


def retained_ls_hb_matrix(raw_hb, config):
    ls_names = get_available_long_channel_names(raw_hb, config)
    X = raw_hb.copy().pick(ls_names).get_data().T
    return X, ls_names


def common_retained_ls_matrices(raw_hb_a, raw_hb_b, config):
    ls_a = set(get_available_long_channel_names(raw_hb_a, config))
    ls_b = set(get_available_long_channel_names(raw_hb_b, config))
    common = sorted(ls_a.intersection(ls_b))
    Xa = raw_hb_a.copy().pick(common).get_data().T
    Xb = raw_hb_b.copy().pick(common).get_data().T
    return Xa, Xb, common


def surrogate_summary_table(X_orig, X_surrogates, sfreq):
    orig_corr = corr_matrix(X_orig)
    orig_stds = X_orig.std(axis=0)
    _, psd0 = channelwise_psd(X_orig, sfreq)

    rows = []
    for i, Xs in enumerate(X_surrogates, start=1):
        sur_corr = corr_matrix(Xs)
        sur_stds = Xs.std(axis=0)
        _, psd1 = channelwise_psd(Xs, sfreq)

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

# %%
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

cw_bundle = get_all_cw_matrices(raw_cw)

print(f"n CW channels: {len(cw_bundle['cw_names'])}")
print(f"CW matrix shape (time x channels): {cw_bundle['cw_matrix'].shape}")
print(f"OD matrix shape (time x channels): {cw_bundle['od_matrix'].shape}")

display(pair_quality_df.head())
display(build_cw_channel_table(raw_cw, CONFIG).head())

# %%
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
    title=f"{SUBJECT} {FILE_LABEL}: original CW QC metric distributions",
)
fig.update_layout(height=900)
fig.show()

# %%
# %%
X_od_orig = cw_bundle["od_matrix"]
X_cw_orig = cw_bundle["cw_matrix"]
cw_idx = cw_bundle["cw_idx"]
cw_names = cw_bundle["cw_names"]
cw_channel_means = cw_bundle["cw_channel_means"]

X_od_surrogates = make_surrogates(X_od_orig, n_surrogates=N_SURROGATES, seed=SEED)
X_cw_surrogates = [od_to_cw_matrix(Xs, cw_channel_means) for Xs in X_od_surrogates]
surrogate_raw_cws = [make_surrogate_raw_cw(raw_cw, Xcw, cw_idx) for Xcw in X_cw_surrogates]

print(f"Made {len(X_od_surrogates)} OD-stage surrogates")
print("Example OD surrogate shape:", X_od_surrogates[0].shape)
print("Example CW surrogate min:", float(X_cw_surrogates[0].min()))
print("Example CW surrogate nonpositive fraction:", float(np.mean(X_cw_surrogates[0] <= 0)))

# %%
# %%
summary_od_df = surrogate_summary_table(X_od_orig, X_od_surrogates, raw_cw.info["sfreq"])
display(summary_od_df)

fig = px.bar(
    summary_od_df,
    x="surrogate_id",
    y=["mean_abs_corr_diff", "mean_abs_std_diff", "mean_abs_psd_diff"],
    barmode="group",
    title="OD-space original vs surrogate summary differences",
)
fig.show()

# %%
# %%
sur_idx = SURROGATE_TO_INSPECT - 1
plot_channels = cw_names[:N_TRACE_CHANNELS_TO_PLOT]

trace_rows = []
for ch in plot_channels:
    ch_idx = cw_names.index(ch)

    for t, y in zip(raw_cw.times, X_od_orig[:, ch_idx]):
        trace_rows.append({
            "time_s": t,
            "signal": y,
            "channel_name": ch,
            "source": "original_od",
        })

    for t, y in zip(raw_cw.times, X_od_surrogates[sur_idx][:, ch_idx]):
        trace_rows.append({
            "time_s": t,
            "signal": y,
            "channel_name": ch,
            "source": f"surrogate_{sur_idx+1}_od",
        })

trace_df = pd.DataFrame(trace_rows)

fig = px.line(
    trace_df,
    x="time_s",
    y="signal",
    color="source",
    facet_row="channel_name",
    title=f"{SUBJECT} {FILE_LABEL}: original vs surrogate OD traces",
)
fig.update_layout(height=900)
fig.show()

# %%
# %%
freqs_orig, psd_orig = channelwise_psd(X_od_orig, raw_cw.info["sfreq"])
freqs_sur, psd_sur = channelwise_psd(X_od_surrogates[sur_idx], raw_cw.info["sfreq"])

psd_rows = []
for ch in plot_channels:
    ch_idx = cw_names.index(ch)

    for f, p in zip(freqs_orig, psd_orig[:, ch_idx]):
        psd_rows.append({
            "freq_hz": f,
            "psd": p,
            "channel_name": ch,
            "source": "original_od",
        })

    for f, p in zip(freqs_sur, psd_sur[:, ch_idx]):
        psd_rows.append({
            "freq_hz": f,
            "psd": p,
            "channel_name": ch,
            "source": f"surrogate_{sur_idx+1}_od",
        })

psd_df = pd.DataFrame(psd_rows)
psd_df["log_psd"] = np.log10(np.maximum(psd_df["psd"], 1e-16))

fig = px.line(
    psd_df,
    x="freq_hz",
    y="log_psd",
    color="source",
    facet_row="channel_name",
    title=f"{SUBJECT} {FILE_LABEL}: original vs surrogate OD PSD",
)
fig.update_layout(height=900)
fig.show()

# %%
# %%
cw_sanity_rows = []
for i, Xcw in enumerate(X_cw_surrogates, start=1):
    cw_sanity_rows.append({
        "surrogate_id": i,
        "min_cw": float(Xcw.min()),
        "max_cw": float(Xcw.max()),
        "mean_cw": float(Xcw.mean()),
        "std_cw": float(Xcw.std()),
        "nonpositive_fraction": float(np.mean(Xcw <= 0)),
    })

cw_sanity_df = pd.DataFrame(cw_sanity_rows)
display(cw_sanity_df)

fig = px.bar(
    cw_sanity_df,
    x="surrogate_id",
    y=["min_cw", "mean_cw", "std_cw", "nonpositive_fraction"],
    barmode="group",
    title="Reconstructed CW surrogate sanity metrics",
)
fig.show()

# %%
# %%
surrogate_snirf_path = SANDBOX_DIR / f"{SUBJECT}_{FILE_LABEL}_surrogate_{sur_idx+1:02d}.snirf"

write_raw_snirf(surrogate_raw_cws[sur_idx], str(surrogate_snirf_path))
print("Wrote:", surrogate_snirf_path)

reloaded_sur_raw_cw = mne.io.read_raw_snirf(surrogate_snirf_path, preload=True, verbose=False)
print(reloaded_sur_raw_cw)
print("Annotations:", len(reloaded_sur_raw_cw.annotations))
print("Channel types head:", reloaded_sur_raw_cw.get_channel_types()[:10])

# %%
# %%
sur_channel_quality_df, sur_pair_quality_df = build_quality_tables(reloaded_sur_raw_cw, CONFIG)

common_pairs = sorted(set(pair_quality_df["pair_name"]).intersection(set(sur_pair_quality_df["pair_name"])))
pair_compare_df = (
    pair_quality_df[pair_quality_df["pair_name"].isin(common_pairs)]
    .merge(
        sur_pair_quality_df[sur_pair_quality_df["pair_name"].isin(common_pairs)],
        on="pair_name",
        suffixes=("_orig", "_sur"),
    )
)

print("Original pair count:", len(pair_quality_df))
print("Surrogate pair count:", len(sur_pair_quality_df))
print("Common pairs:", len(pair_compare_df))

display(pair_compare_df.head())

for metric in ["sci_min", "snr_min", "negative_fraction_max"]:
    fig = px.scatter(
        pair_compare_df,
        x=f"{metric}_orig",
        y=f"{metric}_sur",
        hover_name="pair_name",
        title=f"Original vs surrogate pair QC: {metric}",
    )
    fig.add_shape(
        type="line",
        x0=pair_compare_df[f"{metric}_orig"].min(),
        y0=pair_compare_df[f"{metric}_orig"].min(),
        x1=pair_compare_df[f"{metric}_orig"].max(),
        y1=pair_compare_df[f"{metric}_orig"].max(),
        line=dict(dash="dash"),
    )
    fig.show()

# %%
# %%
ALL_SUBJECTS = sorted(
    [p.name for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("Subj")]
)
ALL_SUBJECTS

# %%
# %%
def compare_one_surrogate_for_subject(subject, surrogate_id=1):
    subject_dir = DATASET_DIR / subject
    snirf_path = subject_dir / "resting_clean.snirf"
    annotation_source_path = subject_dir / "resting_hrf_20.snirf"

    raw_cw = mne.io.read_raw_snirf(snirf_path, preload=True, verbose=False)
    raw_cw = copy_valid_annotations(raw_cw, annotation_source_path)
    raw_cw = sanitize_annotations_to_single_task(raw_cw)

    # Original CW QC
    channel_quality_df, pair_quality_df = build_quality_tables(raw_cw, CONFIG)

    # Build OD/CW matrices
    cw_bundle = get_all_cw_matrices(raw_cw)
    X_od_orig = cw_bundle["od_matrix"]
    cw_idx = cw_bundle["cw_idx"]
    cw_names = cw_bundle["cw_names"]
    cw_channel_means = cw_bundle["cw_channel_means"]

    # Make all surrogates, then select one
    X_od_surrogates = make_surrogates(X_od_orig, n_surrogates=N_SURROGATES, seed=SEED)
    X_od_sur = X_od_surrogates[surrogate_id - 1]
    X_cw_sur = od_to_cw_matrix(X_od_sur, cw_channel_means)
    surrogate_raw_cw = make_surrogate_raw_cw(raw_cw, X_cw_sur, cw_idx)

    # Write/reload exactly like the real path
    out_path = SANDBOX_DIR / f"{subject}_no_hrf_surrogate_{surrogate_id:02d}.snirf"
    write_raw_snirf(surrogate_raw_cw, str(out_path))
    reloaded_sur_raw_cw = mne.io.read_raw_snirf(out_path, preload=True, verbose=False)

    # Surrogate CW QC
    sur_channel_quality_df, sur_pair_quality_df = build_quality_tables(reloaded_sur_raw_cw, CONFIG)

    # Pairwise QC compare on common pairs
    common_pairs = sorted(set(pair_quality_df["pair_name"]).intersection(set(sur_pair_quality_df["pair_name"])))
    pair_compare_df = (
        pair_quality_df[pair_quality_df["pair_name"].isin(common_pairs)]
        .merge(
            sur_pair_quality_df[sur_pair_quality_df["pair_name"].isin(common_pairs)],
            on="pair_name",
            suffixes=("_orig", "_sur"),
        )
    )

    # Threshold crossing counts
    sci_thr = CONFIG["strict_sci_threshold"]
    snr_thr = CONFIG["strict_snr_threshold"]
    neg_thr = CONFIG["strict_negative_fraction_threshold"]

    sci_cross_fail = ((pair_compare_df["sci_min_orig"] >= sci_thr) & (pair_compare_df["sci_min_sur"] < sci_thr)).sum()
    sci_cross_recover = ((pair_compare_df["sci_min_orig"] < sci_thr) & (pair_compare_df["sci_min_sur"] >= sci_thr)).sum()

    snr_cross_fail = ((pair_compare_df["snr_min_orig"] >= snr_thr) & (pair_compare_df["snr_min_sur"] < snr_thr)).sum()
    snr_cross_recover = ((pair_compare_df["snr_min_orig"] < snr_thr) & (pair_compare_df["snr_min_sur"] >= snr_thr)).sum()

    neg_cross_fail = ((pair_compare_df["negative_fraction_max_orig"] <= neg_thr) &
                      (pair_compare_df["negative_fraction_max_sur"] > neg_thr)).sum()
    neg_cross_recover = ((pair_compare_df["negative_fraction_max_orig"] > neg_thr) &
                         (pair_compare_df["negative_fraction_max_sur"] <= neg_thr)).sum()

    # Bad-pair counts
    bad_orig = get_bad_pair_names(pair_quality_df, PIPELINE_LIKE["pruning_style"], CONFIG)
    bad_sur = get_bad_pair_names(sur_pair_quality_df, PIPELINE_LIKE["pruning_style"], CONFIG)

    # OD-space summary
    od_summary = surrogate_summary_table(X_od_orig, [X_od_sur], raw_cw.info["sfreq"]).iloc[0]

    # CW sanity
    cw_nonpositive_fraction = float(np.mean(X_cw_sur <= 0))
    cw_min = float(X_cw_sur.min())
    cw_mean = float(X_cw_sur.mean())
    cw_std = float(X_cw_sur.std())

    # Downstream benchmark path: preprocess both to Hb
    raw_hb_orig, bad_pairs_orig = preprocess_raw_to_hb(raw_cw, pair_quality_df, PIPELINE_LIKE, CONFIG)
    raw_hb_sur, bad_pairs_sur = preprocess_raw_to_hb(reloaded_sur_raw_cw, sur_pair_quality_df, PIPELINE_LIKE, CONFIG)

    Xa_hb, Xs_hb, common_ls_hb = common_retained_ls_matrices(raw_hb_orig, raw_hb_sur, CONFIG)

    if len(common_ls_hb) > 1:
        hb_summary = surrogate_summary_table(Xa_hb, [Xs_hb], raw_hb_orig.info["sfreq"]).iloc[0]
        hb_corr_diff = float(hb_summary["mean_abs_corr_diff"])
        hb_std_diff = float(hb_summary["mean_abs_std_diff"])
        hb_psd_diff = float(hb_summary["mean_abs_psd_diff"])
    else:
        hb_corr_diff = np.nan
        hb_std_diff = np.nan
        hb_psd_diff = np.nan

    return {
        "subject": subject,
        "surrogate_id": surrogate_id,

        "n_pairs_orig": int(len(pair_quality_df)),
        "n_pairs_sur": int(len(sur_pair_quality_df)),
        "n_common_pairs": int(len(pair_compare_df)),

        "n_bad_pairs_orig": int(len(bad_orig)),
        "n_bad_pairs_sur": int(len(bad_sur)),
        "delta_bad_pairs": int(len(bad_sur) - len(bad_orig)),

        "sci_cross_fail": int(sci_cross_fail),
        "sci_cross_recover": int(sci_cross_recover),
        "snr_cross_fail": int(snr_cross_fail),
        "snr_cross_recover": int(snr_cross_recover),
        "neg_cross_fail": int(neg_cross_fail),
        "neg_cross_recover": int(neg_cross_recover),

        "mean_abs_sci_diff": float(np.mean(np.abs(pair_compare_df["sci_min_sur"] - pair_compare_df["sci_min_orig"]))),
        "mean_abs_snr_diff": float(np.mean(np.abs(pair_compare_df["snr_min_sur"] - pair_compare_df["snr_min_orig"]))),
        "mean_abs_negfrac_diff": float(np.mean(np.abs(
            pair_compare_df["negative_fraction_max_sur"] - pair_compare_df["negative_fraction_max_orig"]
        ))),

        "od_mean_abs_corr_diff": float(od_summary["mean_abs_corr_diff"]),
        "od_mean_abs_std_diff": float(od_summary["mean_abs_std_diff"]),
        "od_mean_abs_psd_diff": float(od_summary["mean_abs_psd_diff"]),

        "cw_min": cw_min,
        "cw_mean": cw_mean,
        "cw_std": cw_std,
        "cw_nonpositive_fraction": cw_nonpositive_fraction,

        "n_ls_hb_orig": int(len(get_available_long_channel_names(raw_hb_orig, CONFIG))),
        "n_ls_hb_sur": int(len(get_available_long_channel_names(raw_hb_sur, CONFIG))),
        "n_ls_hb_common": int(len(common_ls_hb)),

        "hb_mean_abs_corr_diff": hb_corr_diff,
        "hb_mean_abs_std_diff": hb_std_diff,
        "hb_mean_abs_psd_diff": hb_psd_diff,
    }

# %%
# %%
rows = []

for subject in ALL_SUBJECTS:
    print(f"Processing {subject}...")
    for surrogate_id in range(1, N_SURROGATES + 1):
        row = compare_one_surrogate_for_subject(subject, surrogate_id=surrogate_id)
        rows.append(row)

summary_long_df = pd.DataFrame(rows)
summary_long_df

# %%
# %%
summary_csv = SANDBOX_DIR / "mprs_all_subjects_summary_long.csv"
summary_long_df.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)

# %%
# %%
subject_summary_df = (
    summary_long_df.groupby("subject", as_index=False)
    .agg(
        sci_cross_fail_mean=("sci_cross_fail", "mean"),
        sci_cross_fail_max=("sci_cross_fail", "max"),
        snr_cross_fail_mean=("snr_cross_fail", "mean"),
        snr_cross_fail_max=("snr_cross_fail", "max"),
        delta_bad_pairs_mean=("delta_bad_pairs", "mean"),
        delta_bad_pairs_max=("delta_bad_pairs", "max"),

        od_mean_abs_psd_diff_mean=("od_mean_abs_psd_diff", "mean"),
        od_mean_abs_psd_diff_max=("od_mean_abs_psd_diff", "max"),
        od_mean_abs_corr_diff_mean=("od_mean_abs_corr_diff", "mean"),

        hb_mean_abs_psd_diff_mean=("hb_mean_abs_psd_diff", "mean"),
        hb_mean_abs_psd_diff_max=("hb_mean_abs_psd_diff", "max"),
        hb_mean_abs_corr_diff_mean=("hb_mean_abs_corr_diff", "mean"),

        n_ls_hb_orig_mean=("n_ls_hb_orig", "mean"),
        n_ls_hb_sur_mean=("n_ls_hb_sur", "mean"),
        n_ls_hb_common_mean=("n_ls_hb_common", "mean"),

        cw_nonpositive_fraction_max=("cw_nonpositive_fraction", "max"),
    )
)

subject_summary_df

# %%
# %%
subject_summary_df = (
    summary_long_df.groupby("subject", as_index=False)
    .agg(
        sci_cross_fail_mean=("sci_cross_fail", "mean"),
        sci_cross_fail_max=("sci_cross_fail", "max"),
        snr_cross_fail_mean=("snr_cross_fail", "mean"),
        snr_cross_fail_max=("snr_cross_fail", "max"),
        delta_bad_pairs_mean=("delta_bad_pairs", "mean"),
        delta_bad_pairs_max=("delta_bad_pairs", "max"),

        od_mean_abs_psd_diff_mean=("od_mean_abs_psd_diff", "mean"),
        od_mean_abs_psd_diff_max=("od_mean_abs_psd_diff", "max"),
        od_mean_abs_corr_diff_mean=("od_mean_abs_corr_diff", "mean"),

        hb_mean_abs_psd_diff_mean=("hb_mean_abs_psd_diff", "mean"),
        hb_mean_abs_psd_diff_max=("hb_mean_abs_psd_diff", "max"),
        hb_mean_abs_corr_diff_mean=("hb_mean_abs_corr_diff", "mean"),

        n_ls_hb_orig_mean=("n_ls_hb_orig", "mean"),
        n_ls_hb_sur_mean=("n_ls_hb_sur", "mean"),
        n_ls_hb_common_mean=("n_ls_hb_common", "mean"),

        cw_nonpositive_fraction_max=("cw_nonpositive_fraction", "max"),
    )
)

subject_summary_df

# %%
# %%
fig = px.bar(
    subject_summary_df.sort_values("delta_bad_pairs_max", ascending=False),
    x="subject",
    y=["delta_bad_pairs_mean", "delta_bad_pairs_max"],
    barmode="group",
    title="Change in bad-pair counts by subject",
)
fig.show()

# %%
# %%
fig = px.scatter(
    subject_summary_df,
    x="n_ls_hb_orig_mean",
    y="n_ls_hb_sur_mean",
    hover_name="subject",
    title="Original vs surrogate retained LS Hb channels",
)
fig.add_shape(
    type="line",
    x0=subject_summary_df["n_ls_hb_orig_mean"].min(),
    y0=subject_summary_df["n_ls_hb_orig_mean"].min(),
    x1=subject_summary_df["n_ls_hb_orig_mean"].max(),
    y1=subject_summary_df["n_ls_hb_orig_mean"].max(),
    line=dict(dash="dash"),
)
fig.show()

# %%
# %%
plot_df = subject_summary_df.melt(
    id_vars=["subject"],
    value_vars=[
        "od_mean_abs_psd_diff_mean",
        "od_mean_abs_psd_diff_max",
        "hb_mean_abs_psd_diff_mean",
        "hb_mean_abs_psd_diff_max",
    ],
    var_name="metric",
    value_name="value",
)

fig = px.bar(
    plot_df,
    x="subject",
    y="value",
    color="metric",
    barmode="group",
    title="OD/Hb PSD difference summary by subject",
)
fig.show()

# %%
# %%
review_df = subject_summary_df.copy()

review_df["flag_sci"] = review_df["sci_cross_fail_max"] > 3
review_df["flag_bad_pairs"] = review_df["delta_bad_pairs_max"] > 3
review_df["flag_cw_nonpositive"] = review_df["cw_nonpositive_fraction_max"] > 0
review_df["flag_ls_drop"] = review_df["n_ls_hb_sur_mean"] < 0.9 * review_df["n_ls_hb_orig_mean"]

review_df["needs_manual_review"] = review_df[
    ["flag_sci", "flag_bad_pairs", "flag_cw_nonpositive", "flag_ls_drop"]
].any(axis=1)

review_df.sort_values(
    ["needs_manual_review", "sci_cross_fail_max", "delta_bad_pairs_max"],
    ascending=[False, False, False],
)


