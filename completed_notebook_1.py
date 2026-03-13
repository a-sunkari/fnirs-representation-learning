# %% [markdown]
# # Includes

# %%
import pandas as pd
import numpy as np
import mne
import mne_nirs
import h5py
import shutil
import plotly.express as px

from pathlib import Path
from mne.preprocessing.nirs import source_detector_distances, short_channels, optical_density, beer_lambert_law

root = Path.home() / "fnirs-representation-learning"
rs_data_dir = root / "snirf_dataset_2"

# %%
clean_file_rows = []

for subject_dir in sorted(rs_data_dir.glob("Subj*")):
    if not subject_dir.is_dir():
        continue

    resting_file = subject_dir / "resting.snirf"
    clean_file = subject_dir / "resting_clean.snirf"

    if not clean_file.exists():
        shutil.copy2(resting_file, clean_file)

        with h5py.File(clean_file, "r+") as f:
            if "stim1" in f["nirs"]:
                del f["nirs"]["stim1"]

        print("created:", clean_file)
        
    else:
        print("already exists:", clean_file)

    clean_file_rows.append({
        "subject": subject_dir.name,
        "clean_file": clean_file,
    })

clean_files_df = pd.DataFrame(clean_file_rows)
clean_files_df

# %%
subject_summary_rows = []
pair_rows = []

for row in clean_files_df.itertuples(index=False):
    subject = row.subject
    clean_file = row.clean_file

    raw_rest = mne.io.read_raw_snirf(clean_file, preload=False)

    # pick all fNIRS channels, then keep only CW amplitude channels
    picks_fnirs = mne.pick_types(raw_rest.info, fnirs=True)
    channel_types = np.array(raw_rest.get_channel_types())
    picks_cw = picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]

    # distances and SS/LS masks
    dists = source_detector_distances(raw_rest.info, picks=picks_cw)

    ss_mask_all = short_channels(raw_rest.info, threshold=0.015)
    ss_mask = ss_mask_all[picks_cw]

    ls_mask = dists >= 0.025

    # channel-level names
    cw_names = np.array(raw_rest.ch_names)[picks_cw]
    pair_names = np.array([name.split(" ")[0] for name in cw_names])

    # pair-level table for this subject
    pair_table = pd.DataFrame({
        "subject": subject,
        "channel_name": cw_names,
        "pair_name": pair_names,
        "distance_m": dists,
        "is_ss": ss_mask,
        "is_ls": ls_mask,
    })

    pair_summary = (
        pair_table.groupby(["subject", "pair_name"], as_index=False)
        .agg(
            distance_m=("distance_m", "first"),
            is_ss=("is_ss", "first"),
            is_ls=("is_ls", "first"),
        )
    )

    pair_summary["group"] = np.select(
        [pair_summary["is_ss"], pair_summary["is_ls"]],
        ["SS", "LS"],
        default="MID"
    )

    pair_rows.append(pair_summary)

    group_counts = pair_summary["group"].value_counts()

    subject_summary_rows.append({
        "subject": subject,
        "file": str(clean_file),
        "sfreq": raw_rest.info["sfreq"],
        "duration_s": raw_rest.times[-1],
        "n_fnirs_channels": len(picks_fnirs),
        "n_cw_channels": len(picks_cw),
        "distance_min_m": float(dists.min()),
        "distance_max_m": float(dists.max()),
        "n_ss_channels": int(ss_mask.sum()),
        "n_ls_channels": int(ls_mask.sum()),
        "n_pairs_total": len(pair_summary),
        "n_pairs_ss": int(group_counts.get("SS", 0)),
        "n_pairs_ls": int(group_counts.get("LS", 0)),
        "n_pairs_mid": int(group_counts.get("MID", 0)),
    })

subject_summary_df = pd.DataFrame(subject_summary_rows)
pair_summary_df = pd.concat(pair_rows, ignore_index=True)

subject_summary_df

# %% [markdown]
# # Inspect semisynthetic HRF files and event annotations

# %%
output_dir = root / "outputs"
output_tables_dir = output_dir / "tables"
output_figures_dir = output_dir / "figures"

output_tables_dir.mkdir(parents=True, exist_ok=True)
output_figures_dir.mkdir(parents=True, exist_ok=True)

file_labels = [
    ("clean", "resting_clean.snirf"),
    ("hrf_20", "resting_hrf_20.snirf"),
    ("hrf_50", "resting_hrf_50.snirf"),
    ("hrf_100", "resting_hrf_100.snirf"),
]

# %%
def get_cw_channel_indices(raw_snirf):
    picks_fnirs = mne.pick_types(raw_snirf.info, fnirs=True)
    channel_types = np.array(raw_snirf.get_channel_types())
    picks_cw = picks_fnirs[channel_types[picks_fnirs] == "fnirs_cw_amplitude"]
    return picks_cw


def build_channel_table(raw_snirf, subject_name, file_label):
    picks_cw = get_cw_channel_indices(raw_snirf)
    cw_names = np.array(raw_snirf.ch_names)[picks_cw]

    dists = source_detector_distances(raw_snirf.info, picks=picks_cw)

    ss_mask_all = short_channels(raw_snirf.info, threshold=0.015)
    ss_mask = ss_mask_all[picks_cw]

    ls_mask = dists >= 0.025
    pair_names = np.array([name.split(" ")[0] for name in cw_names])

    channel_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "channel_name": cw_names,
        "pair_name": pair_names,
        "distance_m": dists,
        "is_ss": ss_mask,
        "is_ls": ls_mask,
    })

    channel_table["group"] = np.select(
        [channel_table["is_ss"], channel_table["is_ls"]],
        ["SS", "LS"],
        default="MID",
    )

    return channel_table


def build_event_table(raw_snirf, subject_name, file_label):
    annotations = raw_snirf.annotations

    event_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "onset_s": annotations.onset,
        "duration_s": annotations.duration,
        "description": annotations.description,
    })

    return event_table

# %%
subject_name = "Subj100"
subject_dir = rs_data_dir / subject_name

subject_file_rows = []
subject_event_tables = []
subject_channel_tables = []

for file_label, file_name in file_labels:
    file_path = subject_dir / file_name

    if not file_path.exists():
        print("missing:", file_path)
        continue

    raw_snirf = mne.io.read_raw_snirf(file_path, preload=False, verbose=False)
    channel_table = build_channel_table(raw_snirf, subject_name, file_label)
    event_table = build_event_table(raw_snirf, subject_name, file_label)

    subject_file_rows.append({
        "subject": subject_name,
        "file_label": file_label,
        "file_path": str(file_path),
        "sfreq": raw_snirf.info["sfreq"],
        "duration_s": raw_snirf.times[-1],
        "n_channels_total": len(raw_snirf.ch_names),
        "n_cw_channels": len(channel_table),
        "n_pairs": channel_table["pair_name"].nunique(),
        "n_ss_pairs": int((channel_table.groupby("pair_name")["group"].first() == "SS").sum()),
        "n_ls_pairs": int((channel_table.groupby("pair_name")["group"].first() == "LS").sum()),
        "n_mid_pairs": int((channel_table.groupby("pair_name")["group"].first() == "MID").sum()),
        "n_annotations": len(raw_snirf.annotations),
        "annotation_descriptions": "|".join(sorted(set(raw_snirf.annotations.description))),
    })

    subject_channel_tables.append(channel_table)
    subject_event_tables.append(event_table)

subject_file_summary_df = pd.DataFrame(subject_file_rows)
subject_channel_summary_df = pd.concat(subject_channel_tables, ignore_index=True)
subject_event_summary_df = pd.concat(subject_event_tables, ignore_index=True)

subject_file_summary_df

# %%
subject_event_summary_df

# %%
annotation_figure = px.scatter(
    subject_event_summary_df,
    x="onset_s",
    y="description",
    color="file_label",
    hover_data=["duration_s"],
    title=f"{subject_name}: annotation overview",
)

annotation_figure.show()
annotation_figure.write_html(output_figures_dir / f"{subject_name.lower()}_annotation_overview.html")

# %%
overlay_rows = []

for file_label, file_name in file_labels:
    file_path = subject_dir / file_name

    if not file_path.exists():
        continue

    raw_snirf = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
    channel_table = build_channel_table(raw_snirf, subject_name, file_label)

    long_channel_names = channel_table.loc[channel_table["group"] == "LS", "channel_name"].tolist()
    selected_channel_names = long_channel_names[:8]

    if len(selected_channel_names) == 0:
        continue

    selected_channel_indices = mne.pick_channels(raw_snirf.ch_names, selected_channel_names)
    selected_channel_data = raw_snirf.get_data(picks=selected_channel_indices)

    mean_signal = selected_channel_data.mean(axis=0)

    overlay_rows.append(pd.DataFrame({
        "time_s": raw_snirf.times,
        "signal": mean_signal,
        "file_label": file_label,
    }))

overlay_df = pd.concat(overlay_rows, ignore_index=True)
overlay_df.head()

# %%
overlay_figure = px.line(
    overlay_df,
    x="time_s",
    y="signal",
    color="file_label",
    title=f"{subject_name}: mean LS CW amplitude overlay",
)

overlay_figure.show()
overlay_figure.write_html(output_figures_dir / f"{subject_name.lower()}_ls_overlay.html")

# %%
all_file_rows = []
all_event_tables = []
all_channel_tables = []

for subject_dir in sorted(rs_data_dir.glob("Subj*")):
    if not subject_dir.is_dir():
        continue

    subject_name = subject_dir.name

    for file_label, file_name in file_labels:
        file_path = subject_dir / file_name

        if not file_path.exists():
            print("missing:", file_path)
            continue

        raw_snirf = mne.io.read_raw_snirf(file_path, preload=False, verbose=False)
        channel_table = build_channel_table(raw_snirf, subject_name, file_label)
        event_table = build_event_table(raw_snirf, subject_name, file_label)

        pair_groups = channel_table.groupby("pair_name")["group"].first()

        all_file_rows.append({
            "subject": subject_name,
            "file_label": file_label,
            "file_path": str(file_path),
            "sfreq": raw_snirf.info["sfreq"],
            "duration_s": raw_snirf.times[-1],
            "n_channels_total": len(raw_snirf.ch_names),
            "n_cw_channels": len(channel_table),
            "n_pairs": channel_table["pair_name"].nunique(),
            "n_ss_pairs": int((pair_groups == "SS").sum()),
            "n_ls_pairs": int((pair_groups == "LS").sum()),
            "n_mid_pairs": int((pair_groups == "MID").sum()),
            "n_annotations": len(raw_snirf.annotations),
            "annotation_descriptions": "|".join(sorted(set(raw_snirf.annotations.description))),
        })

        all_channel_tables.append(channel_table)
        all_event_tables.append(event_table)

all_file_summary_df = pd.DataFrame(all_file_rows)
all_channel_summary_df = pd.concat(all_channel_tables, ignore_index=True)
all_event_summary_df = pd.concat(all_event_tables, ignore_index=True)

# %%
all_file_summary_df

# %%
all_file_summary_df.groupby("file_label")[["n_annotations", "n_ss_pairs", "n_ls_pairs", "n_mid_pairs"]].mean(numeric_only=True)

# %%
all_event_summary_df.groupby(["file_label", "description"]).size().reset_index(name="count")

# %%
subject_name = "Subj100"
subject_dir = rs_data_dir / subject_name

file_label = "hrf_50"
file_name = "resting_hrf_50.snirf"
file_path = subject_dir / file_name

raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
print(raw_cw)
print(raw_cw.annotations)

# %%
raw_od = optical_density(raw_cw.copy())
raw_hb = beer_lambert_law(raw_od, ppf=0.1)

print(raw_hb)
print(raw_hb.get_channel_types()[:10])
print(raw_hb.ch_names[:10])

# %%
pd.Series(raw_hb.get_channel_types()).value_counts()

# %%
def build_hb_channel_table(raw_hb, subject_name, file_label):
    picks_hbo = mne.pick_types(raw_hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(raw_hb.info, fnirs="hbr")
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))

    hb_names = np.array(raw_hb.ch_names)[picks_hb]
    hb_types = np.array(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.array([channel_name.split(" ")[0] for channel_name in hb_names])

    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]

    ss_mask_all = short_channels(raw_hb.info, threshold=0.015)
    ss_mask = ss_mask_all[picks_hb]
    ls_mask = distances_hb >= 0.025

    hb_channel_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "channel_name": hb_names,
        "pair_name": pair_names,
        "chromophore": hb_types,
        "distance_m": distances_hb,
        "is_ss": ss_mask,
        "is_ls": ls_mask,
    })

    hb_channel_table["group"] = np.select(
        [hb_channel_table["is_ss"], hb_channel_table["is_ls"]],
        ["SS", "LS"],
        default="MID",
    )

    return hb_channel_table

# %%
hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)
hb_channel_table.head()

# %%
ls_hbo_channel_names = hb_channel_table.loc[
    (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbo"),
    "channel_name",
].tolist()

ls_hbr_channel_names = hb_channel_table.loc[
    (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbr"),
    "channel_name",
].tolist()

selected_hbo_channel_names = ls_hbo_channel_names[:8]
selected_hbr_channel_names = ls_hbr_channel_names[:8]

print("selected HbO channels:", selected_hbo_channel_names)
print("selected HbR channels:", selected_hbr_channel_names)

# %%
events, event_id = mne.events_from_annotations(raw_hb, verbose=False)

print("event_id:", event_id)
print("events shape:", events.shape)
events[:10]

# %%
epochs_hb = mne.Epochs(
    raw_hb,
    events=events,
    event_id=event_id,
    tmin=-5.0,
    tmax=20.0,
    baseline=(-2.0, 0.0),
    preload=True,
    detrend=None,
    verbose=False,
)

print(epochs_hb)

# %%
epochs_hbo = epochs_hb.copy().pick(selected_hbo_channel_names)
epochs_hbr = epochs_hb.copy().pick(selected_hbr_channel_names)

hbo_epoch_data = epochs_hbo.get_data()
hbr_epoch_data = epochs_hbr.get_data()

mean_hbo_time_course = hbo_epoch_data.mean(axis=(0, 1))
mean_hbr_time_course = hbr_epoch_data.mean(axis=(0, 1))

epoch_time = epochs_hbo.times

mean_epoch_df = pd.DataFrame({
    "time_s": np.concatenate([epoch_time, epoch_time]),
    "signal": np.concatenate([mean_hbo_time_course, mean_hbr_time_course]),
    "chromophore": ["HbO"] * len(epoch_time) + ["HbR"] * len(epoch_time),
})

mean_epoch_df.head()

# %%
mean_epoch_figure = px.line(
    mean_epoch_df,
    x="time_s",
    y="signal",
    color="chromophore",
    title=f"{subject_name} {file_label}: mean event-locked Hb response",
)

mean_epoch_figure.add_vline(x=0.0)
mean_epoch_figure.show()

# %%
amplitude_epoch_rows = []

for file_label, file_name in [("hrf_20", "resting_hrf_20.snirf"),
                              ("hrf_50", "resting_hrf_50.snirf"),
                              ("hrf_100", "resting_hrf_100.snirf")]:

    file_path = subject_dir / file_name
    raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)

    raw_od = optical_density(raw_cw.copy())
    raw_hb = beer_lambert_law(raw_od, ppf=0.1)

    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    ls_hbo_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbo"),
        "channel_name",
    ].tolist()

    ls_hbr_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbr"),
        "channel_name",
    ].tolist()

    selected_hbo_channel_names = ls_hbo_channel_names[:8]
    selected_hbr_channel_names = ls_hbr_channel_names[:8]

    events, event_id = mne.events_from_annotations(raw_hb, verbose=False)

    epochs_hb = mne.Epochs(
        raw_hb,
        events=events,
        event_id=event_id,
        tmin=-5.0,
        tmax=20.0,
        baseline=(-2.0, 0.0),
        preload=True,
        detrend=None,
        verbose=False,
    )

    hbo_epoch_data = epochs_hb.copy().pick(selected_hbo_channel_names).get_data()
    hbr_epoch_data = epochs_hb.copy().pick(selected_hbr_channel_names).get_data()

    mean_hbo_time_course = hbo_epoch_data.mean(axis=(0, 1))
    mean_hbr_time_course = hbr_epoch_data.mean(axis=(0, 1))

    amplitude_epoch_rows.append(pd.DataFrame({
        "time_s": epochs_hb.times,
        "signal": mean_hbo_time_course,
        "chromophore": "HbO",
        "file_label": file_label,
    }))

    amplitude_epoch_rows.append(pd.DataFrame({
        "time_s": epochs_hb.times,
        "signal": mean_hbr_time_course,
        "chromophore": "HbR",
        "file_label": file_label,
    }))

amplitude_epoch_df = pd.concat(amplitude_epoch_rows, ignore_index=True)
amplitude_epoch_df.head()

# %%
amplitude_epoch_figure = px.line(
    amplitude_epoch_df,
    x="time_s",
    y="signal",
    color="file_label",
    facet_row="chromophore",
    title=f"{subject_name}: mean event-locked Hb responses across amplitudes",
)

amplitude_epoch_figure.add_vline(x=0.0)
amplitude_epoch_figure.show()
amplitude_epoch_figure.write_html(output_figures_dir / f"{subject_name.lower()}_hb_epoch_amplitude_comparison.html")

# %% [markdown]
# # Rank long-separation HbO channels by event-locked effect size

# %%
subject_name = "Subj100"
subject_dir = rs_data_dir / subject_name

file_label = "hrf_100"
file_name = "resting_hrf_100.snirf"
file_path = subject_dir / file_name

raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
raw_od = optical_density(raw_cw.copy())
raw_hb = beer_lambert_law(raw_od, ppf=0.1)

events, event_id = mne.events_from_annotations(raw_hb, verbose=False)

epochs_hb = mne.Epochs(
    raw_hb,
    events=events,
    event_id=event_id,
    tmin=-5.0,
    tmax=20.0,
    baseline=None,
    preload=True,
    detrend=None,
    verbose=False,
)

print(epochs_hb)
print(event_id)

# %%
def build_hb_channel_table(raw_hb, subject_name, file_label):
    picks_hbo = mne.pick_types(raw_hb.info, fnirs="hbo")
    picks_hbr = mne.pick_types(raw_hb.info, fnirs="hbr")
    picks_hb = np.sort(np.concatenate([picks_hbo, picks_hbr]))

    hb_names = np.array(raw_hb.ch_names)[picks_hb]
    hb_types = np.array(raw_hb.get_channel_types())[picks_hb]
    pair_names = np.array([channel_name.split(" ")[0] for channel_name in hb_names])

    distances_all = source_detector_distances(raw_hb.info)
    distances_hb = distances_all[picks_hb]

    ss_mask_all = short_channels(raw_hb.info, threshold=0.015)
    ss_mask = ss_mask_all[picks_hb]
    ls_mask = distances_hb >= 0.025

    hb_channel_table = pd.DataFrame({
        "subject": subject_name,
        "file_label": file_label,
        "channel_name": hb_names,
        "pair_name": pair_names,
        "chromophore": hb_types,
        "distance_m": distances_hb,
        "is_ss": ss_mask,
        "is_ls": ls_mask,
    })

    hb_channel_table["group"] = np.select(
        [hb_channel_table["is_ss"], hb_channel_table["is_ls"]],
        ["SS", "LS"],
        default="MID",
    )

    return hb_channel_table

# %%
hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

ls_hbo_channel_names = hb_channel_table.loc[
    (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbo"),
    "channel_name",
].tolist()

print(f"Number of LS HbO channels: {len(ls_hbo_channel_names)}")
print(ls_hbo_channel_names[:10])

# %%
channel_effect_rows = []

baseline_time_mask = (epochs_hb.times >= -2.0) & (epochs_hb.times <= 0.0)
response_time_mask = (epochs_hb.times >= 4.0) & (epochs_hb.times <= 8.0)

for channel_name in ls_hbo_channel_names:
    single_channel_epochs = epochs_hb.copy().pick([channel_name])
    single_channel_data = single_channel_epochs.get_data()[:, 0, :]

    baseline_values = single_channel_data[:, baseline_time_mask].mean(axis=1)
    response_values = single_channel_data[:, response_time_mask].mean(axis=1)

    effect_values = response_values - baseline_values

    channel_effect_rows.append({
        "subject": subject_name,
        "file_label": file_label,
        "channel_name": channel_name,
        "pair_name": channel_name.split(" ")[0],
        "mean_baseline": baseline_values.mean(),
        "mean_response": response_values.mean(),
        "mean_effect_size": effect_values.mean(),
        "std_effect_size": effect_values.std(),
    })

channel_effect_df = pd.DataFrame(channel_effect_rows)
channel_effect_df = channel_effect_df.sort_values("mean_effect_size", ascending=False).reset_index(drop=True)

channel_effect_df.head(15)

# %%
top_n_channels = 15

channel_rank_figure = px.bar(
    channel_effect_df.head(top_n_channels),
    x="channel_name",
    y="mean_effect_size",
    title=f"{subject_name} {file_label}: top LS HbO channels by event-locked effect size",
)

channel_rank_figure.update_layout(
    width=1100,
    height=500,
    xaxis_tickangle=-45,
)

channel_rank_figure.show()

# %%
top_channel_names = channel_effect_df.head(6)["channel_name"].tolist()
top_pair_names = channel_effect_df.head(6)["pair_name"].tolist()

print("Top channel names:")
print(top_channel_names)

print("\nTop pair names:")
print(top_pair_names)

# %%
top_channel_plot_rows = []

for channel_name in top_channel_names:
    single_channel_epochs = epochs_hb.copy().pick([channel_name])
    single_channel_data = single_channel_epochs.get_data()[:, 0, :]

    mean_time_course = single_channel_data.mean(axis=0)

    top_channel_plot_rows.append(pd.DataFrame({
        "time_s": epochs_hb.times,
        "signal": mean_time_course,
        "channel_name": channel_name,
    }))

top_channel_plot_df = pd.concat(top_channel_plot_rows, ignore_index=True)
top_channel_plot_df.head()

# %%
top_channel_time_course_figure = px.line(
    top_channel_plot_df,
    x="time_s",
    y="signal",
    color="channel_name",
    title=f"{subject_name} {file_label}: top LS HbO channel time courses",
)

top_channel_time_course_figure.update_layout(
    width=1100,
    height=550,
)

top_channel_time_course_figure.add_vline(x=0.0)
top_channel_time_course_figure.show()

# %%
top_channel_amplitude_rows = []

for compare_file_label, compare_file_name in [
    ("hrf_20", "resting_hrf_20.snirf"),
    ("hrf_50", "resting_hrf_50.snirf"),
    ("hrf_100", "resting_hrf_100.snirf"),
]:
    compare_file_path = subject_dir / compare_file_name

    raw_cw_compare = mne.io.read_raw_snirf(compare_file_path, preload=True, verbose=False)
    raw_od_compare = optical_density(raw_cw_compare.copy())
    raw_hb_compare = beer_lambert_law(raw_od_compare, ppf=0.1)

    compare_events, compare_event_id = mne.events_from_annotations(raw_hb_compare, verbose=False)

    compare_epochs = mne.Epochs(
        raw_hb_compare,
        events=compare_events,
        event_id=compare_event_id,
        tmin=-5.0,
        tmax=20.0,
        baseline=(-2.0, 0.0),
        preload=True,
        detrend=None,
        verbose=False,
    )

    available_top_channels = [channel_name for channel_name in top_channel_names if channel_name in compare_epochs.ch_names]

    compare_data = compare_epochs.copy().pick(available_top_channels).get_data()
    mean_compare_time_course = compare_data.mean(axis=(0, 1))

    top_channel_amplitude_rows.append(pd.DataFrame({
        "time_s": compare_epochs.times,
        "signal": mean_compare_time_course,
        "file_label": compare_file_label,
    }))

top_channel_amplitude_df = pd.concat(top_channel_amplitude_rows, ignore_index=True)
top_channel_amplitude_df.head()

# %%
top_channel_amplitude_figure = px.line(
    top_channel_amplitude_df,
    x="time_s",
    y="signal",
    color="file_label",
    title=f"{subject_name}: top-channel HbO response across amplitudes",
)

top_channel_amplitude_figure.update_layout(
    width=1100,
    height=500,
)

top_channel_amplitude_figure.add_vline(x=0.0)
top_channel_amplitude_figure.show()


