# %% [markdown]
# # Includes

# %%
import pandas as pd
import numpy as np
import mne
import mne_nirs
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index,
    source_detector_distances,
    short_channels,
)
from mne.preprocessing.nirs import temporal_derivative_distribution_repair as tddr

# %%
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

epoch_tmin = -5.0
epoch_tmax = 20.0
baseline_window = (-2.0, 0.0)

response_window = (4.0, 8.0)

# %% [markdown]
# # Helper functions

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

    ss_mask_all = short_channels(raw_hb.info, threshold=short_separation_threshold_m)
    ss_mask = ss_mask_all[picks_hb]
    ls_mask = distances_hb >= long_separation_threshold_m

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
        verbose=False,
    )

    return epochs_hb, events, event_id

# %%
def compute_channel_effect_table(epochs_hb, raw_hb, subject_name, file_label, pipeline_label):
    hb_channel_table = build_hb_channel_table(raw_hb, subject_name, file_label)

    ls_hbo_channel_names = hb_channel_table.loc[
        (hb_channel_table["group"] == "LS") & (hb_channel_table["chromophore"] == "hbo"),
        "channel_name",
    ].tolist()

    baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
    response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])

    channel_effect_rows = []

    for channel_name in ls_hbo_channel_names:
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

# %%
def build_mean_epoch_plot_table(epochs_hb, selected_channel_names, pipeline_label):
    available_channel_names = [channel_name for channel_name in selected_channel_names if channel_name in epochs_hb.ch_names]

    epochs_hbo = epochs_hb.copy().pick([channel_name for channel_name in available_channel_names if channel_name.endswith("hbo")])
    epochs_hbr = epochs_hb.copy().pick([channel_name for channel_name in available_channel_names if channel_name.endswith("hbr")])

    plot_rows = []

    if len(epochs_hbo.ch_names) > 0:
        hbo_data = epochs_hbo.get_data()
        mean_hbo_time_course = hbo_data.mean(axis=(0, 1))

        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hbo.times,
            "signal": mean_hbo_time_course,
            "chromophore": "HbO",
            "pipeline_label": pipeline_label,
        }))

    if len(epochs_hbr.ch_names) > 0:
        hbr_data = epochs_hbr.get_data()
        mean_hbr_time_course = hbr_data.mean(axis=(0, 1))

        plot_rows.append(pd.DataFrame({
            "time_s": epochs_hbr.times,
            "signal": mean_hbr_time_course,
            "chromophore": "HbR",
            "pipeline_label": pipeline_label,
        }))

    mean_epoch_plot_table = pd.concat(plot_rows, ignore_index=True)
    return mean_epoch_plot_table

# %% [markdown]
# # Raw data exploration - starting with one subject and one file

# %%
raw_cw = mne.io.read_raw_snirf(file_path, preload=True, verbose=False)
raw_cw

# %%
raw_cw.annotations

# %% [markdown]
# ## Pipeline runner function

# %%
def run_pipeline(raw_cw, pipeline_label):
    raw_od = optical_density(raw_cw.copy())

    sci_values = scalp_coupling_index(raw_od)
    sci_table = pd.DataFrame({
        "channel_name": raw_od.ch_names,
        "sci": sci_values,
    })

    if pipeline_label == "Pref":
        raw_processed_od = tddr(raw_od.copy())
        raw_hb = beer_lambert_law(raw_processed_od, ppf=0.1)

    elif pipeline_label == "PnoMC":
        raw_hb = beer_lambert_law(raw_od.copy(), ppf=0.1)

    elif pipeline_label == "Pminimal":
        raw_hb = beer_lambert_law(raw_od.copy(), ppf=0.1)

    else:
        raise ValueError(f"Unknown pipeline_label: {pipeline_label}")

    return {
        "pipeline_label": pipeline_label,
        "raw_od": raw_od,
        "raw_hb": raw_hb,
        "sci_table": sci_table,
    }

# %%
def run_pipeline(raw_cw, pipeline_label):
    raw_od = optical_density(raw_cw.copy())

    sci_values = scalp_coupling_index(raw_od)
    sci_table = pd.DataFrame({
        "channel_name": raw_od.ch_names,
        "sci": sci_values,
    })

    if pipeline_label == "Pref":
        raw_processed_od = tddr(raw_od.copy())
        raw_hb = beer_lambert_law(raw_processed_od, ppf=0.1)
        raw_hb = raw_hb.copy().filter(0.01, 0.20, verbose=False)

    elif pipeline_label == "PnoMC":
        raw_hb = beer_lambert_law(raw_od.copy(), ppf=0.1)
        raw_hb = raw_hb.copy().filter(0.01, 0.20, verbose=False)

    elif pipeline_label == "PnoFilt":
        raw_processed_od = tddr(raw_od.copy())
        raw_hb = beer_lambert_law(raw_processed_od, ppf=0.1)

    else:
        raise ValueError(f"Unknown pipeline_label: {pipeline_label}")

    return {
        "pipeline_label": pipeline_label,
        "raw_od": raw_od,
        "raw_hb": raw_hb,
        "sci_table": sci_table,
    }

# %% [markdown]
# ## Run preprocessing pipelines

# %%
pipeline_labels = ["Pref", "PnoMC", "PnoFilt"]

pipeline_results = []

for pipeline_label in pipeline_labels:
    pipeline_result = run_pipeline(raw_cw, pipeline_label)
    pipeline_results.append(pipeline_result)

len(pipeline_results)

# %%
sci_summary_rows = []

for pipeline_result in pipeline_results:
    pipeline_label = pipeline_result["pipeline_label"]
    sci_table = pipeline_result["sci_table"]

    sci_summary_rows.append({
        "pipeline_label": pipeline_label,
        "n_channels": len(sci_table),
        "mean_sci": sci_table["sci"].mean(),
        "median_sci": sci_table["sci"].median(),
        "min_sci": sci_table["sci"].min(),
        "max_sci": sci_table["sci"].max(),
        "n_sci_below_0_5": int((sci_table["sci"] < 0.5).sum()),
    })

sci_summary_df = pd.DataFrame(sci_summary_rows)
sci_summary_df

# %% [markdown]
# ## Epoch each pipeline output

# %%
epoch_results = []

for pipeline_result in pipeline_results:
    pipeline_label = pipeline_result["pipeline_label"]
    raw_hb = pipeline_result["raw_hb"]

    epochs_hb, events, event_id = make_epochs_from_raw_hb(raw_hb)

    epoch_results.append({
        "pipeline_label": pipeline_label,
        "raw_hb": raw_hb,
        "epochs_hb": epochs_hb,
        "events": events,
        "event_id": event_id,
    })

len(epoch_results)

# %% [markdown]
# ## Select comparison channels from the reference pipeline

# %%
reference_result = [result for result in epoch_results if result["pipeline_label"] == "Pref"][0]

reference_channel_effect_df = compute_channel_effect_table(
    epochs_hb=reference_result["epochs_hb"],
    raw_hb=reference_result["raw_hb"],
    subject_name=subject_name,
    file_label=file_label,
    pipeline_label="Pref",
)

reference_channel_effect_df.head(10)

# %%
top_hbo_channel_names = reference_channel_effect_df.head(6)["channel_name"].tolist()
top_pair_names = [channel_name.replace(" hbo", "") for channel_name in top_hbo_channel_names]
top_hbr_channel_names = [pair_name + " hbr" for pair_name in top_pair_names]

selected_channel_names = top_hbo_channel_names + top_hbr_channel_names

selected_channel_names

# %% [markdown]
# ## Compare event-locked Hb responses across pipelines

# %%
pipeline_plot_tables = []

for epoch_result in epoch_results:
    pipeline_label = epoch_result["pipeline_label"]
    epochs_hb = epoch_result["epochs_hb"]

    mean_epoch_plot_table = build_mean_epoch_plot_table(
        epochs_hb=epochs_hb,
        selected_channel_names=selected_channel_names,
        pipeline_label=pipeline_label,
    )

    pipeline_plot_tables.append(mean_epoch_plot_table)

pipeline_plot_df = pd.concat(pipeline_plot_tables, ignore_index=True)
pipeline_plot_df.head()

# %%
pipeline_comparison_figure = px.line(
    pipeline_plot_df,
    x="time_s",
    y="signal",
    color="pipeline_label",
    facet_row="chromophore",
    title=f"{subject_name} {file_label}: event-locked Hb responses across preprocessing pipelines",
)

pipeline_comparison_figure.update_layout(
    width=1100,
    height=650,
)

pipeline_comparison_figure.add_vline(x=0.0)
pipeline_comparison_figure.show()
pipeline_comparison_figure.write_html(
    output_figures_dir / f"{subject_name.lower()}_{file_label}_pipeline_comparison.html"
)

# %%
# quanitify repsonse size per pipeline
pipeline_metric_rows = []

for epoch_result in epoch_results:
    pipeline_label = epoch_result["pipeline_label"]
    epochs_hb = epoch_result["epochs_hb"]

    available_hbo_channel_names = [channel_name for channel_name in top_hbo_channel_names if channel_name in epochs_hb.ch_names]
    available_hbr_channel_names = [channel_name for channel_name in top_hbr_channel_names if channel_name in epochs_hb.ch_names]

    hbo_data = epochs_hb.copy().pick(available_hbo_channel_names).get_data()
    hbr_data = epochs_hb.copy().pick(available_hbr_channel_names).get_data()

    baseline_time_mask = (epochs_hb.times >= baseline_window[0]) & (epochs_hb.times <= baseline_window[1])
    response_time_mask = (epochs_hb.times >= response_window[0]) & (epochs_hb.times <= response_window[1])

    hbo_baseline = hbo_data[:, :, baseline_time_mask].mean()
    hbo_response = hbo_data[:, :, response_time_mask].mean()
    hbr_baseline = hbr_data[:, :, baseline_time_mask].mean()
    hbr_response = hbr_data[:, :, response_time_mask].mean()

    pipeline_metric_rows.append({
        "subject": subject_name,
        "file_label": file_label,
        "pipeline_label": pipeline_label,
        "hbo_mean_effect": hbo_response - hbo_baseline,
        "hbr_mean_effect": hbr_response - hbr_baseline,
        "n_events": len(epochs_hb),
        "n_hbo_channels": len(available_hbo_channel_names),
        "n_hbr_channels": len(available_hbr_channel_names),
    })

pipeline_metric_df = pd.DataFrame(pipeline_metric_rows)
pipeline_metric_df

# %%
"""
pipeline_metric_df.to_csv(
    output_tables_dir / f"{subject_name.lower()}_{file_label}_pipeline_metric_summary.csv",
    index=False,
)

reference_channel_effect_df.to_csv(
    output_tables_dir / f"{subject_name.lower()}_{file_label}_reference_channel_effects.csv",
    index=False,
)

print("Saved Notebook 02 outputs.")
"""


