# %%
from pathlib import Path
import numpy as np
import scipy.io
from scipy.interpolate import interp1d
import plotly.express as px
import pandas as pd

# %%
ROOT = Path.home() / "fnirs-representation-learning"
HRF_DIR = ROOT / "synthetic_hrf_generation"
BASE_PATH = HRF_DIR / "hrf_50.mat"

mat = scipy.io.loadmat(BASE_PATH, squeeze_me=True, struct_as_record=False)
base_hrf = mat["hrf"]

t_base = np.asarray(base_hrf.t_hrf, dtype=float).reshape(-1)
conc_base = np.asarray(base_hrf.hrf_conc, dtype=float)   # (time, 3)
d_base = np.asarray(base_hrf.hrf_d, dtype=float)         # (time, 2)
d0_base = np.asarray(base_hrf.hrf_d0, dtype=float)       # (time, 2)
dod_base = np.asarray(base_hrf.hrf_dod, dtype=float)     # (time, 2)

print(t_base.shape, conc_base.shape, d_base.shape)

# %%
def warp_and_scale_series(t, y, ttp_target_s=6.0, width_scale=1.0, amplitude_scale=1.0, base_ttp_s=6.0):
    """
    Time-warp and amplitude-scale a time series y(t).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float)

    # map target time axis back to source time axis
    # if width_scale > 1, response becomes wider/slower
    source_t = (t / width_scale) * (base_ttp_s / ttp_target_s)

    f = interp1d(
        t,
        y,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    y_warp = f(source_t)
    return amplitude_scale * y_warp

# %%
def build_custom_hrf_from_base(base_hrf, amplitude_pct, ttp_s, width_scale):
    t = np.asarray(base_hrf.t_hrf, dtype=float).reshape(-1)

    conc_base = np.asarray(base_hrf.hrf_conc, dtype=float)
    d_base = np.asarray(base_hrf.hrf_d, dtype=float)
    d0_base = np.asarray(base_hrf.hrf_d0, dtype=float)
    dod_base = np.asarray(base_hrf.hrf_dod, dtype=float)

    amplitude_scale = amplitude_pct / 50.0

    # Fields centered around 0: direct scaling is fine
    conc_new = warp_and_scale_series(
        t, conc_base,
        ttp_target_s=ttp_s,
        width_scale=width_scale,
        amplitude_scale=amplitude_scale,
        base_ttp_s=6.0,
    )

    d0_new = warp_and_scale_series(
        t, d0_base,
        ttp_target_s=ttp_s,
        width_scale=width_scale,
        amplitude_scale=amplitude_scale,
        base_ttp_s=6.0,
    )

    dod_new = warp_and_scale_series(
        t, dod_base,
        ttp_target_s=ttp_s,
        width_scale=width_scale,
        amplitude_scale=amplitude_scale,
        base_ttp_s=6.0,
    )

    # hrf_d is multiplicative around 1, so scale deviations from 1
    d_delta_base = d_base - 1.0
    d_delta_new = warp_and_scale_series(
        t, d_delta_base,
        ttp_target_s=ttp_s,
        width_scale=width_scale,
        amplitude_scale=amplitude_scale,
        base_ttp_s=6.0,
    )
    d_new = 1.0 + d_delta_new

    return {
        "hrf_SD": base_hrf.hrf_SD,
        "t_hrf": t[:, None],
        "hrf_conc": conc_new,
        "hrf_d": d_new,
        "hrf_d0": d0_new,
        "hrf_dod": dod_new,
    }

# %%
custom_specs = [
    {"label": "hrf_30_fast",      "amplitude_pct": 30, "ttp_s": 5.0, "width_scale": 0.85},
    {"label": "hrf_45_narrow",    "amplitude_pct": 45, "ttp_s": 5.7, "width_scale": 0.90},
    {"label": "hrf_60_wide",      "amplitude_pct": 60, "ttp_s": 6.5, "width_scale": 1.10},
    {"label": "hrf_75_slow",      "amplitude_pct": 75, "ttp_s": 7.2, "width_scale": 1.20},
    {"label": "hrf_85_fastwide",  "amplitude_pct": 85, "ttp_s": 5.5, "width_scale": 1.15},
    {"label": "hrf_40_slow",      "amplitude_pct": 40, "ttp_s": 6.8, "width_scale": 1.05},
]

# %%
CUSTOM_OUT = HRF_DIR / "custom_bank"
CUSTOM_OUT.mkdir(parents=True, exist_ok=True)

for spec in custom_specs:
    hrf_dict = build_custom_hrf_from_base(
        base_hrf,
        amplitude_pct=spec["amplitude_pct"],
        ttp_s=spec["ttp_s"],
        width_scale=spec["width_scale"],
    )
    out_path = CUSTOM_OUT / f"{spec['label']}.mat"
    scipy.io.savemat(out_path, {"hrf": hrf_dict})
    print("wrote", out_path)

# %%
plot_rows = []

# official anchors
for label in ["hrf_20", "hrf_50", "hrf_100"]:
    h = scipy.io.loadmat(HRF_DIR / f"{label}.mat", squeeze_me=True, struct_as_record=False)["hrf"]
    t = np.asarray(h.t_hrf, dtype=float).reshape(-1)
    c = np.asarray(h.hrf_conc, dtype=float)
    for tt, yy in zip(t, c[:, 0]):  # HbO only
        plot_rows.append({"time_s": tt, "value": yy, "label": label, "family": "official"})

# customs
for spec in custom_specs:
    h = scipy.io.loadmat(CUSTOM_OUT / f"{spec['label']}.mat", squeeze_me=True, struct_as_record=False)["hrf"]
    t = np.asarray(h.t_hrf, dtype=float).reshape(-1)
    c = np.asarray(h.hrf_conc, dtype=float)
    for tt, yy in zip(t, c[:, 0]):
        plot_rows.append({"time_s": tt, "value": yy, "label": spec["label"], "family": "custom"})

plot_df = pd.DataFrame(plot_rows)

fig = px.line(
    plot_df,
    x="time_s",
    y="value",
    color="label",
    line_dash="family",
    title="Official + custom HRF bank (HbO)",
)

fig.update_layout(height=800, width=1000)
fig.show()

# %%
for spec in custom_specs:
    h = scipy.io.loadmat(CUSTOM_OUT / f"{spec['label']}.mat", squeeze_me=True, struct_as_record=False)["hrf"]
    d = np.asarray(h.hrf_d, dtype=float)
    print(spec["label"], d.min(), d.max())

# results need to be close to 1 (~0.99 - 1.01)

# %%
plot_rows = []

for label in ["hrf_20", "hrf_50", "hrf_100"]:
    h = scipy.io.loadmat(HRF_DIR / f"{label}.mat", squeeze_me=True, struct_as_record=False)["hrf"]
    t = np.asarray(h.t_hrf, dtype=float).reshape(-1)
    d = np.asarray(h.hrf_d, dtype=float)
    for tt, yy in zip(t, d[:, 0]):
        plot_rows.append({"time_s": tt, "value": yy, "label": label, "family": "official"})

for spec in custom_specs:
    h = scipy.io.loadmat(CUSTOM_OUT / f"{spec['label']}.mat", squeeze_me=True, struct_as_record=False)["hrf"]
    t = np.asarray(h.t_hrf, dtype=float).reshape(-1)
    d = np.asarray(h.hrf_d, dtype=float)
    for tt, yy in zip(t, d[:, 0]):
        plot_rows.append({"time_s": tt, "value": yy, "label": spec["label"], "family": "custom"})

plot_df = pd.DataFrame(plot_rows)

fig = px.line(
    plot_df,
    x="time_s",
    y="value",
    color="label",
    line_dash="family",
    title="Official + custom HRF bank (hrf_d, wavelength 1)",
)
fig.show()


