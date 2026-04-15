# %% [markdown]
# # fNIRS benchmark analysis notebook
# 
# Cleaned notebook-style analysis script for the semisynthetic benchmark.

import ast
import importlib.util
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io
import scipy.integrate
import scipy.stats as stats

HAVE_KALEIDO = importlib.util.find_spec("kaleido") is not None
HAVE_STATSMODELS = importlib.util.find_spec("statsmodels") is not None
HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAVE_IPYTHON = importlib.util.find_spec("IPython") is not None

if HAVE_STATSMODELS:
    from statsmodels.stats.multitest import multipletests
if HAVE_SKLEARN:
    from sklearn.metrics import average_precision_score, roc_auc_score
if HAVE_IPYTHON:
    from IPython.display import display
else:
    def display(x):
        if hasattr(x, "to_string"):
            print(x.to_string())
        else:
            print(x)

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 220)
warnings.filterwarnings("ignore")


# %% [markdown]
# ## Analysis settings

# %%
PROJECT_ROOT = Path(
    os.environ.get("FNIRS_PROJECT_ROOT", Path.home() / "fnirs-representation-learning")
).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("FNIRS_OUTPUT_DIR", PROJECT_ROOT)
).expanduser().resolve()
AGG_DIR = OUTPUT_DIR / "aggregate"
TRUTH_TEMPLATE_DIR = Path(
    os.environ.get("FNIRS_TRUTH_TEMPLATE_DIR", PROJECT_ROOT / "truth_templates")
).expanduser().resolve()

FIG_DIR = OUTPUT_DIR / "analysis_figures_clean"
EXPORT_DIR = OUTPUT_DIR / "analysis_exports_clean"
FIG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_CHROM = "hbo"
PRIMARY_SCORE_TYPE = "canonical_beta"
REFERENCE_PIPELINE = "LocalSS_Glover_AUTO"

FLEX_PIPELINES = [
    "LocalSS_SPM_Derivs_OLS",
    "LocalSS_Gamma_Derivs_OLS",
    "Homer3_tCCA_Gaussian",
    "Homer3_tCCA_ModGamma",
]
EXCLUDED_PIPELINES = [
    "Homer3_tCCA_ModGamma",
    "NoSS_FIR_OLS_NoTDDR",
    "LocalSS_Gamma_Derivs_ARIRLS",
    "LocalSS_SPM_Derivs_ARIRLS",
]

OFFICIAL_ACTIVE_FILES = ["resting_hrf_20", "resting_hrf_50", "resting_hrf_100"]

DEFAULT_EPOCH_TMIN = -5.0
DEFAULT_EPOCH_TMAX = 30.0
DEFAULT_BASELINE_WINDOW = (-5.0, 0.0)
DEFAULT_RESPONSE_WINDOW = (4.0, 8.0)

TRUTH_SUPPORT_THRESHOLD = 0.25
PEAK_WINDOW_HALF_WIDTH_S = 2.0

MIN_PAIRWISE_SUBJECTS = 5
MIN_COMPLETE_CASE_SUBJECTS = 5
MIN_EMPIRICAL_NULL_SHIFTS = 1000
MIN_ACTIVE_NULL_FILES_PER_CLASS = 2

SAVE_FIGURES = True
SHOW_FIGURES = True
SAVE_HTML_TOO = False
SAVE_SVG_TOO = False
INCLUDE_SUPPORTING_FIGURES = True
INCLUDE_PEAK_RATIO_DIAGNOSTICS = True

FIG_WIDTH = 1400
FIG_HEIGHT = 1300
FIG_SCALE = 4
FIG_BASE_FONT_SIZE = 24
FIG_TITLE_FONT_SIZE = 28
FIG_AXIS_FONT_SIZE = 22
FIG_TICK_FONT_SIZE = 20
FIG_LEGEND_FONT_SIZE = 18

CONCENTRATION_UNIT_LABEL = "M HbO"
REPRESENTATIVE_SUBJECT = os.environ.get("FNIRS_REPRESENTATIVE_SUBJECT") or None
REPRESENTATIVE_FILE_LABEL = os.environ.get("FNIRS_REPRESENTATIVE_FILE_LABEL", "resting_hrf_50")
SECOND_REPRESENTATIVE_SUBJECT = os.environ.get("FNIRS_SECOND_REPRESENTATIVE_SUBJECT") or None
SECOND_REPRESENTATIVE_FILE_LABEL = os.environ.get("FNIRS_SECOND_REPRESENTATIVE_FILE_LABEL") or None
PLOT_ALL_OFFICIAL_OVERLAYS = os.environ.get("FNIRS_PLOT_ALL_OFFICIAL_OVERLAYS", "0") == "1"

PUBLICATION_COLORWAY = [
    "#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
    "#6A3D9A", "#8C564B", "#1B9E77", "#7570B3", "#E7298A", "#66A61E", "#A6761D",
]

PIPELINE_DISPLAY_NAMES = {
    "BlockAvg_LocalSS": "Block average + local SS",
    "HighPassOnly_LocalSS_Glover_AUTO": "High-pass-only Glover AUTO",
    "Homer3_tCCA_Gaussian": "Homer3 tCCA Gaussian",
    "LocalSSFilter_Canonical_ARIRLS": "Short-separation filter canonical AR-IRLS",
    "LocalSSReg_CanonicalDerivs_ARIRLS": "Short-separation regressors canonical + derivatives AR-IRLS",
    "LocalSSReg_Canonical_ARIRLS": "Short-separation regressors canonical AR-IRLS",
    "LocalSS_Gamma_Derivs_OLS": "Local SS gamma + derivatives OLS",
    "LocalSS_Glover_AUTO": "Local SS Glover AUTO",
    "LocalSS_Glover_OLS": "Local SS Glover OLS",
    "LocalSS_SPM_AUTO": "Local SS SPM AUTO",
    "LocalSS_SPM_Derivs_OLS": "Local SS SPM + derivatives OLS",
    "LocalSS_SPM_OLS": "Local SS SPM OLS",
    "LooseQC_LocalSS_Glover_AUTO": "Loose-QC local SS Glover AUTO",
    "MultiSSOrth3_Glover_AUTO": "Multi-short-separation orthogonalized Glover AUTO",
    "NoMotion_LocalSS_Glover_AUTO": "No motion correction local SS Glover AUTO",
    "NoSS_Canonical_ARIRLS": "No SS canonical AR-IRLS",
    "NoSS_Canonical_ARIRLS_NoTDDR": "No SS canonical AR-IRLS without TDDR",
    "NoSS_Canonical_OLS": "No SS canonical OLS",
    "NoSS_Canonical_OLS_NoTDDR": "No SS canonical OLS without TDDR",
    "LocalSS_FIR_AUTO": "Local SS FIR AUTO",
    "NoSS_FIR_OLS_NoTDDR": "No SS FIR OLS, no TDDR",
    "NoSS_Glover_AUTO": "No SS Glover AUTO",
    "NoSS_Glover_OLS": "No SS Glover OLS",
    "PooledPCA2_Glover_AUTO": "Pooled PCA Glover AUTO",
    "SSAuxPCA_Glover_AUTO": "Short-separation + auxiliary PCA Glover AUTO",
    "WaveletMC_LocalSS_Glover_AUTO": "Wavelet motion-corrected Glover AUTO",
}
SCORE_TYPE_DISPLAY_NAMES = {
    "shape_peak_amplitude": "Recovered ROI peak amplitude",
    "canonical_beta": "Canonical beta",
    "block_average_score": "Block-average response",
    "roi_peak_amplitude_bc": "Recovered ROI peak amplitude",
    "roi_mean_response_bc": "Mean ROI response",
    "roi_response_auc_bc": "ROI response AUC",
}
ROI_DISPLAY_NAMES = {
    "dlpfc_bilateral": "Bilateral DLPFC",
    "dlpfc_left": "Left DLPFC",
    "dlpfc_right": "Right DLPFC",
}

OVERLAY_HIGHLIGHT_PIPELINES = [
    "Homer3_tCCA_Gaussian",
    "BlockAvg_LocalSS",
    "LocalSS_Glover_AUTO",
    "LocalSS_Glover_OLS",
    "LocalSSReg_Canonical_ARIRLS",
    "LocalSSReg_CanonicalDerivs_ARIRLS",
    "LocalSSFilter_Canonical_ARIRLS",
    "NoSS_Canonical_ARIRLS",
    "LocalSS_FIR_AUTO",
]
OVERLAY_HIGHLIGHT_COLORS = {
    "Homer3_tCCA_Gaussian": "#009E73",
    "BlockAvg_LocalSS": "#6C5CE7",
    "LocalSS_Glover_AUTO": "#0072B2",
    "LocalSS_Glover_OLS": "#56B4E9",
    "LocalSSReg_Canonical_ARIRLS": "#D55E00",
    "LocalSSReg_CanonicalDerivs_ARIRLS": "#CC79A7",
    "LocalSSFilter_Canonical_ARIRLS": "#E69F00",
    "NoSS_Canonical_ARIRLS": "#6A3D9A",
    "LocalSS_FIR_AUTO": "#8C564B",
}

REQUIRED_TABLES_FOR_FULL_ANALYSIS = [
    "pipeline_manifest",
    "channel_availability",
    "canonical_channel_metrics",
    "roi_timecourses",
    "dlpfc_roi_scores",
    "target_vs_nontarget_summary",
    "truth_summary",
]

READ_ENGINE = os.environ.get("FNIRS_READ_ENGINE", "pyarrow")
TABLE_COLUMN_HINTS = {
    "pipeline_manifest": ["pipeline_label", "pipeline_order", "backend", "hrf_model", "nuisance_method", "solver", "secondary_pipeline", "comparison_group", "description"],
    "truth_summary": None,
    "channel_availability": ["subject", "file_label", "pipeline_label", "amplitude_value", "target_pair_retention_fraction", "non_target_pair_retention_fraction", "n_bad_pairs"],
    "canonical_channel_metrics": ["subject", "file_label", "pipeline_label", "amplitude_value", "chromophore", "target_status", "beta", "p_value", "q_value_bh"],
    "block_average_channel_metrics": ["subject", "file_label", "pipeline_label", "amplitude_value", "chromophore", "target_status", "score"],
    "roi_timecourses": ["subject", "file_label", "amplitude_value", "pipeline_label", "backend", "chromophore", "target_status", "time_s", "signal", "curve_source"],
    "dlpfc_roi_scores": ["subject", "file_label", "pipeline_label", "amplitude_value", "chromophore", "score_type", "roi_name", "roi_mean_score"],
    "target_vs_nontarget_summary": ["subject", "file_label", "pipeline_label", "amplitude_value", "chromophore", "score_type", "target_minus_non_target_score"],
    "parametric_null_summary": ["subject", "pipeline_label", "target_status", "chromophore", "false_positive_rate_p_lt_0_05", "false_positive_rate_q_lt_0_05"],
    "empirical_null_pvalues": ["subject", "file_label", "pipeline_label", "score_type", "chromophore", "empirical_p_value", "n_null_shifts"],
    "variability_summary": ["subject", "file_label", "subset_name", "score_type", "chromophore", "active_vs_null", "std_across_pipelines"],
    "pairwise_pipeline_deltas": ["subject", "left_pipeline", "right_pipeline", "score_type", "chromophore", "abs_left_minus_right"],
    "homer3_glm_metadata": ["pipeline_label", "idx_basis", "stim_duration_s_effective", "ppf_json"],
    "config": None,
}

PRIMARY_RECOVERY_REQUIRE_COMPLETE_ANCHORS = True
PRIMARY_DOSE_RESPONSE_REQUIRE_COMPLETE_ANCHORS = True
DLPFC_BRIDGE_SCORE_TYPE = "shape_peak_amplitude"
ALLOW_DLPFC_BRIDGE_FALLBACK = True
DLPFC_SCORE_TYPE_PREFERENCE = [DLPFC_BRIDGE_SCORE_TYPE, "canonical_beta", "block_average_score"]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("AGG_DIR:", AGG_DIR)
print("TRUTH_TEMPLATE_DIR:", TRUTH_TEMPLATE_DIR)
print("EXPORT_DIR:", EXPORT_DIR)
print("FIG_DIR:", FIG_DIR)

if not AGG_DIR.exists():
    raise FileNotFoundError(
        f"Aggregate directory not found: {AGG_DIR}\n"
        "Set FNIRS_OUTPUT_DIR to the benchmark output directory that contains an aggregate folder."
    )

# %% [markdown]
#      ## 1) Tiny helpers

# %%
def parse_scalar_maybe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        if s.lower() in {"none", "nan"}:
            return None
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return ast.literal_eval(s)
            except Exception:
                try:
                    return json.loads(s)
                except Exception:
                    return value
        try:
            if "." in s or "e" in s.lower():
                return float(s)
            return int(s)
        except Exception:
            return value
    return value


def coerce_interval(value, default):
    parsed = parse_scalar_maybe(value)
    if parsed is None:
        return tuple(default)
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        try:
            return (float(parsed[0]), float(parsed[1]))
        except Exception:
            return tuple(default)
    return tuple(default)


def config_lookup(config_df: pd.DataFrame, key: str, default=None):
    if config_df is None or len(config_df) == 0:
        return default
    if key not in config_df.columns:
        return default
    return parse_scalar_maybe(config_df.iloc[0][key])


def to_numeric_if_present(df: pd.DataFrame, columns):
    if df is None or len(df) == 0:
        return df
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_target_status(x):
    x = str(x)
    if x in {"true_target", "target", "1"}:
        return "true_target"
    return "true_non_target"


def apply_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    df = df.copy()
    if "pipeline_label" in df.columns:
        df["pipeline_label"] = df["pipeline_label"].astype(str)
        df["pipeline_display"] = df["pipeline_label"].map(pretty_pipeline_label)
    if "score_type" in df.columns:
        df["score_type_display"] = df["score_type"].astype(str).map(pretty_score_type)
    if "roi_name" in df.columns:
        df["roi_display"] = df["roi_name"].astype(str).map(pretty_roi_name)
    return df


def file_family_from_label(file_label):
    file_label = str(file_label)
    if file_label == "resting_clean":
        return "null_clean"
    if file_label.startswith("resting_clean_surrogate_"):
        return "null_surrogate"
    if file_label in OFFICIAL_ACTIVE_FILES:
        return "official_active"
    if file_label.startswith("orig_hrf_"):
        return "custom_active_orig"
    if file_label.startswith("sur") and "_hrf_" in file_label:
        return "custom_active_surrogate"
    return "other"


def active_vs_null_label(amplitude_value):
    if pd.isna(amplitude_value):
        return "unknown"
    return "null" if float(amplitude_value) == 0 else "active"


def official_anchor_flag(file_label):
    return str(file_label) in OFFICIAL_ACTIVE_FILES


def orient_score(series, chrom):
    if chrom == "hbo":
        return series.astype(float)
    if chrom == "hbr":
        return -series.astype(float)
    return series.astype(float)


def ordered_pipeline_list(pipeline_manifest, fallback_frames=None):
    ordered = []
    seen = set()
    if len(pipeline_manifest) > 0 and {"pipeline_label", "pipeline_order"}.issubset(pipeline_manifest.columns):
        tmp = pipeline_manifest[["pipeline_label", "pipeline_order"]].drop_duplicates().sort_values(["pipeline_order", "pipeline_label"])
        for label in tmp["pipeline_label"].astype(str).tolist():
            if label not in seen:
                ordered.append(label)
                seen.add(label)
    elif len(pipeline_manifest) > 0 and "pipeline_label" in pipeline_manifest.columns:
        for label in pipeline_manifest["pipeline_label"].astype(str).drop_duplicates().tolist():
            if label not in seen:
                ordered.append(label)
                seen.add(label)

    fallback_labels = set()
    for df in fallback_frames or []:
        if len(df) > 0 and "pipeline_label" in df.columns:
            fallback_labels.update(df["pipeline_label"].dropna().astype(str).unique().tolist())

    for label in sorted(fallback_labels):
        if label not in seen:
            ordered.append(label)
            seen.add(label)
    return ordered


def save_table(df: pd.DataFrame, name: str) -> None:
    if df is None or len(df) == 0:
        return
    df.to_csv(EXPORT_DIR / f"{name}.csv", index=False)


def make_subject_level_boxplot(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    out_stem: str,
    points: str = "all",
):
    if df is None or len(df) == 0:
        return None
    plot_df = apply_display_labels(df)
    cat_order = ordered_pipeline_display_labels(plot_df["pipeline_label"].astype(str).unique().tolist())
    fig = px.box(
        plot_df,
        x="pipeline_display",
        y=value_col,
        points=points,
        title=title,
        labels={"pipeline_display": "Pipeline", value_col: y_label},
        category_orders={"pipeline_display": cat_order},
    )
    fig.update_traces(
        marker=dict(size=5, opacity=0.75),
        line=dict(width=1.4),
        whiskerwidth=0.7,
    )
    fig.update_xaxes(tickangle=55)
    style_figure(fig, title=title)
    if SHOW_FIGURES:
        fig.show()
    save_fig(fig, out_stem)
    return fig


def style_figure(fig, title: str | None = None):
    clean_title = publication_title(title) if title is not None else None
    if clean_title is not None:
        fig.update_layout(title=clean_title)
    fig.update_layout(
        template="plotly_white",
        font=dict(size=FIG_BASE_FONT_SIZE, family="Arial"),
        title=dict(font=dict(size=FIG_TITLE_FONT_SIZE), x=0.5, xanchor="center"),
        legend=dict(font=dict(size=FIG_LEGEND_FONT_SIZE), orientation="v", bgcolor="rgba(255,255,255,0.80)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1),
        margin=dict(l=95, r=40, t=110, b=135),
        width=FIG_WIDTH,
        height=FIG_HEIGHT,
        paper_bgcolor="white",
        plot_bgcolor="white",
        colorway=PUBLICATION_COLORWAY,
    )
    fig.update_xaxes(
        title_font=dict(size=FIG_AXIS_FONT_SIZE),
        tickfont=dict(size=FIG_TICK_FONT_SIZE),
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        ticklen=6,
        tickwidth=1.2,
        mirror=False,
        automargin=True,
        showgrid=False,
    )
    fig.update_yaxes(
        title_font=dict(size=FIG_AXIS_FONT_SIZE),
        tickfont=dict(size=FIG_TICK_FONT_SIZE),
        showline=True,
        linewidth=1.5,
        linecolor="black",
        ticks="outside",
        ticklen=6,
        tickwidth=1.2,
        mirror=False,
        zeroline=True,
        zerolinewidth=1,
        gridcolor="rgba(0,0,0,0.12)",
        gridwidth=1,
        automargin=True,
        exponentformat="power",
        showexponent="all",
    )
    return fig


def significance_text(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_reference_significance_annotations(fig, summary: pd.DataFrame, pairwise: pd.DataFrame, reference_pipeline: str | None, y_col: str = "value_mean"):
    if fig is None or len(summary) == 0 or pairwise is None or len(pairwise) == 0 or not reference_pipeline:
        return fig
    y_vals = pd.to_numeric(summary[y_col], errors="coerce")
    if y_vals.notna().sum() == 0:
        return fig
    spread = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if len(y_vals) > 1 else float(np.nanmax(np.abs(y_vals)))
    offset = 0.09 * (spread if np.isfinite(spread) and spread > 0 else max(1.0, float(np.nanmax(np.abs(y_vals)))))
    if "value_ci_hi" in summary.columns:
        tops = {
            str(row["pipeline_label"]): float(row["value_ci_hi"]) if np.isfinite(pd.to_numeric(row["value_ci_hi"], errors="coerce")) else float(row[y_col])
            for _, row in summary.iterrows()
        }
    else:
        tops = {str(row["pipeline_label"]): float(row[y_col]) for _, row in summary.iterrows()}
    levels = {}
    max_annot_y = None
    for _, row in pairwise.iterrows():
        if not bool(row.get("eligible_for_wilcoxon", False)):
            continue
        p = row.get("p_value_holm", row.get("p_value", np.nan))
        star = significance_text(p)
        if star in {"", "ns"}:
            continue
        comp = str(row["comparison_pipeline"])
        if comp not in tops:
            continue
        levels[comp] = levels.get(comp, 0) + 1
        y_here = tops[comp] + offset * levels[comp]
        max_annot_y = y_here if max_annot_y is None else max(max_annot_y, y_here)
        fig.add_annotation(
            x=pretty_pipeline_label(comp),
            y=y_here,
            text=f"<b>{star}</b>",
            showarrow=False,
            yanchor="bottom",
            yshift=8,
            font=dict(size=22, color="black"),
        )
    if max_annot_y is not None:
        fig.update_yaxes(range=[None, max_annot_y + offset * 0.75])
    return fig


def save_fig(fig, name: str) -> None:
    style_figure(fig)
    if not SAVE_FIGURES:
        return
    if HAVE_KALEIDO:
        png_path = FIG_DIR / f"{name}.png"
        fig.write_image(str(png_path), width=FIG_WIDTH, height=FIG_HEIGHT, scale=FIG_SCALE)
        if SAVE_SVG_TOO:
            svg_path = FIG_DIR / f"{name}.svg"
            fig.write_image(str(svg_path), width=FIG_WIDTH, height=FIG_HEIGHT)
    if SAVE_HTML_TOO or not HAVE_KALEIDO:
        fig.write_html(str(FIG_DIR / f"{name}.html"))


def in_window(times_s, window):
    times_s = np.asarray(times_s, dtype=float)
    start_s, end_s = window
    return (times_s >= float(start_s)) & (times_s <= float(end_s))


def baseline_correct(signal, time_s, baseline_window):
    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    mask = in_window(time_s, baseline_window) & np.isfinite(signal)
    if np.any(mask):
        base = float(np.nanmean(signal[mask]))
    else:
        base = 0.0
    return signal - base, base


def peak_value_and_time(signal, time_s, chromophore):
    signal = np.asarray(signal, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    finite = np.isfinite(signal) & np.isfinite(time_s)
    if finite.sum() == 0:
        return (np.nan, np.nan)
    signal = signal[finite]
    time_s = time_s[finite]
    if chromophore == "hbo":
        idx = int(np.nanargmax(signal))
    else:
        idx = int(np.nanargmin(signal))
    return float(signal[idx]), float(time_s[idx])


def bootstrap_mean_ci(values, n_boot: int = 2000, alpha: float = 0.05, random_state: int = 7):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    boot = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(np.mean(sample))
    lo = np.quantile(boot, alpha / 2)
    hi = np.quantile(boot, 1 - alpha / 2)
    return (float(np.mean(values)), float(lo), float(hi))


def summarize_with_bootstrap(df: pd.DataFrame, value_col: str, group_cols):
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean_, lo_, hi_ = bootstrap_mean_ci(grp[value_col].to_numpy())
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                f"{value_col}_mean": mean_,
                f"{value_col}_ci_lo": lo_,
                f"{value_col}_ci_hi": hi_,
                "n_rows": int(len(grp)),
                "n_subjects": int(grp["subject"].nunique()) if "subject" in grp.columns else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def try_complete_case_pivot(df: pd.DataFrame, index_col: str = "subject", column_col: str = "pipeline_label", value_col: str = "value"):
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    return pivot


def kendalls_w_from_rank_matrix(rank_matrix):
    rank_matrix = np.asarray(rank_matrix, dtype=float)
    m, n = rank_matrix.shape
    col_sums = rank_matrix.sum(axis=0)
    s = np.sum((col_sums - np.mean(col_sums)) ** 2)
    return float(12 * s / (m ** 2 * (n ** 3 - n))) if m > 0 and n > 1 else np.nan


def paired_rank_biserial(diff):
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]
    if len(diff) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(diff))
    pos = float(np.sum(ranks[diff > 0]))
    neg = float(np.sum(ranks[diff < 0]))
    denom = pos + neg
    return float((pos - neg) / denom) if denom > 0 else np.nan


def repeated_measures_suite(df: pd.DataFrame, metric_name: str, reference_pipeline: str | None = None):
    """
    Omnibus Friedman is based on strict complete cases across the full set of pipelines.
    Pairwise comparisons are computed on pairwise-overlap subjects, but inferential
    Wilcoxon tests are only reported once the minimum paired-sample threshold is met.
    Descriptive paired effect summaries are still exported below that threshold.
    """
    omnibus_rows = []
    pairwise_rows = []

    pivot_full = try_complete_case_pivot(df, value_col="value")

    if len(pivot_full) >= MIN_COMPLETE_CASE_SUBJECTS and pivot_full.shape[1] >= 3:
        fried = stats.friedmanchisquare(*[pivot_full[col].values for col in pivot_full.columns])
        w = fried.statistic / (len(pivot_full) * (pivot_full.shape[1] - 1))
        omnibus_rows.append(
            {
                "metric_name": metric_name,
                "test": "friedman_complete_case",
                "n_subjects_complete_case": int(len(pivot_full)),
                "n_pipelines": int(pivot_full.shape[1]),
                "statistic": float(fried.statistic),
                "p_value": float(fried.pvalue),
                "kendalls_w": float(w),
            }
        )

    if reference_pipeline is not None and reference_pipeline in df["pipeline_label"].astype(str).unique():
        raw_pvals = []
        for pipeline in sorted(df["pipeline_label"].astype(str).unique()):
            if pipeline == reference_pipeline:
                continue
            pair = df[df["pipeline_label"].isin([reference_pipeline, pipeline])].pivot_table(
                index="subject",
                columns="pipeline_label",
                values="value",
                aggfunc="mean",
            )
            needed = [reference_pipeline, pipeline]
            if not set(needed).issubset(pair.columns):
                continue
            pair = pair[needed].dropna()
            n_pair = int(len(pair))
            if n_pair < 2:
                continue
            diff = pair[pipeline] - pair[reference_pipeline]
            rbc = paired_rank_biserial(diff)
            eligible_for_wilcoxon = n_pair >= MIN_PAIRWISE_SUBJECTS
            if not eligible_for_wilcoxon:
                stat = np.nan
                p = np.nan
            elif np.allclose(diff.to_numpy(dtype=float), 0.0, equal_nan=False):
                stat = 0.0
                p = 1.0
            else:
                try:
                    test = stats.wilcoxon(diff)
                    stat = float(test.statistic)
                    p = float(test.pvalue)
                except Exception:
                    stat = np.nan
                    p = np.nan
            row = {
                "metric_name": metric_name,
                "reference_pipeline": reference_pipeline,
                "comparison_pipeline": pipeline,
                "n_subjects_pairwise": n_pair,
                "eligible_for_wilcoxon": bool(eligible_for_wilcoxon),
                "median_difference": float(np.nanmedian(diff)),
                "mean_difference": float(np.nanmean(diff)),
                "paired_rank_biserial": rbc,
                "wilcoxon_statistic": stat,
                "p_value": p,
            }
            pairwise_rows.append(row)
            raw_pvals.append(p)

        if HAVE_STATSMODELS and pairwise_rows:
            corrected = [np.nan] * len(pairwise_rows)
            valid_p = np.asarray([p for p in raw_pvals if np.isfinite(p)], dtype=float)
            if len(valid_p) > 0:
                _, p_corr, _, _ = multipletests(valid_p, method="holm")
                j = 0
                for i, row in enumerate(pairwise_rows):
                    if np.isfinite(raw_pvals[i]):
                        corrected[i] = float(p_corr[j])
                        j += 1
                for row, corr in zip(pairwise_rows, corrected):
                    row["p_value_holm"] = corr

    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows), pivot_full


def pipeline_bar_from_subject_level(df: pd.DataFrame, value_col: str, title: str, y_label: str, name: str, ascending: bool = False, pairwise: pd.DataFrame | None = None, reference_pipeline: str | None = None):
    if len(df) == 0:
        return pd.DataFrame()
    summary = summarize_with_bootstrap(df.rename(columns={value_col: "value"}), "value", ["pipeline_label"])
    summary["pipeline_display"] = summary["pipeline_label"].map(pretty_pipeline_label)
    summary = summary.sort_values("value_mean", ascending=ascending)
    save_table(summary, f"{name}_summary")
    fig = px.bar(
        summary,
        x="pipeline_display",
        y="value_mean",
        error_y=summary["value_ci_hi"] - summary["value_mean"],
        error_y_minus=summary["value_mean"] - summary["value_ci_lo"],
        title=title,
        labels={"pipeline_display": "Pipeline", "value_mean": unit_label(y_label)},
    )
    fig.update_traces(marker_color="#4C78A8", marker_line_color="black", marker_line_width=0.8)
    fig.update_xaxes(categoryorder="array", categoryarray=summary["pipeline_display"].astype(str).tolist(), tickangle=42)
    fig = add_reference_significance_annotations(fig, summary, pairwise, reference_pipeline, y_col="value_mean")
    if "AUROC" in title or "AUROC" in y_label:
        fig.update_yaxes(range=[0.0, 1.0])
    if "retention fraction" in y_label.lower():
        fig.update_yaxes(range=[0.0, 1.0])
    style_figure(fig, title=title)
    if SHOW_FIGURES:
        fig.show()
    save_fig(fig, name)
    return summary


def pipeline_pointrange_from_subject_level(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    y_label: str,
    name: str,
    ascending: bool = False,
    pairwise: pd.DataFrame | None = None,
    reference_pipeline: str | None = None,
):
    if len(df) == 0:
        return pd.DataFrame()
    summary = summarize_with_bootstrap(df.rename(columns={value_col: "value"}), "value", ["pipeline_label"])
    summary["pipeline_display"] = summary["pipeline_label"].map(pretty_pipeline_label)
    summary = summary.sort_values("value_mean", ascending=ascending)
    save_table(summary, f"{name}_summary")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary["pipeline_display"].astype(str),
            y=summary["value_mean"],
            mode="markers",
            marker=dict(size=11, color="#0072B2", line=dict(color="black", width=0.9)),
            error_y=dict(
                type="data",
                symmetric=False,
                array=(summary["value_ci_hi"] - summary["value_mean"]).to_numpy(),
                arrayminus=(summary["value_mean"] - summary["value_ci_lo"]).to_numpy(),
                thickness=1.6,
                width=0,
                color="black",
            ),
            showlegend=False,
            hovertemplate="%{x}<br>Mean=%{y:.4g}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Pipeline",
        yaxis_title=unit_label(y_label),
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=summary["pipeline_display"].astype(str).tolist(),
        tickangle=42,
    )
    fig = add_reference_significance_annotations(fig, summary, pairwise, reference_pipeline, y_col="value_mean")
    if "AUROC" in title or "AUROC" in y_label:
        fig.update_yaxes(range=[0.0, 1.0])
    if "retention fraction" in y_label.lower():
        fig.update_yaxes(range=[0.0, 1.0])
    style_figure(fig, title=title)
    if SHOW_FIGURES:
        fig.show()
    save_fig(fig, name)
    return summary


def run_metric_suite(
    subject_df: pd.DataFrame,
    value_col: str,
    metric_name: str,
    reference_pipeline: str | None,
    summary_name: str,
    figure_title: str,
    figure_ylabel: str,
    figure_stem: str,
    ascending: bool = False,
    plot_kind: str = "bar",
):
    if len(subject_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    omnibus, pairwise, pivot = repeated_measures_suite(
        subject_df.rename(columns={value_col: "value"}),
        metric_name=metric_name,
        reference_pipeline=reference_pipeline,
    )
    if str(plot_kind).lower() == "pointrange":
        summary = pipeline_pointrange_from_subject_level(
            subject_df, value_col, figure_title, figure_ylabel, figure_stem,
            ascending=ascending, pairwise=pairwise, reference_pipeline=reference_pipeline
        )
    else:
        summary = pipeline_bar_from_subject_level(
            subject_df, value_col, figure_title, figure_ylabel, figure_stem,
            ascending=ascending, pairwise=pairwise, reference_pipeline=reference_pipeline
        )
    save_table(omnibus, f"{summary_name}_omnibus")
    save_table(pairwise, f"{summary_name}_pairwise")
    return summary, omnibus, pairwise, pivot


def infer_pipeline_fields(label: str, backend: str | None = None, hrf_model: str | None = None, nuisance_method: str | None = None):
    label_str = str(label)
    lower = label_str.lower()

    nuisance = nuisance_method if nuisance_method not in [None, "", np.nan] else None
    if nuisance is None:
        if label_str.startswith("Homer3_tCCA"):
            nuisance = "tCCA"
        elif "LocalSS" in label_str:
            nuisance = "LocalSS"
        elif "NoSS" in label_str:
            nuisance = "NoSS"
        elif "LooseQC" in label_str:
            nuisance = "LooseQC + LocalSS"
        elif "WaveletMC" in label_str:
            nuisance = "WaveletMC + LocalSS"
        elif "HighPassOnly" in label_str:
            nuisance = "High-pass only + LocalSS"
        elif "NoMotion" in label_str:
            nuisance = "No motion correction + LocalSS"
        elif "PooledPCA" in label_str:
            nuisance = "Pooled PCA"
        elif "SSAuxPCA" in label_str:
            nuisance = "SSAuxPCA"
        elif "MultiSSOrth" in label_str:
            nuisance = "MultiSS orthogonalization"
        elif "BlockAvg" in label_str:
            nuisance = "LocalSS"
        else:
            nuisance = "Other"

    model = hrf_model if hrf_model not in [None, "", np.nan] else None
    if model is None:
        if "modgamma" in lower:
            model = "Modified gamma basis"
        elif "gaussian" in lower:
            model = "Gaussian basis"
        elif "fir" in lower:
            model = "FIR"
        elif "spm" in lower and "deriv" in lower:
            model = "SPM + derivatives"
        elif "gamma" in lower and "deriv" in lower:
            model = "Gamma + derivatives"
        elif "spm" in lower:
            model = "SPM"
        elif "gamma" in lower:
            model = "Gamma"
        elif "glover" in lower:
            model = "Glover"
        elif "blockavg" in lower:
            model = "Block average"
        else:
            model = "Other"

    solver = backend if backend not in [None, "", np.nan] else None
    if solver is None:
        if "arirls" in lower:
            solver = "AR-IRLS"
        elif "ols" in lower:
            solver = "OLS"
        elif "auto" in lower:
            solver = "AUTO"
        elif label_str.startswith("Homer3_"):
            solver = "Homer3 native GLM"
        elif label_str.startswith("BlockAvg"):
            solver = "Block average"
        elif "fir" in lower:
            solver = "AUTO"
        else:
            solver = "Other"

    family = "flex" if label_str in FLEX_PIPELINES else "standard"
    return nuisance, model, solver, family


def template_key_from_file_label(file_label):
    file_label = str(file_label)
    if file_label.startswith("resting_hrf_"):
        return file_label.replace("resting_", "")
    m = re.search(r"(hrf_\d+(?:_[A-Za-z0-9]+)*)$", file_label)
    return m.group(1) if m else None


def load_truth_curve(file_label, chrom="hbo"):
    key = template_key_from_file_label(file_label)
    if key is None:
        return None
    mat_path = TRUTH_TEMPLATE_DIR / f"{key}.mat"
    if not mat_path.exists():
        return None
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    hrf = mat["hrf"]
    t = np.asarray(hrf.t_hrf, dtype=float).reshape(-1)
    conc = np.asarray(hrf.hrf_conc, dtype=float)
    col = {"hbo": 0, "hbr": 1, "hbt": 2}[chrom]
    y = conc[:, col].reshape(-1)
    return pd.DataFrame({"time_s": t, "truth_signal": y})


_truth_window_cache = {}


def derive_truth_windows(file_label, chrom, baseline_window, epoch_tmax, default_response_window):
    cache_key = (str(file_label), str(chrom), tuple(baseline_window), float(epoch_tmax), tuple(default_response_window))
    if cache_key in _truth_window_cache:
        return _truth_window_cache[cache_key]

    truth = load_truth_curve(file_label, chrom=chrom)
    poststim_window = (0.0, float(epoch_tmax))
    if truth is None or len(truth) == 0:
        out = {
            "has_truth": False,
            "baseline_window": tuple(baseline_window),
            "poststim_window": poststim_window,
            "support_window": tuple(default_response_window),
            "peak_window": tuple(default_response_window),
            "truth_peak_time_s": np.nan,
        }
        _truth_window_cache[cache_key] = out
        return out

    t = truth["time_s"].to_numpy(dtype=float)
    y = truth["truth_signal"].to_numpy(dtype=float)
    y_bc, _ = baseline_correct(y, t, baseline_window)

    post_mask = in_window(t, poststim_window) & np.isfinite(y_bc)
    if post_mask.sum() == 0:
        out = {
            "has_truth": True,
            "baseline_window": tuple(baseline_window),
            "poststim_window": poststim_window,
            "support_window": tuple(default_response_window),
            "peak_window": tuple(default_response_window),
            "truth_peak_time_s": np.nan,
        }
        _truth_window_cache[cache_key] = out
        return out

    peak_value, peak_time = peak_value_and_time(y_bc[post_mask], t[post_mask], chrom)
    abs_peak = abs(peak_value)
    if abs_peak > 0:
        support_mask = post_mask & (np.abs(y_bc) >= TRUTH_SUPPORT_THRESHOLD * abs_peak)
    else:
        support_mask = np.zeros_like(post_mask, dtype=bool)

    if np.any(support_mask):
        support_start = float(t[support_mask][0])
        support_end = float(t[support_mask][-1])
        support_window = (support_start, support_end)
    else:
        support_window = tuple(default_response_window)

    peak_window = (
        max(poststim_window[0], float(peak_time) - PEAK_WINDOW_HALF_WIDTH_S),
        min(poststim_window[1], float(peak_time) + PEAK_WINDOW_HALF_WIDTH_S),
    )

    out = {
        "has_truth": True,
        "baseline_window": tuple(baseline_window),
        "poststim_window": poststim_window,
        "support_window": support_window,
        "peak_window": peak_window,
        "truth_peak_time_s": float(peak_time),
    }
    _truth_window_cache[cache_key] = out
    return out


def compute_curve_metrics(recovered_curve, truth_curve, epoch_times_s, chromophore, baseline_window, eval_window, peak_window, auc_window):
    recovered_curve = np.asarray(recovered_curve, dtype=float)
    truth_curve = np.asarray(truth_curve, dtype=float)
    epoch_times_s = np.asarray(epoch_times_s, dtype=float)

    rec_bc, rec_base = baseline_correct(recovered_curve, epoch_times_s, baseline_window)
    tru_bc, tru_base = baseline_correct(truth_curve, epoch_times_s, baseline_window)

    out = {
        "curve_corr": np.nan,
        "curve_rmse": np.nan,
        "curve_nrmse": np.nan,
        "peak_latency_error_s": np.nan,
        "peak_amplitude_bias": np.nan,
        "peak_amplitude_ratio": np.nan,
        "auc_bias": np.nan,
        "recovered_peak_amplitude": np.nan,
        "recovered_auc": np.nan,
        "truth_peak_amplitude": np.nan,
        "truth_auc": np.nan,
        "recovered_baseline_mean": float(rec_base),
        "truth_baseline_mean": float(tru_base),
    }

    eval_mask = in_window(epoch_times_s, eval_window) & np.isfinite(rec_bc) & np.isfinite(tru_bc)
    if eval_mask.sum() < 3:
        return out

    rec_eval = rec_bc[eval_mask]
    tru_eval = tru_bc[eval_mask]
    t_eval = epoch_times_s[eval_mask]

    if np.std(rec_eval) >= 1e-12 and np.std(tru_eval) >= 1e-12:
        out["curve_corr"] = float(np.corrcoef(rec_eval, tru_eval)[0, 1])
    out["curve_rmse"] = float(np.sqrt(np.mean((rec_eval - tru_eval) ** 2)))
    scale = float(np.max(tru_eval) - np.min(tru_eval))
    out["curve_nrmse"] = float(out["curve_rmse"] / scale) if scale > 1e-12 else np.nan

    peak_mask = in_window(epoch_times_s, peak_window) & np.isfinite(rec_bc) & np.isfinite(tru_bc)
    if peak_mask.sum() < 1:
        peak_mask = eval_mask

    rec_peak, rec_t = peak_value_and_time(rec_bc[peak_mask], epoch_times_s[peak_mask], chromophore)
    tru_peak, tru_t = peak_value_and_time(tru_bc[peak_mask], epoch_times_s[peak_mask], chromophore)
    out["peak_latency_error_s"] = float(rec_t - tru_t)
    out["peak_amplitude_bias"] = float(rec_peak - tru_peak)
    out["peak_amplitude_ratio"] = float(rec_peak / tru_peak) if np.isfinite(tru_peak) and abs(tru_peak) > 1e-12 else np.nan
    out["recovered_peak_amplitude"] = float(rec_peak)
    out["truth_peak_amplitude"] = float(tru_peak)

    auc_mask = in_window(epoch_times_s, auc_window) & np.isfinite(rec_bc) & np.isfinite(tru_bc)
    if auc_mask.sum() < 2:
        auc_mask = eval_mask
    rec_auc = float(scipy.integrate.trapezoid(rec_bc[auc_mask], epoch_times_s[auc_mask]))
    tru_auc = float(scipy.integrate.trapezoid(tru_bc[auc_mask], epoch_times_s[auc_mask]))
    out["recovered_auc"] = rec_auc
    out["truth_auc"] = tru_auc
    out["auc_bias"] = float(rec_auc - tru_auc)
    return out


def summarize_target_minus_non_target_any(df: pd.DataFrame, value_column: str, group_columns=None, score_type: str | None = None):
    if df is None or len(df) == 0 or value_column not in df.columns:
        return pd.DataFrame()
    group_columns = list(group_columns or ["chromophore"])
    grouped = df.groupby([*group_columns, "target_status"], as_index=False).agg(
        mean_score=(value_column, "mean"),
        n_rows=(value_column, "count"),
    )
    wide = grouped.pivot_table(index=group_columns, columns="target_status", values="mean_score").reset_index()
    wide = wide.rename(columns={"true_target": "mean_target_score", "true_non_target": "mean_non_target_score"})
    counts = grouped.groupby(group_columns, as_index=False).agg(n_rows=("n_rows", "sum"))
    wide = wide.merge(counts, on=group_columns, how="left")
    if "mean_target_score" not in wide.columns:
        wide["mean_target_score"] = np.nan
    if "mean_non_target_score" not in wide.columns:
        wide["mean_non_target_score"] = np.nan
    wide["target_minus_non_target_score"] = wide["mean_target_score"] - wide["mean_non_target_score"]
    if score_type is not None:
        wide["score_type"] = score_type
    return wide


def amplitude_response_table(df: pd.DataFrame, use_official_only: bool = False, chrom: str = PRIMARY_CHROM, score_type: str = "roi_peak_amplitude_bc", require_complete_anchor_set: bool = True):
    if len(df) == 0:
        return pd.DataFrame()

    sub = df[(df["active_vs_null"] == "active") & (df["chromophore"] == chrom) & (df["score_type"] == score_type)].copy()
    if use_official_only:
        sub = sub[sub["is_official_anchor"] == True].copy()

    rows = []
    expected_anchor_set = set(OFFICIAL_ACTIVE_FILES)

    for (subject, pipeline_label), grp in sub.groupby(["subject", "pipeline_label"]):
        grp = grp.copy()
        if use_official_only and require_complete_anchor_set:
            present_anchor_set = set(grp["file_label"].astype(str).unique())
            if present_anchor_set != expected_anchor_set:
                continue
        grp = grp.groupby("amplitude_value", as_index=False).agg(
            target_minus_non_target_score=("target_minus_non_target_score", "mean"),
            n_files=("file_label", "nunique"),
        ).sort_values("amplitude_value")
        x = grp["amplitude_value"].astype(float).to_numpy()
        y = grp["target_minus_non_target_score"].astype(float).to_numpy()
        if len(np.unique(x)) < 2:
            continue
        lr = stats.linregress(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        rows.append(
            {
                "subject": subject,
                "pipeline_label": pipeline_label,
                "n_points": int(len(grp)),
                "slope": float(lr.slope),
                "intercept": float(lr.intercept),
                "r_value": float(lr.rvalue),
                "r_squared": float(lr.rvalue ** 2),
                "p_value": float(lr.pvalue),
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "spearman_p_value": float(rho_p) if np.isfinite(rho_p) else np.nan,
                "chromophore": chrom,
                "score_type": score_type,
                "analysis_set": "official_anchors" if use_official_only else "all_active_files",
            }
        )
    return pd.DataFrame(rows)


def restrict_to_complete_official_anchor_sets(df: pd.DataFrame):
    if df is None or len(df) == 0:
        return pd.DataFrame()
    required = set(OFFICIAL_ACTIVE_FILES)
    keep_keys = []
    for (subject, pipeline_label), grp in df.groupby(["subject", "pipeline_label"]):
        present = set(grp["file_label"].astype(str).unique())
        if present == required:
            keep_keys.append((subject, pipeline_label))
    if not keep_keys:
        return df.iloc[0:0].copy()
    keep_df = pd.DataFrame(keep_keys, columns=["subject", "pipeline_label"])
    return df.merge(keep_df, on=["subject", "pipeline_label"], how="inner")


def active_null_discrimination_table(df: pd.DataFrame, chrom: str = PRIMARY_CHROM, score_type: str = "roi_peak_amplitude_bc", use_official_active_only: bool = True):
    if (not HAVE_SKLEARN) or df is None or len(df) == 0 or score_type not in df.get("score_type", pd.Series(dtype=str)).astype(str).unique():
        return pd.DataFrame()

    sub = df[(df["chromophore"] == chrom) & (df["score_type"] == score_type)].copy()
    if len(sub) == 0:
        return pd.DataFrame()

    if use_official_active_only:
        sub = sub[(sub["active_vs_null"] == "null") | (sub["is_official_anchor"] == True)].copy()

    rows = []
    for (subject, pipeline_label), grp in sub.groupby(["subject", "pipeline_label"]):
        grp = grp.copy()
        grp["y_true"] = (grp["active_vs_null"] == "active").astype(int)
        class_counts = grp["y_true"].value_counts().to_dict()
        n_null = int(class_counts.get(0, 0))
        n_active = int(class_counts.get(1, 0))
        if n_null < MIN_ACTIVE_NULL_FILES_PER_CLASS or n_active < MIN_ACTIVE_NULL_FILES_PER_CLASS:
            continue
        y_true = grp["y_true"].to_numpy(dtype=int)
        scores = grp["target_minus_non_target_score"].to_numpy(dtype=float)
        if np.all(~np.isfinite(scores)):
            continue
        try:
            auroc = float(roc_auc_score(y_true, scores))
            auprc = float(average_precision_score(y_true, scores))
        except Exception:
            continue
        rows.append({
            "subject": subject,
            "pipeline_label": pipeline_label,
            "chromophore": chrom,
            "score_type": score_type,
            "n_active_files": n_active,
            "n_null_files": n_null,
            "auroc": auroc,
            "auprc": auprc,
            "analysis_set": "official_active_plus_null" if use_official_active_only else "all_active_plus_null",
        })
    return pd.DataFrame(rows)


def _normalize_curve(y):
    y = np.asarray(y, dtype=float)
    denom = np.nanmax(np.abs(y))
    if not np.isfinite(denom) or denom <= 0:
        return y
    return y / denom


def load_roi_timecourses_for_subject_file(roi_timecourses: pd.DataFrame, subject: str, file_label: str):
    if len(roi_timecourses) == 0:
        return pd.DataFrame()
    return roi_timecourses.loc[
        (roi_timecourses["subject"].astype(str) == str(subject)) &
        (roi_timecourses["file_label"].astype(str) == str(file_label))
    ].copy()


def choose_representative_subject_file(roi_timecourses: pd.DataFrame, pipelines, preferred_file="resting_hrf_50", chrom: str = PRIMARY_CHROM):
    if len(roi_timecourses) == 0:
        return (None, None)
    candidate = roi_timecourses[
        (roi_timecourses["chromophore"].astype(str).str.lower() == chrom.lower()) &
        (roi_timecourses["target_status"].astype(str) == "true_target")
    ].copy()
    if len(candidate) == 0:
        return (None, None)
    if preferred_file is not None:
        preferred = candidate[candidate["file_label"].astype(str) == str(preferred_file)].copy()
        if len(preferred) > 0:
            candidate = preferred
    counts = candidate.groupby(["subject", "file_label"], as_index=False).agg(
        n_pipelines=("pipeline_label", "nunique")
    )
    counts = counts.sort_values(["n_pipelines", "subject", "file_label"], ascending=[False, True, True])
    if len(counts) == 0:
        return (None, None)
    row = counts.iloc[0]
    return (str(row["subject"]), str(row["file_label"]))


def plot_roi_recovery(roi_timecourses: pd.DataFrame, subject: str, file_label: str, pipelines, chrom: str = "hbo"):
    sub = load_roi_timecourses_for_subject_file(roi_timecourses, subject, file_label)
    if len(sub) == 0:
        raise ValueError(f"No roi_timecourses found for {subject} / {file_label}")

    sub = sub[
        (sub["chromophore"].astype(str).str.lower() == chrom.lower()) &
        (sub["target_status"].astype(str) == "true_target")
    ].copy()
    if len(sub) == 0:
        raise ValueError("No true_target roi_timecourses rows found for that subject/file/chrom")

    truth = load_truth_curve(file_label, chrom=chrom)
    if truth is None:
        raise ValueError(f"Truth template not found for {file_label}")

    truth_bc, _ = baseline_correct(
        truth["truth_signal"].to_numpy(dtype=float),
        truth["time_s"].to_numpy(dtype=float),
        BASELINE_WINDOW,
    )

    available_pipelines = [p for p in pipelines if p in set(sub["pipeline_label"].astype(str).unique())]
    highlight = [p for p in OVERLAY_HIGHLIGHT_PIPELINES if p in available_pipelines]
    others = [p for p in available_pipelines if p not in highlight]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=truth["time_s"],
            y=truth_bc,
            mode="lines",
            name="Truth target HRF",
            line=dict(color="black", width=4),
        )
    )

    for pipeline_label in others:
        target = sub[sub["pipeline_label"].astype(str) == str(pipeline_label)].sort_values("time_s")
        if len(target) == 0:
            continue
        y_bc, _ = baseline_correct(target["signal"].to_numpy(dtype=float), target["time_s"].to_numpy(dtype=float), BASELINE_WINDOW)
        fig.add_trace(
            go.Scatter(
                x=target["time_s"],
                y=y_bc,
                mode="lines",
                name=pretty_pipeline_label(pipeline_label),
                line=dict(color="rgba(140,140,140,0.45)", width=1.2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for pipeline_label in highlight:
        target = sub[sub["pipeline_label"].astype(str) == str(pipeline_label)].sort_values("time_s")
        if len(target) == 0:
            continue
        y_bc, _ = baseline_correct(target["signal"].to_numpy(dtype=float), target["time_s"].to_numpy(dtype=float), BASELINE_WINDOW)
        fig.add_trace(
            go.Scatter(
                x=target["time_s"],
                y=y_bc,
                mode="lines",
                name=pretty_pipeline_label(pipeline_label),
                line=dict(color=OVERLAY_HIGHLIGHT_COLORS.get(pipeline_label, "#0072B2"), width=3.2),
            )
        )

    chrom_label = "HbO" if str(chrom).lower() == "hbo" else ("HbR" if str(chrom).lower() == "hbr" else str(chrom))
    fig.update_layout(
        title=f"Representative ROI recovery: {subject}, {file_label.replace('resting_', '').replace('_', '-').upper()}, {chrom_label}",
        xaxis_title="Time (s)",
        yaxis_title=f"Baseline-corrected Δ{chrom_label} concentration change (M {chrom_label})",
        template="plotly_white",
        width=1550,
        height=900,
        legend=dict(orientation="v"),

    )
    fig.update_yaxes(exponentformat="power", showexponent="all")
    style_figure(fig, title=f"Representative ROI recovery: {subject}, {file_label.replace('resting_', '').replace('_', '-').upper()}, {chrom_label}")
    fig.update_layout(margin=dict(l=95, r=260, t=120, b=90))
    return fig










# %% [markdown]
#      ## 2) Load aggregate tables

# %%
pipeline_manifest = read_table(AGG_DIR / "pipeline_manifest", TABLE_COLUMN_HINTS["pipeline_manifest"])
master_run_log = read_table(AGG_DIR / "master_run_log", TABLE_COLUMN_HINTS["master_run_log"])
truth_summary = read_table(AGG_DIR / "truth_summary", TABLE_COLUMN_HINTS["truth_summary"])
channel_quality = read_table(AGG_DIR / "channel_quality", TABLE_COLUMN_HINTS["channel_quality"])
pair_quality = read_table(AGG_DIR / "pair_quality", TABLE_COLUMN_HINTS["pair_quality"])
channel_availability = read_table(AGG_DIR / "channel_availability", TABLE_COLUMN_HINTS["channel_availability"])
canonical = read_table(AGG_DIR / "canonical_channel_metrics", TABLE_COLUMN_HINTS["canonical_channel_metrics"])
block_avg = read_table(AGG_DIR / "block_average_channel_metrics", TABLE_COLUMN_HINTS["block_average_channel_metrics"])
fir = read_table(AGG_DIR / "fir_channel_metrics", TABLE_COLUMN_HINTS["fir_channel_metrics"])
shape = read_table(AGG_DIR / "shape_fidelity", TABLE_COLUMN_HINTS["shape_fidelity"])
shape_fidelity_summary = read_table(AGG_DIR / "shape_fidelity_summary", TABLE_COLUMN_HINTS["shape_fidelity_summary"])
roi_timecourses = read_table(AGG_DIR / "roi_timecourses", TABLE_COLUMN_HINTS["roi_timecourses"])
roi_scores = read_table(AGG_DIR / "roi_scores", TABLE_COLUMN_HINTS["roi_scores"])
dlpfc_roi_scores = read_table(AGG_DIR / "dlpfc_roi_scores", TABLE_COLUMN_HINTS["dlpfc_roi_scores"])
target_vs_nontarget = read_table(AGG_DIR / "target_vs_nontarget_summary", TABLE_COLUMN_HINTS["target_vs_nontarget_summary"])
parametric_null_summary = read_table(AGG_DIR / "parametric_null_summary", TABLE_COLUMN_HINTS["parametric_null_summary"])
empirical_null_shift = read_table(AGG_DIR / "empirical_null_shift", TABLE_COLUMN_HINTS["empirical_null_shift"])
empirical_null_pvalues = read_table(AGG_DIR / "empirical_null_pvalues", TABLE_COLUMN_HINTS["empirical_null_pvalues"])
variability_summary = read_table(AGG_DIR / "variability_summary", TABLE_COLUMN_HINTS["variability_summary"])
pairwise_pipeline_deltas = read_table(AGG_DIR / "pairwise_pipeline_deltas", TABLE_COLUMN_HINTS["pairwise_pipeline_deltas"])
nuisance_detail = read_table(AGG_DIR / "nuisance_detail", TABLE_COLUMN_HINTS["nuisance_detail"])
homer3_channel_hrf = read_table(AGG_DIR / "homer3_channel_hrf", TABLE_COLUMN_HINTS["homer3_channel_hrf"])
homer3_beta = read_table(AGG_DIR / "homer3_beta", TABLE_COLUMN_HINTS["homer3_beta"])
homer3_glm_metadata = read_table(AGG_DIR / "homer3_glm_metadata", TABLE_COLUMN_HINTS["homer3_glm_metadata"])
pipeline_performance_summary = read_table(AGG_DIR / "pipeline_performance_summary", TABLE_COLUMN_HINTS["pipeline_performance_summary"])
config_df = read_table(AGG_DIR / "config", TABLE_COLUMN_HINTS["config"])
error_log = read_table(AGG_DIR / "error_log", TABLE_COLUMN_HINTS["error_log"])

# Downcast large numeric ROI arrays to save memory without changing analysis logic.
for _df_name in ["roi_timecourses", "canonical", "block_avg", "fir", "roi_scores", "dlpfc_roi_scores", "target_vs_nontarget", "parametric_null_summary", "empirical_null_pvalues", "variability_summary", "pairwise_pipeline_deltas"]:
    _df = globals().get(_df_name)
    if isinstance(_df, pd.DataFrame) and len(_df) > 0:
        for _col in [c for c in ["amplitude_value", "time_s", "signal", "beta", "score", "target_minus_non_target_score", "roi_mean_score", "roi_std_score", "false_positive_rate_p_lt_0_05", "false_positive_rate_q_lt_0_05", "empirical_p_value", "n_null_shifts", "std_across_pipelines", "abs_left_minus_right"] if c in _df.columns]:
            _df[_col] = pd.to_numeric(_df[_col], errors="coerce", downcast="float")
        for _col in [c for c in ["subject", "file_label", "pipeline_label", "chromophore", "target_status", "score_type", "roi_name", "backend", "curve_source"] if c in _df.columns]:
            if str(_df[_col].dtype) in {"object", "string"}:
                _df[_col] = _df[_col].astype("category")
missing_required_tables = []
for table_stem in REQUIRED_TABLES_FOR_FULL_ANALYSIS:
    df_name = {
        "pipeline_manifest": "pipeline_manifest",
        "channel_availability": "channel_availability",
        "canonical_channel_metrics": "canonical",
        "roi_timecourses": "roi_timecourses",
        "roi_scores": "roi_scores",
        "dlpfc_roi_scores": "dlpfc_roi_scores",
        "target_vs_nontarget_summary": "target_vs_nontarget",
        "truth_summary": "truth_summary",
    }[table_stem]
    if len(globals().get(df_name, pd.DataFrame())) == 0:
        missing_required_tables.append(table_stem)

if missing_required_tables:
    missing_required_df = pd.DataFrame({"missing_required_table": missing_required_tables})
    display(missing_required_df)
    save_table(missing_required_df, "missing_required_tables_v8")
    print("WARNING: Full thesis analysis is not structurally complete until these tables are present.")

for df_name in [
    "truth_summary", "channel_quality", "pair_quality", "channel_availability", "canonical",
    "block_avg", "fir", "shape", "shape_fidelity_summary", "roi_timecourses", "roi_scores",
    "dlpfc_roi_scores", "target_vs_nontarget", "parametric_null_summary",
    "empirical_null_shift", "empirical_null_pvalues", "variability_summary",
    "pairwise_pipeline_deltas", "homer3_channel_hrf", "homer3_beta", "homer3_glm_metadata",
    "pipeline_performance_summary",
]:
    df = globals().get(df_name)
    if isinstance(df, pd.DataFrame) and len(df) > 0:
        to_numeric_if_present(df, [
            "amplitude_value", "time_s", "signal", "beta", "beta_value", "score",
            "curve_corr", "curve_rmse", "curve_nrmse", "peak_latency_error_s",
            "peak_amplitude_bias", "peak_amplitude_ratio", "auc_bias",
            "recovered_peak_amplitude", "recovered_auc", "truth_peak_amplitude", "truth_auc",
            "empirical_p_value", "n_null_shifts", "p_value", "q_value_bh",
            "target_pair_retention_fraction", "non_target_pair_retention_fraction",
            "false_positive_rate_p_lt_0_05", "false_positive_rate_q_lt_0_05",
            "roi_mean_score", "roi_std_score",
        ])
        if "target_status" in df.columns:
            df["target_status"] = df["target_status"].map(standardize_target_status)
        if "file_label" in df.columns:
            df["file_family"] = df["file_label"].map(file_family_from_label)
            df["is_official_anchor"] = df["file_label"].map(official_anchor_flag)
        if "amplitude_value" in df.columns:
            df["active_vs_null"] = df["amplitude_value"].map(active_vs_null_label)
        df = apply_display_labels(df)
        globals()[df_name] = df


RUN_CONFIG = config_df.iloc[0].to_dict() if len(config_df) > 0 else {}
EPOCH_TMIN = float(config_lookup(config_df, "epoch_tmin", DEFAULT_EPOCH_TMIN))
EPOCH_TMAX = float(config_lookup(config_df, "epoch_tmax", DEFAULT_EPOCH_TMAX))
BASELINE_WINDOW = coerce_interval(config_lookup(config_df, "baseline_window", DEFAULT_BASELINE_WINDOW), DEFAULT_BASELINE_WINDOW)
RESPONSE_WINDOW = coerce_interval(config_lookup(config_df, "response_window", DEFAULT_RESPONSE_WINDOW), DEFAULT_RESPONSE_WINDOW)
EMPIRICAL_NULL_SHIFT_COUNT = int(config_lookup(config_df, "empirical_null_shift_count", 0) or 0)
PPF_VALUE = config_lookup(config_df, "ppf_value", None)

PIPELINE_ORDER_ALL = ordered_pipeline_list(
    pipeline_manifest,
    fallback_frames=[channel_availability, canonical, block_avg, fir, shape, roi_timecourses, target_vs_nontarget, dlpfc_roi_scores],
)
PIPELINE_ORDER = [p for p in PIPELINE_ORDER_ALL if p not in EXCLUDED_PIPELINES]

for df_name in [
    "pipeline_manifest", "master_run_log", "truth_summary", "channel_quality", "pair_quality",
    "channel_availability", "canonical", "block_avg", "fir", "shape", "shape_fidelity_summary",
    "roi_timecourses", "roi_scores", "dlpfc_roi_scores", "target_vs_nontarget",
    "parametric_null_summary", "empirical_null_shift", "empirical_null_pvalues",
    "variability_summary", "pairwise_pipeline_deltas", "nuisance_detail",
    "homer3_channel_hrf", "homer3_beta", "homer3_glm_metadata",
    "pipeline_performance_summary", "error_log",
]:
    df = globals().get(df_name)
    if isinstance(df, pd.DataFrame) and len(df) > 0 and "pipeline_label" in df.columns:
        globals()[df_name] = df[df["pipeline_label"].astype(str).isin(PIPELINE_ORDER)].copy()

STANDARD_PIPELINES = [p for p in PIPELINE_ORDER if p not in FLEX_PIPELINES]
ACTIVE_FLEX_PIPELINES = [p for p in FLEX_PIPELINES if p in PIPELINE_ORDER]

print("Excluded pipelines:", EXCLUDED_PIPELINES)
print("n pipelines in analysis:", len(PIPELINE_ORDER))
print("PIPELINE_ORDER:", PIPELINE_ORDER)
print("n standard pipelines:", len(STANDARD_PIPELINES))
print("n active flex pipelines:", len(ACTIVE_FLEX_PIPELINES))
print("ACTIVE_FLEX_PIPELINES:", ACTIVE_FLEX_PIPELINES)
print("Epoch window:", (EPOCH_TMIN, EPOCH_TMAX))
print("Baseline window:", BASELINE_WINDOW)
print("Response window:", RESPONSE_WINDOW)
print("PPF value:", PPF_VALUE)
print("Empirical null shift count from config:", EMPIRICAL_NULL_SHIFT_COUNT)









# %% [markdown]
#      ## 3) Audit / schema sanity checks
# 
# 
# 
#      This section explicitly checks the current aggregate schema, current run progress,
# 
# 
# 
#      and common failure modes that matter for interpretation.

# %%
table_inventory_rows = []
for name in [
    "pipeline_manifest", "master_run_log", "truth_summary", "channel_quality", "pair_quality",
    "channel_availability", "canonical", "block_avg", "fir", "shape", "shape_fidelity_summary",
    "roi_timecourses", "roi_scores", "dlpfc_roi_scores", "target_vs_nontarget",
    "parametric_null_summary", "empirical_null_shift", "empirical_null_pvalues",
    "variability_summary", "pairwise_pipeline_deltas", "nuisance_detail",
    "homer3_channel_hrf", "homer3_beta", "homer3_glm_metadata",
    "pipeline_performance_summary", "config_df", "error_log",
]:
    df = globals().get(name)
    if isinstance(df, pd.DataFrame):
        table_inventory_rows.append({
            "table_name": name,
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "has_data": bool(len(df) > 0),
        })
table_inventory = pd.DataFrame(table_inventory_rows).sort_values("table_name")
display(table_inventory)
save_table(table_inventory, "audit_table_inventory")

run_progress = pd.DataFrame()
if len(channel_availability) > 0:
    run_progress = channel_availability.groupby(["subject", "file_label"], as_index=False).agg(
        n_pipelines_present=("pipeline_label", "nunique")
    )
    progress_summary = pd.DataFrame([{
        "n_subject_file_combinations": int(len(run_progress)),
        "n_subjects_seen": int(run_progress["subject"].nunique()),
        "median_pipelines_per_subject_file": float(run_progress["n_pipelines_present"].median()),
        "min_pipelines_per_subject_file": int(run_progress["n_pipelines_present"].min()),
        "max_pipelines_per_subject_file": int(run_progress["n_pipelines_present"].max()),
    }])
    display(progress_summary)
    save_table(run_progress, "audit_run_progress_subject_file")
    save_table(progress_summary, "audit_run_progress_summary")

shape_validity_audit = pd.DataFrame()
if len(shape) > 0:
    shape_validity_audit = shape.groupby("pipeline_label", as_index=False).agg(
        n_shape_rows=("pipeline_label", "count"),
        n_valid_curve_corr=("curve_corr", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
        n_valid_peak_amp=("recovered_peak_amplitude", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
    )
    if len(roi_timecourses) > 0:
        roi_presence = roi_timecourses.groupby("pipeline_label", as_index=False).agg(n_roi_rows=("pipeline_label", "count"))
        shape_validity_audit = shape_validity_audit.merge(roi_presence, on="pipeline_label", how="left")
    else:
        shape_validity_audit["n_roi_rows"] = 0
    shape_validity_audit["suspicious_shape_export"] = (
        (shape_validity_audit["n_shape_rows"] > 0) &
        (shape_validity_audit["n_valid_curve_corr"] == 0) &
        (shape_validity_audit["n_roi_rows"].fillna(0) > 0)
    )
    display(shape_validity_audit.sort_values(["suspicious_shape_export", "pipeline_label"], ascending=[False, True]))
    save_table(shape_validity_audit, "audit_shape_validity")

unit_audit_rows = [{
    "source": "config",
    "ppf_value": PPF_VALUE,
    "epoch_tmin": EPOCH_TMIN,
    "epoch_tmax": EPOCH_TMAX,
    "baseline_window": str(BASELINE_WINDOW),
    "response_window": str(RESPONSE_WINDOW),
    "empirical_null_shift_count": EMPIRICAL_NULL_SHIFT_COUNT,
}]
if len(homer3_glm_metadata) > 0:
    homer_meta_summary = homer3_glm_metadata.groupby("pipeline_label", as_index=False).agg(
        idx_basis_mode=("idx_basis", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan),
        stim_duration_s_effective_median=("stim_duration_s_effective", "median"),
        ppf_json_example=("ppf_json", "first"),
        n_rows=("pipeline_label", "count"),
    )
    display(homer_meta_summary)
    save_table(homer_meta_summary, "audit_homer3_metadata_summary")
    unit_audit_rows.extend(homer_meta_summary.to_dict(orient="records"))

unit_audit = pd.DataFrame(unit_audit_rows)
save_table(unit_audit, "audit_unit_summary")

audit_notes = []
if OUTPUT_DIR.name == "benchmark_outputs_merged_all_runs":
    audit_notes.append("Resolved output directory is benchmark_outputs_merged_all_runs. Verify that this is the intended merged directory for the current run.")
if len(shape_validity_audit) > 0 and shape_validity_audit["suspicious_shape_export"].any():
    flagged = ", ".join(shape_validity_audit.loc[shape_validity_audit["suspicious_shape_export"], "pipeline_label"].astype(str).tolist())
    audit_notes.append(f"Shape export appears suspicious for: {flagged}. All-pipeline shape figures below therefore use ROI timecourses instead of raw channel-level shape rows.")
if EMPIRICAL_NULL_SHIFT_COUNT > 0 and EMPIRICAL_NULL_SHIFT_COUNT < MIN_EMPIRICAL_NULL_SHIFTS:
    audit_notes.append(
        f"Empirical null shift count is only {EMPIRICAL_NULL_SHIFT_COUNT}. Empirical-null p-value figures are treated as diagnostic and will be suppressed by default."
    )
if len(empirical_null_pvalues) > 0 and "n_null_shifts" in empirical_null_pvalues.columns:
    max_null_shifts = int(pd.to_numeric(empirical_null_pvalues["n_null_shifts"], errors="coerce").max())
    if max_null_shifts < MIN_EMPIRICAL_NULL_SHIFTS:
        audit_notes.append(
            f"Aggregate empirical null table currently contains at most {max_null_shifts} null shifts per row. That is too few for thesis-grade empirical p-value plots."
        )
if not audit_notes:
    audit_notes.append("No schema-level blockers were detected in the aggregate tables.")
audit_notes_df = pd.DataFrame({"audit_note": audit_notes})
display(audit_notes_df)
save_table(audit_notes_df, "audit_notes")









# %% [markdown]
#      ## 4) Build ROI-derived comparable tables
# 
# 
# 
#      All main all-pipeline figures below are built from `roi_timecourses` so that
# 
# 
# 
#      every pipeline is evaluated on the same kind of object.
# 
# 
# 
#      Key design choices:
# 
# 
# 
#      - ROI amplitude metrics are computed on **baseline-corrected** curves.
# 
# 
# 
#      - Active-file peak / support windows are **truth-aware** when templates exist.
# 
# 
# 
#      - Shape metrics are computed from **ROI target curves for all pipelines**.
# 
# 
# 
#      - Null quietness uses **post-stim baseline-corrected magnitude**, not raw offsets.

# %%
def build_roi_curve_feature_tables(roi_df: pd.DataFrame):
    feature_rows = []
    shape_rows = []
    truth_window_rows = []

    if roi_df is None or len(roi_df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_cols = [
        "subject", "file_label", "amplitude_value", "pipeline_label", "backend",
        "chromophore", "target_status"
    ]

    for keys, grp in roi_df.groupby(group_cols):
        grp = grp.sort_values("time_s")
        t = grp["time_s"].astype(float).to_numpy()
        y = grp["signal"].astype(float).to_numpy()
        subject, file_label, amplitude_value, pipeline_label, backend, chrom, target_status = keys
        chrom = str(chrom).lower()
        target_status = str(target_status)

        truth_windows = derive_truth_windows(
            file_label=file_label,
            chrom=chrom,
            baseline_window=BASELINE_WINDOW,
            epoch_tmax=EPOCH_TMAX,
            default_response_window=RESPONSE_WINDOW,
        )
        truth_window_rows.append({
            "file_label": file_label,
            "chromophore": chrom,
            "baseline_window": str(truth_windows["baseline_window"]),
            "poststim_window": str(truth_windows["poststim_window"]),
            "support_window": str(truth_windows["support_window"]),
            "peak_window": str(truth_windows["peak_window"]),
            "truth_peak_time_s": truth_windows["truth_peak_time_s"],
            "has_truth": truth_windows["has_truth"],
        })

        y_bc, base_mean = baseline_correct(y, t, BASELINE_WINDOW)

        support_mask = in_window(t, truth_windows["support_window"]) & np.isfinite(y_bc)
        peak_mask = in_window(t, truth_windows["peak_window"]) & np.isfinite(y_bc)
        poststim_mask = in_window(t, truth_windows["poststim_window"]) & np.isfinite(y_bc)

        if support_mask.sum() == 0:
            support_mask = in_window(t, RESPONSE_WINDOW) & np.isfinite(y_bc)
        if peak_mask.sum() == 0:
            peak_mask = support_mask
        if poststim_mask.sum() == 0:
            poststim_mask = np.isfinite(y_bc)

        mean_response_bc = float(np.nanmean(y_bc[support_mask])) if np.any(support_mask) else np.nan
        response_auc_bc = float(scipy.integrate.trapezoid(y_bc[support_mask], t[support_mask])) if np.sum(support_mask) >= 2 else np.nan
        peak_amp_bc, peak_time = peak_value_and_time(y_bc[peak_mask], t[peak_mask], chrom) if np.any(peak_mask) else (np.nan, np.nan)
        peak_amp_raw, _ = peak_value_and_time(y[peak_mask], t[peak_mask], chrom) if np.any(peak_mask) else (np.nan, np.nan)
        mean_abs_poststim_bc = float(np.nanmean(np.abs(y_bc[poststim_mask]))) if np.any(poststim_mask) else np.nan
        peak_abs_poststim_bc = float(np.nanmax(np.abs(y_bc[poststim_mask]))) if np.any(poststim_mask) else np.nan

        feature_rows.append(
            {
                "subject": subject,
                "file_label": file_label,
                "amplitude_value": amplitude_value,
                "pipeline_label": pipeline_label,
                "backend": backend,
                "chromophore": chrom,
                "target_status": target_status,
                "curve_source": grp["curve_source"].iloc[0] if "curve_source" in grp.columns else "roi_timecourses",
                "baseline_mean": float(base_mean),
                "mean_response_bc": mean_response_bc,
                "response_auc_bc": response_auc_bc,
                "peak_amplitude_bc": peak_amp_bc,
                "peak_amplitude_raw": peak_amp_raw,
                "peak_time_s": peak_time,
                "mean_abs_poststim_bc": mean_abs_poststim_bc,
                "peak_abs_poststim_bc": peak_abs_poststim_bc,
            }
        )

        if target_status == "true_target" and float(amplitude_value) > 0:
            truth = load_truth_curve(file_label, chrom=chrom)
            if truth is not None:
                truth_y = np.interp(t, truth["time_s"], truth["truth_signal"], left=0.0, right=0.0)
                shape_rows.append(
                    {
                        "subject": subject,
                        "file_label": file_label,
                        "amplitude_value": amplitude_value,
                        "pipeline_label": pipeline_label,
                        "backend": backend,
                        "chromophore": chrom,
                        "target_status": target_status,
                        "curve_source": grp["curve_source"].iloc[0] if "curve_source" in grp.columns else "roi_timecourses",
                        **compute_curve_metrics(
                            y,
                            truth_y,
                            t,
                            chrom,
                            baseline_window=BASELINE_WINDOW,
                            eval_window=(0.0, EPOCH_TMAX),
                            peak_window=truth_windows["peak_window"],
                            auc_window=truth_windows["support_window"],
                        ),
                    }
                )

    feature_df = pd.DataFrame(feature_rows)
    shape_df = pd.DataFrame(shape_rows)
    truth_window_df = pd.DataFrame(truth_window_rows).drop_duplicates()

    summary_tables = []
    for value_column, score_type in [
        ("peak_amplitude_bc", "roi_peak_amplitude_bc"),
        ("mean_response_bc", "roi_mean_response_bc"),
        ("response_auc_bc", "roi_response_auc_bc"),
    ]:
        tmp = summarize_target_minus_non_target_any(
            feature_df,
            value_column=value_column,
            group_columns=["subject", "file_label", "amplitude_value", "pipeline_label", "backend", "chromophore"],
            score_type=score_type,
        )
        if len(tmp) > 0:
            summary_tables.append(tmp)

    summary_df = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    return feature_df, shape_df, summary_df, truth_window_df


roi_curve_features, roi_curve_shape_targets, roi_target_vs_nontarget, truth_window_table = build_roi_curve_feature_tables(roi_timecourses)
save_table(roi_curve_features, "roi_curve_features_v7")
save_table(roi_curve_shape_targets, "roi_curve_shape_targets_v7")
save_table(roi_target_vs_nontarget, "roi_target_vs_nontarget_summary_v7")
save_table(truth_window_table, "truth_windows_v7")

if len(roi_target_vs_nontarget) > 0:
    roi_target_vs_nontarget["file_family"] = roi_target_vs_nontarget["file_label"].map(file_family_from_label)
    roi_target_vs_nontarget["is_official_anchor"] = roi_target_vs_nontarget["file_label"].map(official_anchor_flag)
    roi_target_vs_nontarget["active_vs_null"] = roi_target_vs_nontarget["amplitude_value"].map(active_vs_null_label)

print("roi_curve_features:", roi_curve_features.shape)
print("roi_curve_shape_targets:", roi_curve_shape_targets.shape)
print("roi_target_vs_nontarget:", roi_target_vs_nontarget.shape)
print("truth_window_table:", truth_window_table.shape)









# %% [markdown]
#      ### Plotting audit
# 
# 
# 
#      Main thesis-facing summary choices in this version:
# 
#      - bars for retention and the standard-family primary endpoint
# 
#      - point-range plots for waveform fidelity, amplitude recovery, dose-response,
# 
#        null quietness, AUROC, and DLPFC bridge summaries
# 
#      - box plots kept where the distribution itself is informative

# %% [markdown]
#      ## 5) Figure 1 — Pipeline family overview

# %%
pipeline_overview = pd.DataFrame({"pipeline_label": PIPELINE_ORDER})
if len(pipeline_manifest) > 0:
    manifest_cols = [
        c for c in [
            "pipeline_label", "pipeline_order", "backend", "hrf_model", "nuisance_method",
            "solver", "secondary_pipeline", "comparison_group", "description"
        ]
        if c in pipeline_manifest.columns
    ]
    if manifest_cols:
        pipeline_overview = pipeline_overview.merge(
            pipeline_manifest[manifest_cols].drop_duplicates(subset=["pipeline_label"]),
            on="pipeline_label",
            how="left",
        )

rows = []
for _, row in pipeline_overview.iterrows():
    nuisance, model, solver, family = infer_pipeline_fields(
        row["pipeline_label"],
        backend=row["backend"] if "backend" in row.index else None,
        hrf_model=row["hrf_model"] if "hrf_model" in row.index else None,
        nuisance_method=row["nuisance_method"] if "nuisance_method" in row.index else None,
    )
    rows.append(
        {
            "Pipeline": row["pipeline_label"],
            "Family": family,
            "Nuisance / preprocessing": nuisance,
            "Response model / basis": model,
            "Solver / backend": solver,
            "Group": row["comparison_group"] if "comparison_group" in row.index else "",
        }
    )

pipeline_overview_pretty = pd.DataFrame(rows)
save_table(pipeline_overview_pretty, "pipeline_family_overview_v7")

fig = go.Figure(
    data=[
        go.Table(
            header=dict(values=list(pipeline_overview_pretty.columns)),
            cells=dict(values=[pipeline_overview_pretty[c] for c in pipeline_overview_pretty.columns]),
        )
    ]
)
fig.update_layout(title="Pipeline family overview")
save_fig(fig, "pipeline_family_overview_v7")
if SHOW_FIGURES:
    fig.show()









# %% [markdown]
#      ## 6) Figure 2 — Retention across all subject-file jobs

# %%
retention_summary = pd.DataFrame()
retention_long = pd.DataFrame()
if len(channel_availability) > 0:
    retention_summary = channel_availability.groupby(["pipeline_label", "active_vs_null", "file_family"], as_index=False).agg(
        mean_target_retention=("target_pair_retention_fraction", "mean"),
        mean_non_target_retention=("non_target_pair_retention_fraction", "mean"),
        median_target_retention=("target_pair_retention_fraction", "median"),
        median_non_target_retention=("non_target_pair_retention_fraction", "median"),
        mean_bad_pairs=("n_bad_pairs", "mean"),
        n_subject_file_rows=("file_label", "count"),
    )
    save_table(retention_summary, "retention_summary_by_pipeline_v7")

    retention_long = channel_availability.melt(
        id_vars=["subject", "file_label", "pipeline_label", "active_vs_null", "file_family"],
        value_vars=["target_pair_retention_fraction", "non_target_pair_retention_fraction"],
        var_name="pair_type",
        value_name="retention_fraction",
    )
    retention_long["pair_type"] = retention_long["pair_type"].replace({
        "target_pair_retention_fraction": "Target channels",
        "non_target_pair_retention_fraction": "Non-target channels",
    })
    retention_long = apply_display_labels(retention_long)

    retention_subject = retention_long.groupby(["subject", "pipeline_label"], as_index=False).agg(
        mean_retention_fraction=("retention_fraction", "mean")
    )
    save_table(retention_subject, "retention_subject_level_for_stats_v7")
    retention_long = apply_display_labels(retention_long)

    retention_order = ordered_pipeline_display_labels(
        retention_long["pipeline_label"].astype(str).unique().tolist()
    )
    fig = px.box(
        retention_long,
        x="pipeline_display",
        y="retention_fraction",
        color="pair_type",
        points=False,
        title="Channel retention across semisynthetic data",
        labels={
            "pipeline_display": "Pipeline",
            "retention_fraction": "Retention fraction",
            "pair_type": "Channel class",
        },
        category_orders={"pipeline_display": retention_order},
        color_discrete_sequence=["#4C78A8", "#E45756"],
    )
    
    fig.update_traces(line=dict(width=1.25), whiskerwidth=0.65)
    fig.update_xaxes(tickangle=55)
    style_figure(fig, title="Channel retention across subject-file combinations")
    save_fig(fig, "retention_box_all_v14")
    if SHOW_FIGURES:
        fig.show()

    retention_subject = retention_long.groupby(["subject", "pipeline_label"], as_index=False).agg(
        mean_retention_fraction=("retention_fraction", "mean")
    )
    save_table(retention_subject, "retention_subject_level_for_stats_v7")
    make_subject_level_boxplot(
        retention_subject,
        value_col="mean_retention_fraction",
        title="Subject-level mean channel retention by pipeline",
        y_label="Mean retention fraction",
        out_stem="retention_subject_level_box_v7",
        points="all",
    )
    _, retention_omnibus, retention_pairwise, retention_pivot = run_metric_suite(
        retention_subject,
        value_col="mean_retention_fraction",
        metric_name="subject_mean_retention_all_pairs",
        reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in retention_subject["pipeline_label"].astype(str).unique() else None,
        summary_name="retention_subject_mean_v7",
        figure_title="Mean channel retention by pipeline",
        figure_ylabel="Mean retention fraction",
        figure_stem="retention_subject_mean_bar_v7",
        ascending=False,
    )



# %% [markdown]
#      ## 7) Figure 3 — Waveform fidelity by pipeline (all pipelines, ROI curves only)
# 
# 
# 
#      The old notebook mixed channel-level shape summaries with ROI-level shape summaries.
# 
# 
# 
#      This version evaluates shape on the same ROI target curves for every pipeline.

# %%
roi_shape_subject_summary = pd.DataFrame()
if len(roi_curve_shape_targets) > 0:
    roi_shape_subject_summary = roi_curve_shape_targets.groupby(["subject", "pipeline_label", "chromophore"], as_index=False).agg(
        mean_curve_corr=("curve_corr", "mean"),
        mean_curve_nrmse=("curve_nrmse", "mean"),
        mean_peak_latency_error_s=("peak_latency_error_s", "mean"),
        mean_peak_amplitude_ratio=("peak_amplitude_ratio", "mean"),
    )
    save_table(roi_shape_subject_summary, "roi_shape_subject_summary_active_targets_v7")

shape_hbo = roi_shape_subject_summary[roi_shape_subject_summary["chromophore"] == PRIMARY_CHROM].copy() if len(roi_shape_subject_summary) > 0 else pd.DataFrame()

shape_corr_subject = shape_hbo[["subject", "pipeline_label", "mean_curve_corr"]].copy() if len(shape_hbo) > 0 else pd.DataFrame()
shape_nrmse_subject = shape_hbo[["subject", "pipeline_label", "mean_curve_nrmse"]].copy() if len(shape_hbo) > 0 else pd.DataFrame()

_, shape_corr_omnibus, shape_corr_pairwise, _ = run_metric_suite(
    shape_corr_subject,
    value_col="mean_curve_corr",
    metric_name="subject_mean_curve_corr_active_targets_roi_hbo",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in shape_corr_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="shape_curve_corr_roi_v7",
    figure_title="ROI waveform-shape correlation by pipeline (active targets, HbO)",
    figure_ylabel="Mean waveform-shape correlation",
    figure_stem="shape_curve_corr_roi_bar_v7",
    ascending=False,
)

_, shape_nrmse_omnibus, shape_nrmse_pairwise, _ = run_metric_suite(
    shape_nrmse_subject,
    value_col="mean_curve_nrmse",
    metric_name="subject_mean_curve_nrmse_active_targets_roi_hbo",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in shape_nrmse_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="shape_curve_nrmse_roi_v7",
    figure_title="ROI waveform normalized RMSE by pipeline (active targets, HbO)",
    figure_ylabel="Mean normalized RMSE",
    figure_stem="shape_curve_nrmse_roi_bar_v7",
    ascending=True,
)

shape_latency_subject = shape_hbo[["subject", "pipeline_label", "mean_peak_latency_error_s"]].copy() if len(shape_hbo) > 0 else pd.DataFrame()
if len(shape_latency_subject) > 0:
    shape_latency_subject["mean_abs_peak_latency_error_s"] = np.abs(shape_latency_subject["mean_peak_latency_error_s"].astype(float))

_, shape_latency_omnibus, shape_latency_pairwise, _ = run_metric_suite(
    shape_latency_subject,
    value_col="mean_abs_peak_latency_error_s",
    metric_name="subject_mean_absolute_peak_latency_error_active_targets_roi_hbo",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in shape_latency_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="shape_peak_latency_roi_v7",
    figure_title="Peak-latency error by pipeline (active targets, HbO)",
    figure_ylabel="Mean absolute peak-latency error (s)",
    figure_stem="shape_peak_latency_roi_bar_v7",
    ascending=True,
)









# %% [markdown]
#      ## 8) Figure 4 — ROI peak target-minus-non-target by pipeline (official anchors, HbO)
# 
# 
# 
#      Main amplitude/recovery figure for all pipelines.
# 
# 
# 
#      This version uses **baseline-corrected ROI peak amplitude**.

# %%
roi_peak_primary = pd.DataFrame()
if len(roi_target_vs_nontarget) > 0:
    roi_peak_primary = roi_target_vs_nontarget[
        (roi_target_vs_nontarget["score_type"] == "roi_peak_amplitude_bc") &
        (roi_target_vs_nontarget["chromophore"] == PRIMARY_CHROM) &
        (roi_target_vs_nontarget["active_vs_null"] == "active") &
        (roi_target_vs_nontarget["is_official_anchor"] == True)
    ].copy()
    if PRIMARY_RECOVERY_REQUIRE_COMPLETE_ANCHORS:
        roi_peak_primary = restrict_to_complete_official_anchor_sets(roi_peak_primary)

roi_peak_subject = pd.DataFrame()
if len(roi_peak_primary) > 0:
    roi_peak_subject = roi_peak_primary.groupby(["subject", "pipeline_label"], as_index=False).agg(
        mean_target_minus_non_target_peak=("target_minus_non_target_score", "mean"),
        n_files=("file_label", "nunique"),
    )
    save_table(roi_peak_subject, "roi_peak_subject_level_official_v7")

_, roi_peak_omnibus, roi_peak_pairwise, _ = run_metric_suite(
    roi_peak_subject,
    value_col="mean_target_minus_non_target_peak",
    metric_name="subject_mean_roi_peak_target_minus_non_target_official_hbo_bc",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in roi_peak_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="roi_peak_official_v7",
    figure_title="Recovered ROI peak target–non-target amplitude by pipeline (official anchors, HbO)",
    figure_ylabel=f"Mean ROI peak target–non-target amplitude ({CONCENTRATION_UNIT_LABEL})",
    figure_stem="roi_peak_target_minus_non_target_official_pointrange_v14",
    ascending=False,
    plot_kind="pointrange",
)









# %% [markdown]
#      ## 9) Figure 5 — ROI peak dose-response slope by pipeline (official anchors, HbO)
# 
# 
# 
#      This figure now requires the full official anchor set per subject/pipeline.

# %%
roi_peak_amp_resp_official = amplitude_response_table(
    roi_target_vs_nontarget,
    use_official_only=True,
    chrom=PRIMARY_CHROM,
    score_type="roi_peak_amplitude_bc",
    require_complete_anchor_set=PRIMARY_DOSE_RESPONSE_REQUIRE_COMPLETE_ANCHORS,
)
roi_peak_amp_resp_all = amplitude_response_table(
    roi_target_vs_nontarget,
    use_official_only=False,
    chrom=PRIMARY_CHROM,
    score_type="roi_peak_amplitude_bc",
    require_complete_anchor_set=False,
)
save_table(roi_peak_amp_resp_official, "roi_peak_amplitude_response_official_v7")
save_table(roi_peak_amp_resp_all, "roi_peak_amplitude_response_all_active_v7")

_, roi_peak_slope_omnibus, roi_peak_slope_pairwise, _ = run_metric_suite(
    roi_peak_amp_resp_official,
    value_col="slope",
    metric_name="subject_roi_peak_dose_response_slope_official_hbo_bc",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in roi_peak_amp_resp_official.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="roi_peak_amplitude_response_official_v7",
    figure_title="Dose-response slope of recovered ROI peak amplitude by pipeline (official anchors, HbO)",
    figure_ylabel="Mean subject-level dose-response slope",
    figure_stem="roi_peak_amplitude_response_slope_official_pointrange_v14",
    ascending=False,
    plot_kind="pointrange",
)









# %% [markdown]
#      ## 10) Figure 6 — Mean absolute null ROI response by pipeline (HbO)
# 
# 
# 
#      Main all-pipeline null figure.
# 
# 
# 
#      Uses baseline-corrected post-stim magnitude, not raw ROI offsets.

# %%
null_roi_subject = pd.DataFrame()
if len(roi_curve_features) > 0:
    null_roi = roi_curve_features[
        (roi_curve_features["amplitude_value"] == 0) &
        (roi_curve_features["chromophore"] == PRIMARY_CHROM)
    ].copy()
    if len(null_roi) > 0:
        null_roi_subject = null_roi.groupby(["subject", "pipeline_label"], as_index=False).agg(
            mean_abs_null_response=("mean_abs_poststim_bc", "mean"),
            median_abs_null_response=("mean_abs_poststim_bc", "median"),
            mean_peak_abs_null_response=("peak_abs_poststim_bc", "mean"),
            n_files=("file_label", "nunique"),
        )
        save_table(null_roi_subject, "null_roi_subject_level_abs_response_v7")

_, null_roi_omnibus, null_roi_pairwise, _ = run_metric_suite(
    null_roi_subject,
    value_col="mean_abs_null_response",
    metric_name="subject_mean_absolute_null_roi_response_hbo_bc_poststim",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in null_roi_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="null_roi_abs_response_v7",
    figure_title="Null ROI response magnitude by pipeline (HbO)",
    figure_ylabel=f"Mean absolute null ROI response ({CONCENTRATION_UNIT_LABEL})",
    figure_stem="null_absolute_roi_response_pointrange_v14",
    ascending=True,
    plot_kind="pointrange",
)

# Literature-informed supporting benchmark: active-vs-null discrimination using the
# same ROI recovery score. This provides a sensitivity/specificity-style summary without
# requiring expensive reruns for empirical null permutation distributions.
active_null_discrimination = active_null_discrimination_table(
    roi_target_vs_nontarget,
    chrom=PRIMARY_CHROM,
    score_type="roi_peak_amplitude_bc",
    use_official_active_only=True,
)
save_table(active_null_discrimination, "active_null_discrimination_roi_peak_v7")

_, active_null_auroc_omnibus, active_null_auroc_pairwise, _ = run_metric_suite(
    active_null_discrimination,
    value_col="auroc",
    metric_name="subject_active_vs_null_auroc_roi_peak_hbo",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in active_null_discrimination.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="active_null_discrimination_auroc_v7",
    figure_title="Active-versus-null discrimination by pipeline (AUROC, HbO)",
    figure_ylabel="Subject-level AUROC",
    figure_stem="active_null_discrimination_auroc_pointrange_v14",
    ascending=False,
    plot_kind="pointrange",
)









# %% [markdown]
#      ## 11) Figure 7 — Standard-family recovery figure
# 
# 
# 
#      Supporting figure for the canonical-beta family only.

# %%
standard_subject_level_primary = pd.DataFrame()
if len(target_vs_nontarget) > 0:
    standard_primary_rows = target_vs_nontarget[
        (target_vs_nontarget["score_type"] == PRIMARY_SCORE_TYPE) &
        (target_vs_nontarget["chromophore"] == PRIMARY_CHROM) &
        (target_vs_nontarget["active_vs_null"] == "active") &
        (target_vs_nontarget["pipeline_label"].isin(STANDARD_PIPELINES))
    ].copy()
    if len(standard_primary_rows) > 0:
        standard_subject_level_primary = standard_primary_rows.groupby(["subject", "pipeline_label"], as_index=False).agg(
            mean_target_minus_non_target_score=("target_minus_non_target_score", "mean"),
            median_target_minus_non_target_score=("target_minus_non_target_score", "median"),
            n_files=("file_label", "nunique"),
        )
        save_table(standard_subject_level_primary, "standard_subject_level_primary_endpoint_v7")

_, standard_primary_omnibus, standard_primary_pairwise, standard_primary_pivot = run_metric_suite(
    standard_subject_level_primary,
    value_col="mean_target_minus_non_target_score",
    metric_name=f"subject_mean_{PRIMARY_SCORE_TYPE}_{PRIMARY_CHROM}_active_standard_only_v7",
    reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in standard_subject_level_primary.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
    summary_name="standard_recovery_primary_v7",
    figure_title="Primary endpoint by pipeline (canonical beta, HbO; standard family)",
    figure_ylabel="Mean canonical beta (a.u.)",
    figure_stem="standard_primary_endpoint_bootstrap_bar_v7",
    ascending=False,
)









# %% [markdown]
#      ## 12) Supporting inferential null-safety diagnostics
# 
# 
# 
#      Empirical-null p-values are only shown if the null distribution is dense enough to
# 
# 
# 
#      support interpretable p-value resolution. When empirical null reruns were infeasible,
# 
# 
# 
#      the thesis should rely primarily on (i) all-pipeline null quietness above and (ii)
# 
# 
# 
#      parametric null false-positive rates within score families that yield comparable
# 
# 
# 
#      inferential statistics.

# %%
empirical_null_is_thesis_grade = False
if len(empirical_null_pvalues) > 0 and "n_null_shifts" in empirical_null_pvalues.columns:
    empirical_null_is_thesis_grade = int(pd.to_numeric(empirical_null_pvalues["n_null_shifts"], errors="coerce").max()) >= MIN_EMPIRICAL_NULL_SHIFTS

if empirical_null_is_thesis_grade and len(empirical_null_pvalues) > 0:
    emp_plot = empirical_null_pvalues[
        (empirical_null_pvalues["score_type"] == PRIMARY_SCORE_TYPE) &
        (empirical_null_pvalues["chromophore"] == PRIMARY_CHROM) &
        (empirical_null_pvalues["pipeline_label"].isin(STANDARD_PIPELINES))
    ].copy()
    if len(emp_plot) > 0:
        fig = px.box(
            emp_plot,
            x="pipeline_label",
            y="empirical_p_value",
            title="Empirical-null p-values by pipeline (canonical beta, HbO; standard family)",
            labels={"pipeline_label": "Pipeline", "empirical_p_value": "Empirical p-value"},
        )
        fig.update_xaxes(tickangle=60)
        save_fig(fig, "standard_empirical_null_pvalues_box_v7")
        if SHOW_FIGURES:
            fig.show()

        emp_subject = emp_plot.groupby(["subject", "pipeline_label"], as_index=False).agg(
            mean_empirical_p=("empirical_p_value", "mean")
        )
        save_table(emp_subject, "standard_empirical_null_subject_level_v7")
        _, emp_omnibus, emp_pairwise, _ = run_metric_suite(
            emp_subject,
            value_col="mean_empirical_p",
            metric_name="subject_mean_empirical_null_p_standard_hbo_v7",
            reference_pipeline=REFERENCE_PIPELINE if REFERENCE_PIPELINE in emp_subject.get("pipeline_label", pd.Series(dtype=str)).astype(str).unique() else None,
            summary_name="standard_empirical_null_v7",
            figure_title="Mean empirical-null p-value by pipeline (standard family)",
            figure_ylabel="Mean empirical p-value (higher is more null-safe)",
            figure_stem="standard_empirical_null_subject_mean_bar_v7",
            ascending=False,
        )
else:
    print("Skipping empirical-null p-value figure: empirical reruns were absent or too sparse for defensible p-value resolution. Use null-quietness, AUROC, and parametric-null summaries instead.")

if INCLUDE_SUPPORTING_FIGURES and len(parametric_null_summary) > 0:
    param_plot = parametric_null_summary[parametric_null_summary["pipeline_label"].isin(STANDARD_PIPELINES)].copy()
    if len(param_plot) > 0 and "false_positive_rate_p_lt_0_05" in param_plot.columns:
        fig = px.box(
            param_plot,
            x="pipeline_label",
            y="false_positive_rate_p_lt_0_05",
            color="target_status" if "target_status" in param_plot.columns else None,
            title="Parametric false-positive rate by pipeline (standard family)",
            labels={"pipeline_label": "Pipeline", "false_positive_rate_p_lt_0_05": "False-positive rate at p < 0.05"},
        )
        fig.update_xaxes(tickangle=60)
        save_fig(fig, "standard_parametric_null_fpr_v7")
        if SHOW_FIGURES:
            fig.show()









# %% [markdown]
#      ## 13) Figure 9 — DLPFC ROI scores across subjects
# 
# 
# 
#      The old notebook implicitly preferred `canonical_beta` whenever it existed, which
# 
# 
# 
#      could silently exclude non-canonical families. This version selects the score family
# 
# 
# 
#      explicitly and labels the figure accordingly.

# %%
dlpfc_subject_summary = pd.DataFrame()
dlpfc_bridge_note = ""
if len(dlpfc_roi_scores) > 0:
    dlpfc_active = dlpfc_roi_scores[
        (dlpfc_roi_scores["active_vs_null"] == "active") &
        (dlpfc_roi_scores["chromophore"] == PRIMARY_CHROM)
    ].copy()

    available_score_types = sorted(dlpfc_active["score_type"].dropna().astype(str).unique().tolist()) if "score_type" in dlpfc_active.columns else []

    dlpfc_score_type = None
    if DLPFC_BRIDGE_SCORE_TYPE in available_score_types:
        dlpfc_score_type = DLPFC_BRIDGE_SCORE_TYPE
        dlpfc_bridge_note = f"DLPFC bridge uses the prespecified score family {pretty_score_type(dlpfc_score_type)}."
    elif ALLOW_DLPFC_BRIDGE_FALLBACK:
        for score_type in DLPFC_SCORE_TYPE_PREFERENCE:
            if score_type in available_score_types:
                dlpfc_score_type = score_type
                break
        if dlpfc_score_type is not None:
            dlpfc_bridge_note = f"Prespecified DLPFC bridge score {pretty_score_type(DLPFC_BRIDGE_SCORE_TYPE)} was unavailable; fallback used {pretty_score_type(dlpfc_score_type)}. Interpret as a bridge-compatibility compromise, not a silent estimand change."
    else:
        dlpfc_bridge_note = f"Prespecified DLPFC bridge score {DLPFC_BRIDGE_SCORE_TYPE} was unavailable; no fallback was applied."

    if dlpfc_score_type is not None:
        dlpfc_active = dlpfc_active[dlpfc_active["score_type"] == dlpfc_score_type].copy()

    if len(dlpfc_active) > 0:
        dlpfc_subject_summary = dlpfc_active.groupby(["subject", "pipeline_label", "roi_name"], as_index=False).agg(
            mean_roi_score=("roi_mean_score", "mean"),
            median_roi_score=("roi_mean_score", "median"),
            n_files=("file_label", "nunique"),
        )
        dlpfc_subject_summary = apply_display_labels(dlpfc_subject_summary)
        save_table(dlpfc_subject_summary, "dlpfc_subject_summary_bridge_v7")
        title_suffix = f"{pretty_score_type(dlpfc_score_type)}, HbO"

        bilateral_order = (
            dlpfc_subject_summary[dlpfc_subject_summary["roi_name"] == "dlpfc_bilateral"]
            .groupby(["pipeline_label", "pipeline_display"], as_index=False)["mean_roi_score"]
            .mean()
            .sort_values("mean_roi_score", ascending=False)
        )
        ordered_display = bilateral_order["pipeline_display"].astype(str).tolist()

        rows = []
        for roi_name in ["dlpfc_bilateral", "dlpfc_left", "dlpfc_right"]:
            roi_df = dlpfc_subject_summary[dlpfc_subject_summary["roi_name"] == roi_name].copy()
            if len(roi_df) == 0:
                continue
            summ = summarize_with_bootstrap(
                roi_df.rename(columns={"mean_roi_score": "value"}),
                "value",
                ["pipeline_label", "pipeline_display"],
            )
            summ["roi_name"] = roi_name
            summ["roi_display"] = pretty_roi_name(roi_name)
            rows.append(summ)

        dlpfc_plot_summary = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        save_table(dlpfc_plot_summary, "dlpfc_bridge_plot_summary_v11")

        if len(dlpfc_plot_summary) > 0:
            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=False, shared_yaxes=False,
                vertical_spacing=0.08,
                subplot_titles=[pretty_roi_name("dlpfc_bilateral"), pretty_roi_name("dlpfc_left"), pretty_roi_name("dlpfc_right")]
            )
            for i, roi_name in enumerate(["dlpfc_bilateral", "dlpfc_left", "dlpfc_right"], start=1):
                sub_plot = dlpfc_plot_summary[dlpfc_plot_summary["roi_name"] == roi_name].copy()
                if len(sub_plot) == 0:
                    continue
                sub_plot["pipeline_display"] = pd.Categorical(sub_plot["pipeline_display"], categories=ordered_display, ordered=True)
                sub_plot = sub_plot.sort_values("pipeline_display", ascending=True)
                fig.add_trace(
                    go.Scatter(
                        x=sub_plot["value_mean"],
                        y=sub_plot["pipeline_display"].astype(str),
                        mode="markers",
                        marker=dict(size=10, color="#0072B2"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=sub_plot["value_ci_hi"] - sub_plot["value_mean"],
                            arrayminus=sub_plot["value_mean"] - sub_plot["value_ci_lo"],
                            thickness=1.5,
                            width=0,
                            color="#0072B2",
                        ),
                        showlegend=False,
                        hovertemplate="%{y}<br>Mean=%{x:.3e}<extra></extra>",
                    ),
                    row=i, col=1
                )
                fig.update_yaxes(title_text="", categoryorder="array", categoryarray=ordered_display[::-1], automargin=True, tickfont=dict(size=13), row=i, col=1)
                fig.update_xaxes(title_text="Mean DLPFC ROI score (95% bootstrap CI)" if i == 3 else "", row=i, col=1)

            fig.update_layout(
                title=f"DLPFC bridge metric by pipeline ({title_suffix})",
                height=1850,
                width=1850,
                margin=dict(l=320, r=40, t=120, b=90),
            )
            save_fig(fig, "dlpfc_roi_bridge_pointrange_v11")
            if SHOW_FIGURES:
                fig.show()

        lr = dlpfc_subject_summary[dlpfc_subject_summary["roi_name"].isin(["dlpfc_left", "dlpfc_right"])].copy()
        if len(lr) > 0:
            lr_wide = lr.pivot_table(index=["subject", "pipeline_label"], columns="roi_name", values="mean_roi_score").reset_index()
            if {"dlpfc_left", "dlpfc_right"}.issubset(lr_wide.columns):
                lr_wide["right_minus_left"] = lr_wide["dlpfc_right"] - lr_wide["dlpfc_left"]
                lr_wide = apply_display_labels(lr_wide)
                save_table(lr_wide, "dlpfc_left_right_asymmetry_bridge_v7")
                if INCLUDE_SUPPORTING_FIGURES:
                    fig = px.box(
                        lr_wide,
                        x="pipeline_display",
                        y="right_minus_left",
                        title=f"DLPFC right–left asymmetry by pipeline ({title_suffix})",
                        labels={"pipeline_display": "Pipeline", "right_minus_left": "Right–left DLPFC ROI score"},
                    )
                    fig.update_xaxes(tickangle=60)
                    style_figure(fig, title=f"DLPFC right–left asymmetry by pipeline ({title_suffix})")
                    save_fig(fig, "dlpfc_right_minus_left_v14")
                    if SHOW_FIGURES:
                        fig.show()
                    bilateral = dlpfc_subject_summary[dlpfc_subject_summary["roi_name"] == "dlpfc_bilateral"][["subject", "pipeline_label", "mean_roi_score"]].rename(columns={"mean_roi_score": "bilateral_score"})
                    asym_norm = lr_wide.merge(bilateral, on=["subject", "pipeline_label"], how="left")
                    asym_norm["normalized_right_minus_left"] = asym_norm["right_minus_left"] / asym_norm["bilateral_score"].replace({0: np.nan})
                    save_table(asym_norm, "dlpfc_left_right_normalized_asymmetry_bridge_v16")
                    if INCLUDE_SUPPORTING_FIGURES:
                        fig = px.box(
                            asym_norm,
                            x="pipeline_display",
                            y="normalized_right_minus_left",
                            title=f"Normalized DLPFC right–left asymmetry by pipeline ({title_suffix})",
                            labels={"pipeline_display": "Pipeline", "normalized_right_minus_left": "(Right – left) / bilateral DLPFC score"},
                        )
                        fig.update_xaxes(tickangle=60)
                        style_figure(fig, title=f"Normalized DLPFC right–left asymmetry by pipeline ({title_suffix})")
                        save_fig(fig, "dlpfc_right_minus_left_normalized_v16")
                        if SHOW_FIGURES:
                            fig.show()

if dlpfc_bridge_note:
    print(dlpfc_bridge_note)
    pd.DataFrame([{"note": dlpfc_bridge_note}]).to_csv(EXPORT_DIR / "dlpfc_bridge_note_v7.csv", index=False)









# %% [markdown]
#      ## 14) Figure 10 — Official-anchor ROI recovery overlays (all available subjects)
# 
# 
# 
#      ModGamma is excluded for now. These plots are baseline-corrected and remain on the
# 
# 
# 
#      recovered concentration scale; they are **not normalized**.
# 
# 
# 
#      The notebook generates one overlay for each available subject × official HRF file
# 
# 
# 
#      (`resting_hrf_20`, `resting_hrf_50`, `resting_hrf_100`) present in the exported data,
# 
# 
# 
#      plotting all available non-ModGamma pipelines for that subject/file.

# %%
official_overlay_inventory = pd.DataFrame()
official_overlay_failures = pd.DataFrame()

if len(roi_timecourses) > 0:
    overlay_candidates = roi_timecourses[
        (roi_timecourses["file_label"].isin(OFFICIAL_ACTIVE_FILES)) &
        (roi_timecourses["chromophore"].astype(str).str.lower() == PRIMARY_CHROM.lower()) &
        (roi_timecourses["target_status"].astype(str) == "true_target")
    ][["subject", "file_label"]].drop_duplicates().sort_values(["subject", "file_label"])

    overlay_rows = []
    failure_rows = []

    for _, row in overlay_candidates.iterrows():
        subject = str(row["subject"])
        file_label = str(row["file_label"])
        sub = load_roi_timecourses_for_subject_file(roi_timecourses, subject, file_label)
        sub = sub[
            (sub["chromophore"].astype(str).str.lower() == PRIMARY_CHROM.lower()) &
            (sub["target_status"].astype(str) == "true_target")
        ].copy()
        available_pipelines = [p for p in PIPELINE_ORDER if p in set(sub["pipeline_label"].astype(str).unique())]
        overlay_rows.append({
            "subject": subject,
            "file_label": file_label,
            "n_available_pipelines": int(len(available_pipelines)),
            "pipelines": ", ".join(available_pipelines),
        })

    official_overlay_inventory = pd.DataFrame(overlay_rows)
    save_table(official_overlay_inventory, "official_overlay_inventory_v7")

    if REPRESENTATIVE_SUBJECT is None:
        rep_subject, rep_file = choose_representative_subject_file(
            roi_timecourses,
            pipelines=PIPELINE_ORDER,
            preferred_file=REPRESENTATIVE_FILE_LABEL,
            chrom=PRIMARY_CHROM,
        )
    else:
        rep_subject, rep_file = REPRESENTATIVE_SUBJECT, REPRESENTATIVE_FILE_LABEL

    print("Representative overlay selection:", rep_subject, rep_file)

    if rep_subject is not None and rep_file is not None:
        try:
            fig = plot_roi_recovery(
                roi_timecourses=roi_timecourses,
                subject=rep_subject,
                file_label=rep_file,
                pipelines=PIPELINE_ORDER,
                chrom=PRIMARY_CHROM,
            )
            save_fig(fig, f"roi_recovery_bc_{rep_subject}_{rep_file}_{PRIMARY_CHROM}_representative_v15")
            if SHOW_FIGURES:
                fig.show()
        except Exception as exc:
            failure_rows.append({
                "subject": rep_subject,
                "file_label": rep_file,
                "reason": str(exc),
            })

    if SECOND_REPRESENTATIVE_SUBJECT is not None and SECOND_REPRESENTATIVE_FILE_LABEL is not None:
        try:
            fig = plot_roi_recovery(
                roi_timecourses=roi_timecourses,
                subject=SECOND_REPRESENTATIVE_SUBJECT,
                file_label=SECOND_REPRESENTATIVE_FILE_LABEL,
                pipelines=PIPELINE_ORDER,
                chrom=PRIMARY_CHROM,
            )
            save_fig(fig, f"roi_recovery_bc_{SECOND_REPRESENTATIVE_SUBJECT}_{SECOND_REPRESENTATIVE_FILE_LABEL}_{PRIMARY_CHROM}_representative2_v16")
            if SHOW_FIGURES:
                fig.show()
        except Exception as exc:
            failure_rows.append({
                "subject": SECOND_REPRESENTATIVE_SUBJECT,
                "file_label": SECOND_REPRESENTATIVE_FILE_LABEL,
                "reason": str(exc),
            })

    if PLOT_ALL_OFFICIAL_OVERLAYS:
        for _, row in official_overlay_inventory.iterrows():
            subject = str(row["subject"])
            file_label = str(row["file_label"])
            if rep_subject == subject and rep_file == file_label:
                continue
            try:
                fig = plot_roi_recovery(
                    roi_timecourses=roi_timecourses,
                    subject=subject,
                    file_label=file_label,
                    pipelines=PIPELINE_ORDER,
                    chrom=PRIMARY_CHROM,
                )
                save_fig(fig, f"roi_recovery_bc_{subject}_{file_label}_{PRIMARY_CHROM}_all_available_v15")
                if SHOW_FIGURES:
                    fig.show()
            except Exception as exc:
                failure_rows.append({
                    "subject": subject,
                    "file_label": file_label,
                    "reason": str(exc),
                })

    official_overlay_failures = pd.DataFrame(failure_rows)

save_table(official_overlay_failures, "official_overlay_failures_v7")

print("Official-overlay subject/file combinations:", len(official_overlay_inventory))
if len(official_overlay_failures) > 0:
    print("Official-overlay failures:", len(official_overlay_failures))
    display(official_overlay_failures)









# %%




# %% [markdown]
#      ## 15) Peak-amplitude ratio diagnostics
# 
# 
# 
#      Still diagnostic only. Uses ROI target-curve shape metrics.

# %%
peak_ratio_diagnostic = pd.DataFrame()
if INCLUDE_PEAK_RATIO_DIAGNOSTICS and len(roi_shape_subject_summary) > 0:
    peak_ratio_diagnostic = roi_shape_subject_summary[
        roi_shape_subject_summary["chromophore"] == PRIMARY_CHROM
    ].groupby("pipeline_label", as_index=False).agg(
        median_peak_amplitude_ratio=("mean_peak_amplitude_ratio", "median"),
        mean_peak_amplitude_ratio=("mean_peak_amplitude_ratio", "mean"),
        n_subjects=("subject", "nunique"),
    )
    save_table(peak_ratio_diagnostic, "peak_amplitude_ratio_diagnostic_v7")
    sane_mask = peak_ratio_diagnostic["median_peak_amplitude_ratio"].between(0.1, 10.0, inclusive="both")
    sane_fraction = float(np.mean(sane_mask)) if len(peak_ratio_diagnostic) > 0 else 0.0
    print("Peak-amplitude ratio sanity fraction (median ratio in [0.1, 10]):", sane_fraction)
    if sane_fraction >= 0.8:
        peak_ratio_subject = roi_shape_subject_summary[
            roi_shape_subject_summary["chromophore"] == PRIMARY_CHROM
        ][["subject", "pipeline_label", "mean_peak_amplitude_ratio"]].copy()
        peak_ratio_subject = peak_ratio_subject.rename(columns={"mean_peak_amplitude_ratio": "peak_amplitude_ratio"})
        peak_ratio_summary = summarize_with_bootstrap(
            peak_ratio_subject.rename(columns={"peak_amplitude_ratio": "value"}),
            "value",
            ["pipeline_label"],
        )
        peak_ratio_summary["pipeline_display"] = peak_ratio_summary["pipeline_label"].map(pretty_pipeline_label)
        peak_ratio_summary = peak_ratio_summary.sort_values("value_mean", ascending=False)
        save_table(peak_ratio_summary, "peak_amplitude_ratio_diagnostic_summary_v16")
        fig = go.Figure()
        fig.add_hline(y=1.0, line_dash="dash", line_color="black", line_width=1.2)
        fig.add_trace(
            go.Scatter(
                x=peak_ratio_summary["pipeline_display"].astype(str),
                y=peak_ratio_summary["value_mean"],
                mode="markers",
                marker=dict(size=11, color="#0072B2", line=dict(color="black", width=0.9)),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(peak_ratio_summary["value_ci_hi"] - peak_ratio_summary["value_mean"]).to_numpy(),
                    arrayminus=(peak_ratio_summary["value_mean"] - peak_ratio_summary["value_ci_lo"]).to_numpy(),
                    thickness=1.6,
                    width=0,
                    color="black",
                ),
                showlegend=False,
                hovertemplate="%{x}<br>Mean ratio=%{y:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Peak-amplitude ratio by pipeline",
            xaxis_title="Pipeline",
            yaxis_title="Peak-amplitude ratio",
        )
        fig.update_xaxes(
            categoryorder="array",
            categoryarray=peak_ratio_summary["pipeline_display"].astype(str).tolist(),
            tickangle=42,
        )
        style_figure(fig, title="Peak-amplitude ratio by pipeline")
        save_fig(fig, "peak_amplitude_ratio_diagnostic_pointrange_v16")
        if SHOW_FIGURES:
            fig.show()
    else:
        print("Skipping raw peak-amplitude ratio figure: unresolved scale mismatch across recovered and truth amplitudes.")









# %% [markdown]
#      ## 16) Pipeline performance synthesis heatmap
# 
# 
# 
#      Compact overview of the main benchmark tradeoffs across the key thesis metrics.

# %%
heatmap_summary = pd.DataFrame()
heatmap_long_rows = []

def _add_heatmap_metric(summary_df, metric_label, better="higher"):
    if summary_df is None or len(summary_df) == 0:
        return
    tmp = summary_df[["pipeline_label", "value_mean"]].copy()
    tmp["metric"] = metric_label
    tmp["better_direction"] = better
    heatmap_long_rows.extend(tmp.to_dict(orient="records"))

if len(shape_corr_subject) > 0:
    _add_heatmap_metric(shape_corr_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_curve_corr", "mean")), "Shape correlation", "higher")
if len(shape_nrmse_subject) > 0:
    _add_heatmap_metric(shape_nrmse_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_curve_nrmse", "mean")), "Normalized RMSE", "lower")
if len(shape_latency_subject) > 0:
    _add_heatmap_metric(shape_latency_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_abs_peak_latency_error_s", "mean")), "Peak-latency error", "lower")
if len(roi_peak_subject) > 0:
    _add_heatmap_metric(roi_peak_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_target_minus_non_target_peak", "mean")), "Target-minus-non-target peak", "higher")
if len(roi_peak_amp_resp_official) > 0:
    _add_heatmap_metric(roi_peak_amp_resp_official.groupby("pipeline_label", as_index=False).agg(value_mean=("slope", "mean")), "Dose-response slope", "higher")
if len(null_roi_subject) > 0:
    _add_heatmap_metric(null_roi_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_abs_null_response", "mean")), "Null response", "lower")
if len(active_null_discrimination) > 0:
    _add_heatmap_metric(active_null_discrimination.groupby("pipeline_label", as_index=False).agg(value_mean=("auroc", "mean")), "AUROC", "higher")
if len(retention_subject) > 0:
    _add_heatmap_metric(retention_subject.groupby("pipeline_label", as_index=False).agg(value_mean=("mean_retention_fraction", "mean")), "Retention", "higher")

if len(heatmap_long_rows) > 0:
    heatmap_long = pd.DataFrame(heatmap_long_rows)
    std_rows = []
    for metric, grp in heatmap_long.groupby("metric"):
        vals = grp["value_mean"].astype(float).copy()
        if grp["better_direction"].iloc[0] == "lower":
            vals = -vals
        mu = float(vals.mean())
        sd = float(vals.std(ddof=0))
        z = np.zeros(len(vals)) if (not np.isfinite(sd) or sd < 1e-15) else (vals - mu) / sd
        out = grp.copy()
        out["z_better"] = z
        std_rows.append(out)

    heatmap_summary = pd.concat(std_rows, ignore_index=True)
    heatmap_summary["pipeline_display"] = heatmap_summary["pipeline_label"].map(pretty_pipeline_label)
    save_table(heatmap_summary, "pipeline_performance_heatmap_summary_v15")

    heatmap_wide = heatmap_summary.pivot_table(index="pipeline_display", columns="metric", values="z_better", aggfunc="mean")
    heatmap_wide = heatmap_wide.reindex(index=ordered_pipeline_display_labels(heatmap_summary["pipeline_label"].astype(str).unique().tolist()))
    wanted = ["Shape correlation", "Normalized RMSE", "Peak-latency error", "Target-minus-non-target peak", "Dose-response slope", "Null response", "AUROC", "Retention"]
    heatmap_wide = heatmap_wide[[c for c in wanted if c in heatmap_wide.columns]]

    fig = px.imshow(
        heatmap_wide,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        origin="lower",
        title="Pipeline performance synthesis heatmap",
        labels=dict(x="Metric", y="Pipeline", color="Standardized performance"),
    )
    fig.update_traces(hovertemplate="Pipeline=%{y}<br>Metric=%{x}<br>Standardized performance=%{z:.2f}<extra></extra>")
    style_figure(fig, title="Pipeline performance synthesis heatmap")
    fig.update_xaxes(tickangle=25)
    fig.update_yaxes(automargin=True)
    save_fig(fig, "pipeline_performance_heatmap_v15")
    if SHOW_FIGURES:
        fig.show()


# %% [markdown]
#      ## 16) Supporting / appendix analyses

# %%
if INCLUDE_SUPPORTING_FIGURES:
    if len(channel_availability) > 0:
        completion = channel_availability.copy()
        completion["present"] = 1
        heatmap = completion.pivot_table(index="file_label", columns="pipeline_label", values="present", aggfunc="max", fill_value=0)
        heatmap = heatmap.reindex(columns=[p for p in PIPELINE_ORDER if p in heatmap.columns])
        fig = px.imshow(
            heatmap,
            aspect="auto",
            text_auto=True,
            title="Pipeline coverage across files",
            labels=dict(x="Pipeline", y="File", color="Present"),
        )
        fig.update_xaxes(tickangle=60)
        save_fig(fig, "completion_heatmap_files_by_pipeline_v7")
        if SHOW_FIGURES:
            fig.show()

        completion_subject = channel_availability.groupby(["subject", "pipeline_label"], as_index=False).agg(n_files=("file_label", "nunique"))
        completion_subject = apply_display_labels(completion_subject)
        save_table(completion_subject, "completion_subject_counts_v7")
        fig = px.box(
            completion_subject,
            x="pipeline_display",
            y="n_files",
            points="all",
            title="Completed files per subject by pipeline",
            labels={"pipeline_display": "Pipeline", "n_files": "Number of files"},
        )
        fig.update_xaxes(tickangle=60)
        save_fig(fig, "completion_subject_box_v7")
        if SHOW_FIGURES:
            fig.show()

    if len(standard_subject_level_primary) > 0:
        ranks = standard_subject_level_primary.copy()
        ranks["subject_rank"] = ranks.groupby("subject")["mean_target_minus_non_target_score"].rank(method="average", ascending=False)
        rank_summary = ranks.groupby("pipeline_label", as_index=False).agg(
            median_rank=("subject_rank", "median"),
            mean_rank=("subject_rank", "mean"),
            std_rank=("subject_rank", "std"),
            top1_count=("subject_rank", lambda x: int(np.sum(np.asarray(x) == 1))),
            top3_count=("subject_rank", lambda x: int(np.sum(np.asarray(x) <= 3))),
            n_subjects=("subject", "nunique"),
        ).sort_values("median_rank")
        rank_summary["top1_fraction"] = rank_summary["top1_count"] / rank_summary["n_subjects"]
        rank_summary["top3_fraction"] = rank_summary["top3_count"] / rank_summary["n_subjects"]
        save_table(rank_summary, "rank_consistency_standard_primary_v7")
        fig = px.bar(
            rank_summary,
            x="pipeline_label",
            y="median_rank",
            title="Median within-subject pipeline rank for the primary endpoint",
            labels={"pipeline_label": "Pipeline", "median_rank": "Median rank"},
        )
        fig.update_xaxes(tickangle=60)
        save_fig(fig, "rank_consistency_standard_median_rank_v7")
        if SHOW_FIGURES:
            fig.show()

        rank_pivot = ranks.pivot_table(index="subject", columns="pipeline_label", values="subject_rank")
        rank_pivot = rank_pivot[[c for c in STANDARD_PIPELINES if c in rank_pivot.columns]]
        if len(rank_pivot.dropna()) >= 2:
            W = kendalls_w_from_rank_matrix(rank_pivot.dropna().values)
            print("Kendall's W for subject-level rank agreement (standard family):", W)
            pd.DataFrame([{"metric": "standard_primary_rank_agreement", "kendalls_w": W}]).to_csv(EXPORT_DIR / "kendalls_w_rank_agreement_v7.csv", index=False)

    if HAVE_SKLEARN and len(canonical) > 0:
        classification_rows = []
        tmp = canonical[(canonical["active_vs_null"] == "active") & (canonical["pipeline_label"].isin(STANDARD_PIPELINES))].copy()
        if len(tmp) > 0:
            tmp["y_true"] = tmp["target_status"].map(lambda x: 1 if standardize_target_status(x) == "true_target" else 0)
            tmp["score_oriented"] = [orient_score(pd.Series([beta]), chrom).iloc[0] for beta, chrom in zip(tmp["beta"], tmp["chromophore"])]
            for keys, grp in tmp.groupby(["subject", "file_label", "pipeline_label", "chromophore"]):
                if grp["y_true"].nunique() < 2:
                    continue
                classification_rows.append(
                    {
                        "subject": keys[0],
                        "file_label": keys[1],
                        "pipeline_label": keys[2],
                        "chromophore": keys[3],
                        "metric_source": "canonical_beta",
                        "auroc": float(roc_auc_score(grp["y_true"], grp["score_oriented"])),
                        "auprc": float(average_precision_score(grp["y_true"], grp["score_oriented"])),
                        "n_channels": int(len(grp)),
                    }
                )
        classification_df = pd.DataFrame(classification_rows)
        save_table(classification_df, "channel_level_classification_metrics_standard_v7")
        if len(classification_df) > 0:
            cls_subject_level = classification_df.groupby(["subject", "pipeline_label", "chromophore", "metric_source"], as_index=False).agg(
                mean_auroc=("auroc", "mean"),
                mean_auprc=("auprc", "mean"),
            )
            save_table(cls_subject_level, "channel_level_classification_subject_summary_standard_v7")
            cls_hbo = cls_subject_level[(cls_subject_level["chromophore"] == PRIMARY_CHROM) & (cls_subject_level["metric_source"] == "canonical_beta")].copy()
            if len(cls_hbo) > 0:
                pipeline_pointrange_from_subject_level(
                    cls_hbo,
                    value_col="mean_auroc",
                    title="Channel-level AUROC by pipeline (canonical beta, HbO; standard family)",
                    y_label="AUROC",
                    name="classification_auroc_standard_pointrange_v16",
                    ascending=False,
                )

    if len(variability_summary) > 0:
        keep = variability_summary[
            (variability_summary["subset_name"] == "all_pipelines") &
            (variability_summary["score_type"] == PRIMARY_SCORE_TYPE) &
            (variability_summary["chromophore"] == PRIMARY_CHROM)
        ].copy()
        if "active_vs_null" in keep.columns:
            keep = keep[keep["active_vs_null"] == "active"].copy()
        if len(keep) > 0:
            fig = px.histogram(
                keep,
                x="std_across_pipelines",
                nbins=30,
                title="Across-pipeline variability for the primary endpoint",
                labels={"std_across_pipelines": "Standard deviation across pipelines"},
            )
            save_fig(fig, "variability_sd_histogram_v7")
            if SHOW_FIGURES:
                fig.show()
            save_table(keep, "variability_summary_primary_active_v7")

    if len(pairwise_pipeline_deltas) > 0:
        pdel = pairwise_pipeline_deltas[(pairwise_pipeline_deltas["score_type"] == PRIMARY_SCORE_TYPE) & (pairwise_pipeline_deltas["chromophore"] == PRIMARY_CHROM)].copy()
        if len(pdel) > 0:
            pairwise_summary = pdel.groupby(["left_pipeline", "right_pipeline"], as_index=False).agg(
                mean_abs_delta=("abs_left_minus_right", "mean"),
                median_abs_delta=("abs_left_minus_right", "median"),
                n_rows=("subject", "count"),
            )
            save_table(pairwise_summary, "pairwise_pipeline_delta_summary_primary_v7")

print("Exports written to:", EXPORT_DIR)
print("Figures directory:", FIG_DIR)
print("Done.")


















