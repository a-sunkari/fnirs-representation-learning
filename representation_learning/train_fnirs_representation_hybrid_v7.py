#!/usr/bin/env python3
"""
Hybrid multiview fNIRS representation-learning trainer.

Core idea
---------
One sample = one subject + one file_label condition.
Each sample has many pipeline views.
Each view is a 4-channel timecourse:
  [target HbO, target HbR, non-target HbO, non-target HbR]

Model design
------------
1) Shared multiview timecourse encoder over pipeline-specific timecourses.
2) Handcrafted feature branch built from the same sample.
3) Fusion of learned invariant embedding + handcrafted feature embedding.
4) Multitask heads:
   - HRF present vs null
   - continuous amplitude regression
   - amplitude-tier auxiliary classification on positive samples
5) Cross-pipeline consistency loss is retained, but applied only to the multiview branch.

Why this version exists
-----------------------
The earlier pure multiview model produced reasonably pipeline-invariant embeddings,
but underperformed a simple classical baseline for detection. This version keeps
the representation-learning core while explicitly incorporating the strong
handcrafted signal into the neural model itself.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
except Exception as exc:
    raise SystemExit(
        "pyarrow is required to read parquet efficiently. Activate the right environment.\n"
        f"Import error: {exc}"
    )

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    cohen_kappa_score,
    brier_score_loss,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

CHANNEL_ORDER = [
    ("true_target", "hbo"),
    ("true_target", "hbr"),
    ("true_non_target", "hbo"),
    ("true_non_target", "hbr"),
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class _GradScaleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None

def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    if float(scale) == 1.0:
        return x
    if float(scale) == 0.0:
        return x.detach()
    return _GradScaleFn.apply(x, float(scale))

def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pipeline_drop_default(name: str) -> bool:
    low = name.lower()
    return ("fir" in low) or ("modgamma" in low)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def dedupe_time_signal(t: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t = np.asarray(t)[order]
    y = np.asarray(y)[order]
    if len(t) == 0:
        return t, y
    uniq_t, inv = np.unique(t, return_inverse=True)
    if len(uniq_t) == len(t):
        return t, y
    sums = np.zeros(len(uniq_t), dtype=np.float64)
    counts = np.zeros(len(uniq_t), dtype=np.int64)
    for i, idx in enumerate(inv):
        sums[idx] += y[i]
        counts[idx] += 1
    return uniq_t, sums / np.maximum(counts, 1)

def interpolate_channel(t: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    t, y = dedupe_time_signal(t, y)
    if len(t) < 2:
        raise ValueError("Not enough timepoints to interpolate")
    return np.interp(grid, t, y, left=y[0], right=y[-1]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hybrid pipeline-invariant fNIRS model from aggregated parquet.")
    p.add_argument("--parquet", type=str, required=True, help="roi_timecourses_aggregated.parquet")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--cache-name", type=str, default="tensor_cache.npz", help="Name for cached tensor dataset")
    p.add_argument("--cache-dir", type=str, default="", help="Optional directory for tensor cache; defaults to output-dir")
    p.add_argument("--time-start", type=float, default=0.0, help="Common grid start (seconds)")
    p.add_argument("--time-end", type=float, default=20.0, help="Common grid end (seconds)")
    p.add_argument("--n-timepoints", type=int, default=101, help="Common grid size")
    p.add_argument("--min-views-per-sample", type=int, default=4, help="Drop samples with fewer valid pipeline views")
    p.add_argument("--min-samples-per-pipeline", type=int, default=10, help="Drop pipelines with too few complete samples")
    p.add_argument("--drop-pipeline-substring", action="append", default=[], help="Optional extra pipeline substring to drop")

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--hidden-channels", type=int, nargs="+", default=[32, 64, 64])
    p.add_argument("--feature-hidden-dim", type=int, default=32)
    p.add_argument("--task-hidden-dim", type=int, default=64)
    p.add_argument("--consistency-proj-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--view-dropout", type=float, default=0.0, help="Randomly hide some valid views during training")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Number of initial epochs to train only the detection side (plus consistency) before turning on amplitude/severity losses.")
    p.add_argument("--severity-shared-grad-scale", type=float, default=1.0,
                   help="Scale factor for how much amplitude/severity gradients are allowed to flow back into the shared fused representation. 1.0 = fully shared, 0.0 = fully detached.")
    p.add_argument("--tower-orth-weight", type=float, default=0.0,
                   help="Penalty weight encouraging the presence and severity task embeddings to use different directions.")

    p.add_argument("--consistency-weight", type=float, default=0.05)
    p.add_argument("--amplitude-weight", type=float, default=1.0)
    p.add_argument("--presence-weight", type=float, default=1.0)
    p.add_argument("--tier-weight", type=float, default=0.25)
    p.add_argument("--num-amp-tiers", type=int, default=4, help="Target number of positive-sample amplitude tiers")
    p.add_argument("--severity-head", type=str, default="cumulative_link", choices=["cumulative_link", "flat_ce", "soft_label_ce"],
                   help="Auxiliary severity head. cumulative_link is rank-aware; flat_ce is ordinary multiclass; soft_label_ce uses neighboring-class soft targets.")
    p.add_argument("--amp-regression-loss", type=str, default="heteroscedastic", choices=["smooth_l1", "heteroscedastic", "beta_nll"],
                   help="Amplitude regression loss. heteroscedastic predicts a mean and input-dependent log-variance; beta_nll is a stabilized heteroscedastic variant.")
    p.add_argument("--task-weighting", type=str, default="uncertainty", choices=["manual", "uncertainty"],
                   help="How to combine presence, amplitude, ordinal, and consistency losses.")
    p.add_argument("--amp-logvar-min", type=float, default=-4.0, help="Minimum clamp for amplitude log-variance.")
    p.add_argument("--amp-logvar-max", type=float, default=2.0, help="Maximum clamp for amplitude log-variance.")
    p.add_argument("--severity-target", type=str, default="fixed_amp_classes",
                   choices=["fixed_amp_classes", "quantile_tiers"],
                   help="How to define the ordered severity target. fixed_amp_classes uses the real ordered amplitude labels; quantile_tiers reproduces the older fold-specific tiering.")
    p.add_argument("--fixed-amp-class-values", type=float, nargs="*", default=None,
                   help="Optional explicit ordered positive amplitude values, e.g. 20 30 40 45 50 60 75 85 100. If omitted, unique positive amplitudes in the parquet are used.")
    p.add_argument("--full-eval-amp-values", type=float, nargs="*", default=None,
                   help="Optional explicit ordered amplitude values used only for the common snapped full-level evaluation across all variants. If omitted, unique positive amplitudes in the dataset are used.")
    p.add_argument("--low-blunted-threshold", type=float, default=50.0,
                   help="Threshold in raw amplitude units for low/blunted vs normal/high reporting. Values below this are treated as low/blunted.")
    p.add_argument("--amp-prediction-mode", type=str, default="direct", choices=["direct", "class_expectation_residual"],
                   help="direct uses the regression head directly. class_expectation_residual anchors the prediction to the class-probability expected amplitude and adds a bounded residual.")
    p.add_argument("--residual-limit-raw", type=float, default=15.0,
                   help="Maximum absolute residual in raw amplitude units when using class_expectation_residual mode.")
    p.add_argument("--soft-label-sigma", type=float, default=1.0,
                   help="Standard deviation in class-index units for soft_label_ce targets.")
    p.add_argument("--rank-weight", type=float, default=0.0,
                   help="Weight for pairwise ranking loss on positive-sample amplitude ordering.")
    p.add_argument("--rank-margin-scaled", type=float, default=0.0,
                   help="Optional scaled margin for the ranking loss. 0 uses logistic ranking without a hard margin.")
    p.add_argument("--beta-nll-beta", type=float, default=0.5,
                   help="Beta parameter for beta-NLL amplitude regression. Only used when amp-regression-loss=beta_nll.")
    p.add_argument("--severity-class-balance", type=str, default="inverse_freq", choices=["none", "inverse_freq"],
                   help="How to weight auxiliary severity classes during training. inverse_freq can help when coarse bins are imbalanced.")

    p.add_argument("--folds", type=int, default=4, help="Grouped CV folds across subjects")
    p.add_argument("--single-fold-only", type=int, default=-1, help="Run only one fold index for quick smoke testing")
    p.add_argument("--val-frac-within-train", type=float, default=0.2, help="GroupShuffleSplit val fraction inside each train fold")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()



# ---------------------------------------------------------------------
# Build tensor dataset from aggregated parquet
# ---------------------------------------------------------------------

def build_tensor_cache(
    parquet_path: str,
    cache_npz: str,
    cache_meta_json: str,
    time_start: float,
    time_end: float,
    n_timepoints: int,
    min_views_per_sample: int,
    min_samples_per_pipeline: int,
    drop_pipeline_substrings: List[str],
) -> Dict:
    grid = np.linspace(time_start, time_end, n_timepoints, dtype=np.float32)
    dataset = ds.dataset(parquet_path, format="parquet")

    needed = [
        "subject",
        "file_label",
        "pipeline_norm",
        "chromophore",
        "target_status",
        "time_s",
        "signal",
        "amplitude_value",
    ]
    scanner = dataset.scanner(columns=needed, batch_size=250_000)

    curve_parts: Dict[Tuple[str, str, str, str], List[Tuple[np.ndarray, np.ndarray]]] = {}
    sample_meta: Dict[str, Dict] = {}

    for batch_idx, rb in enumerate(scanner.to_batches(), start=1):
        df = rb.to_pandas()
        df["subject"] = df["subject"].map(norm_text)
        df["file_label"] = df["file_label"].map(norm_text)
        df["pipeline_norm"] = df["pipeline_norm"].map(norm_text)
        df["chromophore"] = df["chromophore"].str.lower()
        df["target_status"] = df["target_status"].map(norm_text)
        df["sample_id"] = df["subject"] + "||" + df["file_label"]

        keep = ~df["pipeline_norm"].str.lower().apply(pipeline_drop_default)
        for s in drop_pipeline_substrings:
            if s:
                keep &= ~df["pipeline_norm"].str.lower().str.contains(re.escape(s.lower()), regex=True)
        df = df.loc[keep].copy()
        if df.empty:
            continue

        df = df.loc[
            df["target_status"].isin({"true_target", "true_non_target"})
            & df["chromophore"].isin({"hbo", "hbr"})
        ].copy()
        if df.empty:
            continue

        meta_df = df[["sample_id", "subject", "file_label", "amplitude_value"]].drop_duplicates()
        for row in meta_df.itertuples(index=False):
            if row.sample_id not in sample_meta:
                sample_meta[row.sample_id] = {
                    "subject": row.subject,
                    "file_label": row.file_label,
                    "amplitude_value": float(row.amplitude_value),
                }

        gb = df.groupby(["sample_id", "pipeline_norm", "target_status", "chromophore"], sort=False)
        for key, g in gb:
            t = g["time_s"].to_numpy(dtype=np.float32, copy=False)
            y = g["signal"].to_numpy(dtype=np.float32, copy=False)
            curve_parts.setdefault(key, []).append((t, y))

        if batch_idx % 20 == 0:
            print(f"[build] processed {batch_idx} batches; curves so far={len(curve_parts):,}", flush=True)

    required_channel_keys = set(CHANNEL_ORDER)
    pipeline_sample_complete: Dict[str, int] = {}
    sample_pipeline_channels: Dict[Tuple[str, str], set] = {}

    for (sample_id, pipeline, target_status, chrom), parts in curve_parts.items():
        sample_pipeline_channels.setdefault((sample_id, pipeline), set()).add((target_status, chrom))

    for (sample_id, pipeline), chset in sample_pipeline_channels.items():
        if chset == required_channel_keys:
            pipeline_sample_complete[pipeline] = pipeline_sample_complete.get(pipeline, 0) + 1

    pipeline_names = sorted([p for p, n in pipeline_sample_complete.items() if n >= min_samples_per_pipeline])
    pipeline_to_idx = {p: i for i, p in enumerate(pipeline_names)}
    print(f"[build] kept {len(pipeline_names)} pipelines after coverage filter >= {min_samples_per_pipeline}", flush=True)

    X_list = []
    mask_list = []
    presence_list = []
    amp_raw_list = []
    subject_list = []
    file_list = []
    feature_list = []

    dropped_too_few_views = 0
    incomplete_groups = 0

    for sample_id in sorted(sample_meta.keys()):
        meta = sample_meta[sample_id]
        subject = meta["subject"]
        file_label = meta["file_label"]
        amp_raw = float(meta["amplitude_value"])
        presence = 1.0 if amp_raw > 0 else 0.0

        x = np.zeros((len(pipeline_names), 4, len(grid)), dtype=np.float32)
        m = np.zeros((len(pipeline_names),), dtype=np.float32)

        for pipeline in pipeline_names:
            pidx = pipeline_to_idx[pipeline]
            ok = True
            chans = []
            for target_status, chrom in CHANNEL_ORDER:
                key = (sample_id, pipeline, target_status, chrom)
                if key not in curve_parts:
                    ok = False
                    break
                parts = curve_parts[key]
                t = np.concatenate([a for a, _ in parts]).astype(np.float32, copy=False)
                y = np.concatenate([b for _, b in parts]).astype(np.float32, copy=False)
                try:
                    chans.append(interpolate_channel(t, y, grid))
                except Exception:
                    ok = False
                    break
            if ok and len(chans) == 4:
                x[pidx] = np.stack(chans, axis=0)
                m[pidx] = 1.0
            else:
                incomplete_groups += 1

        if int(m.sum()) < min_views_per_sample:
            dropped_too_few_views += 1
            continue

        xm = x[m.astype(bool)].mean(axis=0)
        tt_hbo, tt_hbr, nt_hbo, nt_hbr = xm
        diff_hbo = tt_hbo - nt_hbo
        diff_hbr = tt_hbr - nt_hbr
        feats = np.array([
            tt_hbo.max(), tt_hbo.mean(), tt_hbo.sum(),
            tt_hbr.min(), tt_hbr.mean(), tt_hbr.sum(),
            nt_hbo.max(), nt_hbo.mean(), nt_hbo.sum(),
            nt_hbr.min(), nt_hbr.mean(), nt_hbr.sum(),
            diff_hbo.max(), diff_hbo.mean(), diff_hbo.sum(),
            diff_hbr.min(), diff_hbr.mean(), diff_hbr.sum(),
            grid[np.argmax(tt_hbo)], grid[np.argmin(tt_hbr)],
        ], dtype=np.float32)

        X_list.append(x)
        mask_list.append(m)
        presence_list.append(presence)
        amp_raw_list.append(amp_raw)
        subject_list.append(subject)
        file_list.append(file_label)
        feature_list.append(feats)

    if not X_list:
        raise RuntimeError("No samples survived cache building. Relax min_views_per_sample or check parquet.")

    X = np.stack(X_list, axis=0)
    view_mask = np.stack(mask_list, axis=0)
    y_presence = np.asarray(presence_list, dtype=np.float32)
    y_amp_raw = np.asarray(amp_raw_list, dtype=np.float32)
    amp_scale = max(1.0, float(y_amp_raw.max()))
    y_amp_scaled = y_amp_raw / amp_scale
    baseline_features = np.stack(feature_list, axis=0)

    np.savez_compressed(
        cache_npz,
        X=X,
        view_mask=view_mask,
        y_presence=y_presence,
        y_amp_raw=y_amp_raw,
        y_amp_scaled=y_amp_scaled,
        baseline_features=baseline_features,
        grid=grid,
        pipeline_names=np.asarray(pipeline_names, dtype=object),
        sample_ids=np.asarray([f"{s}||{f}" for s, f in zip(subject_list, file_list)], dtype=object),
        subjects=np.asarray(subject_list, dtype=object),
        file_labels=np.asarray(file_list, dtype=object),
        amp_scale=np.asarray([amp_scale], dtype=np.float32),
    )
    with open(cache_meta_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_samples": int(len(X)),
                "n_pipelines": int(len(pipeline_names)),
                "grid_start": float(time_start),
                "grid_end": float(time_end),
                "n_timepoints": int(n_timepoints),
                "min_views_per_sample": int(min_views_per_sample),
                "min_samples_per_pipeline": int(min_samples_per_pipeline),
                "dropped_too_few_views": int(dropped_too_few_views),
                "pipeline_names": pipeline_names,
                "pipeline_complete_sample_counts": pipeline_sample_complete,
                "incomplete_groups_count": int(incomplete_groups),
            },
            f,
            indent=2,
        )

    return {
        "X": X,
        "view_mask": view_mask,
        "y_presence": y_presence,
        "y_amp_raw": y_amp_raw,
        "y_amp_scaled": y_amp_scaled,
        "baseline_features": baseline_features,
        "grid": grid,
        "pipeline_names": pipeline_names,
        "sample_ids": np.asarray([f"{s}||{f}" for s, f in zip(subject_list, file_list)], dtype=object),
        "subjects": np.asarray(subject_list, dtype=object),
        "file_labels": np.asarray(file_list, dtype=object),
        "amp_scale": amp_scale,
    }

def load_or_build_cache(args: argparse.Namespace) -> Dict:
    outdir = Path(args.output_dir)
    ensure_dir(outdir)
    cache_root = Path(args.cache_dir) if args.cache_dir else outdir
    ensure_dir(cache_root)
    cache_npz = cache_root / args.cache_name
    cache_meta = cache_root / (Path(args.cache_name).stem + "_meta.json")

    if cache_npz.exists():
        arr = np.load(cache_npz, allow_pickle=True)
        data = {k: arr[k] for k in arr.files}
        data["pipeline_names"] = data["pipeline_names"].tolist()
        data["amp_scale"] = float(np.asarray(data["amp_scale"]).reshape(-1)[0])
        return data

    return build_tensor_cache(
        parquet_path=args.parquet,
        cache_npz=str(cache_npz),
        cache_meta_json=str(cache_meta),
        time_start=args.time_start,
        time_end=args.time_end,
        n_timepoints=args.n_timepoints,
        min_views_per_sample=args.min_views_per_sample,
        min_samples_per_pipeline=args.min_samples_per_pipeline,
        drop_pipeline_substrings=args.drop_pipeline_substring,
    )


# ---------------------------------------------------------------------
# Dataset + normalization
# ---------------------------------------------------------------------

def fit_pipeline_normalizer(X: np.ndarray, view_mask: np.ndarray, train_idx: np.ndarray) -> Dict[str, np.ndarray]:
    _, P, C, _ = X.shape
    mean = np.zeros((P, C, 1), dtype=np.float32)
    std = np.ones((P, C, 1), dtype=np.float32)

    for p in range(P):
        sel = train_idx[view_mask[train_idx, p] > 0.5]
        if len(sel) == 0:
            continue
        vals = X[sel, p]
        mean[p, :, 0] = vals.mean(axis=(0, 2))
        s = vals.std(axis=(0, 2))
        std[p, :, 0] = np.where(s < 1e-6, 1.0, s)
    return {"mean": mean, "std": std}

def apply_pipeline_normalizer(X: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    return ((X - norm["mean"][None, ...]) / norm["std"][None, ...]).astype(np.float32)

def fit_feature_scaler(features: np.ndarray, train_idx: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features[train_idx])
    return scaler

def build_amp_targets(
    y_amp_raw: np.ndarray,
    train_idx: np.ndarray,
    severity_target: str,
    target_num_tiers: int,
    fixed_amp_class_values: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    targets = np.full(len(y_amp_raw), -1, dtype=np.int64)

    if severity_target == "fixed_amp_classes":
        if fixed_amp_class_values:
            target_values = np.asarray(sorted(set(float(v) for v in fixed_amp_class_values if float(v) > 0)), dtype=np.float32)
        else:
            target_values = np.asarray(sorted(set(float(v) for v in y_amp_raw if float(v) > 0)), dtype=np.float32)

        if len(target_values) == 0:
            return targets, np.asarray([], dtype=np.float32)

        lookup = {round(float(v), 6): i for i, v in enumerate(target_values)}
        pos_mask = y_amp_raw > 0
        for idx in np.where(pos_mask)[0]:
            key = round(float(y_amp_raw[idx]), 6)
            if key in lookup:
                targets[idx] = int(lookup[key])
            else:
                targets[idx] = int(np.argmin(np.abs(target_values - float(y_amp_raw[idx]))))
        return targets, target_values.astype(np.float32)

    pos_train = y_amp_raw[train_idx][y_amp_raw[train_idx] > 0]
    if len(pos_train) < 2 or target_num_tiers <= 1:
        return targets, np.asarray([], dtype=np.float32)

    quantiles = np.linspace(0.0, 1.0, target_num_tiers + 1)[1:-1]
    edges = np.quantile(pos_train, quantiles).astype(np.float32)
    edges = np.unique(np.round(edges, 6))
    if len(edges) == 0:
        return targets, np.asarray([], dtype=np.float32)

    pos_mask = y_amp_raw > 0
    targets[pos_mask] = np.digitize(y_amp_raw[pos_mask], edges, right=True).astype(np.int64)
    return targets, edges.astype(np.float32)

class FnirsSampleDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        view_mask: np.ndarray,
        baseline_features: np.ndarray,
        y_presence: np.ndarray,
        y_amp_scaled: np.ndarray,
        y_amp_raw: np.ndarray,
        y_amp_tier: np.ndarray,
        idx: np.ndarray,
    ):
        self.X = X[idx]
        self.view_mask = view_mask[idx]
        self.baseline_features = baseline_features[idx]
        self.y_presence = y_presence[idx]
        self.y_amp_scaled = y_amp_scaled[idx]
        self.y_amp_raw = y_amp_raw[idx]
        self.y_amp_tier = y_amp_tier[idx]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.tensor(self.X[i], dtype=torch.float32),
            "view_mask": torch.tensor(self.view_mask[i], dtype=torch.float32),
            "baseline_features": torch.tensor(self.baseline_features[i], dtype=torch.float32),
            "presence": torch.tensor(self.y_presence[i], dtype=torch.float32),
            "amp_scaled": torch.tensor(self.y_amp_scaled[i], dtype=torch.float32),
            "amp_raw": torch.tensor(self.y_amp_raw[i], dtype=torch.float32),
            "amp_tier": torch.tensor(self.y_amp_tier[i], dtype=torch.long),
        }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = y + identity
        return F.gelu(y)

class ViewEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: List[int], embedding_dim: int, dropout: float):
        super().__init__()
        layers = []
        chs = [in_channels] + list(hidden_channels)
        for i in range(len(chs) - 1):
            layers.append(ResidualBlock1D(chs[i], chs[i + 1], kernel_size=5, dilation=2 ** i, dropout=dropout))
        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(chs[-1], chs[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(chs[-1], embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h).squeeze(-1)
        return self.proj(h)

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CumulativeLinkHead(nn.Module):
    """
    Ordered-threshold ordinal head.
    Produces K-1 logits corresponding to P(y > k).
    Thresholds are parameterized via a base threshold plus positive increments,
    which keeps them ordered by construction.
    """
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = max(1, int(num_classes))
        self.score = nn.Linear(in_dim, 1)
        if self.num_classes > 1:
            self.theta0 = nn.Parameter(torch.zeros(1))
            self.delta_unconstrained = nn.Parameter(torch.zeros(self.num_classes - 2)) if self.num_classes > 2 else None
        else:
            self.theta0 = None
            self.delta_unconstrained = None

    def thresholds(self) -> Optional[torch.Tensor]:
        if self.num_classes <= 1:
            return None
        if self.num_classes == 2:
            return self.theta0.view(1)
        deltas = F.softplus(self.delta_unconstrained) + 1e-4
        th = torch.cat([self.theta0.view(1), self.theta0 + torch.cumsum(deltas, dim=0)], dim=0)
        return th

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        score = self.score(x).squeeze(-1)
        if self.num_classes <= 1:
            logits = score.new_zeros((score.shape[0], 1))
        else:
            th = self.thresholds().to(score.device)
            logits = score.unsqueeze(1) - th.unsqueeze(0)
        return {"score": score, "logits": logits}

def cumulative_link_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes <= 1:
        return y.new_zeros((y.shape[0], 1), dtype=torch.float32)
    ks = torch.arange(num_classes - 1, device=y.device).view(1, -1)
    return (y.view(-1, 1) > ks).float()


def soft_label_targets(y: torch.Tensor, num_classes: int, sigma: float) -> torch.Tensor:
    if num_classes <= 1:
        return y.new_ones((y.shape[0], 1), dtype=torch.float32)
    sigma = float(max(sigma, 1e-4))
    ks = torch.arange(num_classes, device=y.device, dtype=torch.float32).view(1, -1)
    yc = y.view(-1, 1).float()
    logits = -0.5 * ((ks - yc) / sigma) ** 2
    probs = torch.softmax(logits, dim=1)
    return probs


def inverse_frequency_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    weights = np.ones(num_classes, dtype=np.float32)
    valid = y[y >= 0]
    if len(valid) == 0 or num_classes <= 1:
        return weights
    counts = np.bincount(valid.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return weights.astype(np.float32)


class MultiTaskUncertaintyWeighter(nn.Module):
    """
    Learnable task weighting following the homoscedastic uncertainty idea from
    Kendall et al. Each task gets a trainable log-variance parameter. We apply
    these weights on top of the user-provided base task weights.
    """
    def __init__(self):
        super().__init__()
        self.log_vars = nn.ParameterDict({
            "presence": nn.Parameter(torch.zeros(1)),
            "amp": nn.Parameter(torch.zeros(1)),
            "tier": nn.Parameter(torch.zeros(1)),
            "consistency": nn.Parameter(torch.zeros(1)),
        })

    def forward(self, weighted_losses: Dict[str, torch.Tensor], active_tasks: Optional[Dict[str, bool]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = None
        details = {}
        for name, loss in weighted_losses.items():
            is_active = True if active_tasks is None else bool(active_tasks.get(name, True))
            if not is_active:
                term = loss.new_tensor(0.0)
                details[name] = loss.new_tensor(0.0)
            else:
                log_var = self.log_vars[name]
                term = torch.exp(-log_var) * loss + log_var
                details[name] = torch.exp(-log_var).detach()
            total = term if total is None else total + term
        return total, details


class HybridMultiviewFnirsNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        baseline_feat_dim: int,
        hidden_channels: List[int],
        embedding_dim: int,
        feature_hidden_dim: int,
        task_hidden_dim: int,
        consistency_proj_dim: int,
        dropout: float,
        num_amp_tiers: int,
        severity_head: str,
        amp_regression_loss: str,
        task_weighting: str,
        amp_prediction_mode: str = "direct",
        class_values_scaled: Optional[np.ndarray] = None,
        residual_limit_scaled: float = 0.15,
        severity_shared_grad_scale: float = 1.0,
    ):
        super().__init__()
        self.severity_head_type = severity_head
        self.amp_regression_loss = amp_regression_loss
        self.task_weighting = task_weighting
        self.num_amp_tiers = max(1, int(num_amp_tiers))
        self.amp_prediction_mode = amp_prediction_mode
        self.residual_limit_scaled = float(residual_limit_scaled)
        self.severity_shared_grad_scale = float(severity_shared_grad_scale)
        if class_values_scaled is not None and len(class_values_scaled):
            self.register_buffer("class_values_scaled", torch.tensor(class_values_scaled, dtype=torch.float32))
        else:
            self.class_values_scaled = None

        self.encoder = ViewEncoder(in_channels, hidden_channels, embedding_dim, dropout)
        self.feature_encoder = FeatureEncoder(baseline_feat_dim, feature_hidden_dim, embedding_dim, dropout)

        self.view_score = nn.Linear(embedding_dim, 1)
        self.consistency_proj = nn.Sequential(
            nn.Linear(embedding_dim, consistency_proj_dim),
            nn.LayerNorm(consistency_proj_dim),
            nn.GELU(),
            nn.Linear(consistency_proj_dim, consistency_proj_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.presence_adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )
        self.severity_adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        self.presence_tower = MLPBlock(embedding_dim, task_hidden_dim, embedding_dim, dropout)
        self.severity_tower = MLPBlock(embedding_dim, task_hidden_dim, embedding_dim, dropout)

        self.presence_head = nn.Linear(embedding_dim, 1)
        if self.amp_regression_loss in {"heteroscedastic", "beta_nll"}:
            self.amplitude_head = nn.Linear(embedding_dim, 2)  # mean, logvar
        else:
            self.amplitude_head = nn.Linear(embedding_dim, 1)

        if self.severity_head_type == "cumulative_link":
            self.ordinal_head = CumulativeLinkHead(embedding_dim, self.num_amp_tiers)
        else:
            self.ordinal_head = nn.Linear(embedding_dim, self.num_amp_tiers)

        self.loss_weighter = MultiTaskUncertaintyWeighter() if self.task_weighting == "uncertainty" else None

    def masked_attention_pool(
        self,
        z: torch.Tensor,
        view_mask: torch.Tensor,
        training: bool,
        view_dropout: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled_mask = view_mask.clone()

        if training and view_dropout > 0:
            keep = torch.bernoulli(torch.full_like(pooled_mask, 1.0 - view_dropout))
            pooled_mask = pooled_mask * keep
            none_kept = pooled_mask.sum(dim=1) < 0.5
            if none_kept.any():
                pooled_mask[none_kept] = view_mask[none_kept]

        logits = self.view_score(z).squeeze(-1)
        logits = logits.masked_fill(pooled_mask < 0.5, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * pooled_mask
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
        pooled = torch.sum(weights.unsqueeze(-1) * z, dim=1)
        return pooled, weights, pooled_mask

    def forward(
        self,
        x: torch.Tensor,
        baseline_features: torch.Tensor,
        view_mask: torch.Tensor,
        training: bool = False,
        view_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        bsz, n_views, n_ch, tlen = x.shape
        z_views = self.encoder(x.view(bsz * n_views, n_ch, tlen)).view(bsz, n_views, -1)
        z_pooled, attn, pooled_mask = self.masked_attention_pool(z_views, view_mask, training=training, view_dropout=view_dropout)
        z_views_cons = self.consistency_proj(z_views)
        z_pooled_cons = self.consistency_proj(z_pooled)

        h_feat = self.feature_encoder(baseline_features)
        z_fused = self.fusion(torch.cat([z_pooled, h_feat], dim=-1))

        z_presence_in = self.presence_adapter(z_fused)
        z_severity_in = self.severity_adapter(scale_gradient(z_fused, self.severity_shared_grad_scale))

        z_presence = self.presence_tower(z_presence_in)
        z_severity = self.severity_tower(z_severity_in)

        presence_logit = self.presence_head(z_presence).squeeze(-1)

        amp_raw_out = self.amplitude_head(z_severity)
        if self.amp_regression_loss in {"heteroscedastic", "beta_nll"}:
            amp_base = amp_raw_out[:, 0]
            amp_logvar = amp_raw_out[:, 1]
        else:
            amp_base = amp_raw_out.squeeze(-1)
            amp_logvar = torch.zeros_like(amp_base)

        if self.severity_head_type == "cumulative_link":
            ordinal_out = self.ordinal_head(z_severity)
            amp_tier_logit = ordinal_out["logits"]
            amp_tier_score = ordinal_out["score"]
            class_probs = None
        else:
            amp_tier_logit = self.ordinal_head(z_severity)
            amp_tier_score = amp_base
            class_probs = torch.softmax(amp_tier_logit, dim=1) if amp_tier_logit.shape[1] > 1 else None

        class_expectation = None
        if (
            self.amp_prediction_mode == "class_expectation_residual"
            and class_probs is not None
            and self.class_values_scaled is not None
            and amp_tier_logit.shape[1] == len(self.class_values_scaled)
        ):
            class_expectation = torch.matmul(class_probs, self.class_values_scaled)
            residual = torch.tanh(amp_base) * self.residual_limit_scaled
            amp_pred = class_expectation + residual
        else:
            amp_pred = amp_base

        return {
            "z_views": z_views,
            "z_pooled": z_pooled,
            "z_views_cons": z_views_cons,
            "z_pooled_cons": z_pooled_cons,
            "z_fused": z_fused,
            "z_presence_in": z_presence_in,
            "z_severity_in": z_severity_in,
            "z_presence": z_presence,
            "z_severity": z_severity,
            "h_feat": h_feat,
            "attention": attn,
            "pooled_mask": pooled_mask,
            "presence_logit": presence_logit,
            "amp_pred": amp_pred,
            "amp_logvar": amp_logvar,
            "amp_tier_logit": amp_tier_logit,
            "amp_tier_score": amp_tier_score,
            "class_probs": class_probs,
            "class_expectation": class_expectation,
        }


def masked_consistency_loss(z_views: torch.Tensor, z_pooled: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
    diff = (z_views - z_pooled.unsqueeze(1)) ** 2
    diff = diff.mean(dim=-1)
    diff = diff * view_mask
    denom = torch.clamp(view_mask.sum(), min=1.0)
    return diff.sum() / denom


def pairwise_rank_loss(pred: torch.Tensor, truth: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    diff_truth = truth.unsqueeze(1) - truth.unsqueeze(0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    mask = diff_truth > 0
    if not mask.any():
        return pred.new_tensor(0.0)
    signed = diff_pred[mask] - float(margin)
    return F.softplus(-signed).mean()


def tower_orthogonality_loss(z_presence: torch.Tensor, z_severity: torch.Tensor) -> torch.Tensor:
    zp = F.normalize(z_presence, dim=-1)
    zs = F.normalize(z_severity, dim=-1)
    cos = (zp * zs).sum(dim=-1)
    return (cos ** 2).mean()


def phase_multipliers(args: argparse.Namespace, epoch: int) -> Dict[str, float]:
    mult = {"presence": 1.0, "amp": 1.0, "tier": 1.0, "rank": 1.0, "consistency": 1.0, "orth": 1.0}
    if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
        mult["amp"] = 0.0
        mult["tier"] = 0.0
        mult["rank"] = 0.0
        mult["orth"] = 0.0
    return mult


def compute_loss(
    model: HybridMultiviewFnirsNet,
    outputs: Dict[str, torch.Tensor],
    presence_true: torch.Tensor,
    amp_true_scaled: torch.Tensor,
    amp_tier_true: torch.Tensor,
    args: argparse.Namespace,
    pos_weight: Optional[torch.Tensor],
    severity_class_weights: Optional[torch.Tensor] = None,
    epoch: int = 1,
) -> Dict[str, torch.Tensor]:
    presence_loss = F.binary_cross_entropy_with_logits(
        outputs["presence_logit"],
        presence_true,
        pos_weight=pos_weight,
    )

    pos_mask = presence_true > 0.5
    if pos_mask.any():
        if args.amp_regression_loss in {"heteroscedastic", "beta_nll"}:
            mu = outputs["amp_pred"][pos_mask]
            logvar = outputs["amp_logvar"][pos_mask].clamp(args.amp_logvar_min, args.amp_logvar_max)
            sq_err = (mu - amp_true_scaled[pos_mask]) ** 2
            base_nll = 0.5 * (torch.exp(-logvar) * sq_err + logvar)
            if args.amp_regression_loss == "beta_nll":
                amp_loss = (base_nll * torch.exp(args.beta_nll_beta * logvar.detach())).mean()
            else:
                amp_loss = base_nll.mean()
        else:
            amp_loss = F.smooth_l1_loss(outputs["amp_pred"][pos_mask], amp_true_scaled[pos_mask])
    else:
        amp_loss = outputs["amp_pred"].new_tensor(0.0)

    tier_logits = outputs["amp_tier_logit"]
    if pos_mask.any() and tier_logits.shape[-1] >= 1:
        pos_targets = amp_tier_true[pos_mask]
        sample_weights = None
        if severity_class_weights is not None and pos_targets.numel() > 0:
            sample_weights = severity_class_weights[pos_targets.clamp(min=0)]
        if args.severity_head == "cumulative_link":
            y_ord = cumulative_link_targets(pos_targets, tier_logits.shape[-1] + 1)
            loss_mat = F.binary_cross_entropy_with_logits(tier_logits[pos_mask], y_ord, reduction="none")
            loss_vec = loss_mat.mean(dim=1)
            if sample_weights is not None:
                tier_loss = (loss_vec * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1e-6)
            else:
                tier_loss = loss_vec.mean()
        else:
            logits = tier_logits[pos_mask]
            if logits.shape[-1] > 1:
                if args.severity_head == "soft_label_ce":
                    y_soft = soft_label_targets(pos_targets, logits.shape[-1], args.soft_label_sigma)
                    loss_vec = -(y_soft * F.log_softmax(logits, dim=1)).sum(dim=1)
                else:
                    loss_vec = F.cross_entropy(logits, pos_targets, reduction="none")
                if sample_weights is not None:
                    tier_loss = (loss_vec * sample_weights).sum() / torch.clamp(sample_weights.sum(), min=1e-6)
                else:
                    tier_loss = loss_vec.mean()
            else:
                tier_loss = outputs["amp_pred"].new_tensor(0.0)
    else:
        tier_loss = outputs["amp_pred"].new_tensor(0.0)

    cons_loss = masked_consistency_loss(outputs["z_views_cons"], outputs["z_pooled_cons"], outputs["pooled_mask"])

    if pos_mask.any() and args.rank_weight > 0:
        rank_loss = pairwise_rank_loss(outputs["amp_pred"][pos_mask], amp_true_scaled[pos_mask], margin=args.rank_margin_scaled)
    else:
        rank_loss = outputs["amp_pred"].new_tensor(0.0)

    orth_loss = tower_orthogonality_loss(outputs["z_presence"], outputs["z_severity"]) if args.tower_orth_weight > 0 else outputs["amp_pred"].new_tensor(0.0)
    phase = phase_multipliers(args, epoch)

    weighted_losses = {
        "presence": phase["presence"] * args.presence_weight * presence_loss,
        "amp": phase["amp"] * args.amplitude_weight * (amp_loss + phase["rank"] * args.rank_weight * rank_loss),
        "tier": phase["tier"] * args.tier_weight * tier_loss,
        "consistency": phase["consistency"] * args.consistency_weight * cons_loss,
    }
    active_tasks = {k: float(weighted_losses[k].detach().abs().item()) > 0 for k in weighted_losses}
    orth_total = phase["orth"] * args.tower_orth_weight * orth_loss

    if model.loss_weighter is not None:
        total_core, learned_weights = model.loss_weighter(weighted_losses, active_tasks)
        total = total_core + orth_total
    else:
        total = sum(weighted_losses.values()) + orth_total
        learned_weights = {k: outputs["amp_pred"].new_tensor(1.0 if active_tasks[k] else 0.0) for k in weighted_losses}

    return {
        "total": total,
        "presence": presence_loss.detach(),
        "amp": amp_loss.detach(),
        "tier": tier_loss.detach(),
        "consistency": cons_loss.detach(),
        "rank": rank_loss.detach(),
        "orth": orth_loss.detach(),
        "weight_presence": learned_weights["presence"].detach(),
        "weight_amp": learned_weights["amp"].detach(),
        "weight_tier": learned_weights["tier"].detach(),
        "weight_consistency": learned_weights["consistency"].detach(),
    }



# ---------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------

class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = None
        self.best_state = None
        self.counter = 0

    def step(self, value: float, model: nn.Module) -> bool:
        if self.best is None or value < self.best:
            self.best = value
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def run_epoch(model, loader, optimizer, device, args, pos_weight, severity_class_weights=None, epoch: int = 1):
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"total": 0.0, "presence": 0.0, "amp": 0.0, "tier": 0.0, "consistency": 0.0, "rank": 0.0, "orth": 0.0,
              "weight_presence": 0.0, "weight_amp": 0.0, "weight_tier": 0.0, "weight_consistency": 0.0}
    n = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        baseline_features = batch["baseline_features"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        presence = batch["presence"].to(device, non_blocking=True)
        amp_scaled = batch["amp_scaled"].to(device, non_blocking=True)
        amp_tier = batch["amp_tier"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            out = model(
                x,
                baseline_features,
                view_mask,
                training=train_mode,
                view_dropout=args.view_dropout if train_mode else 0.0,
            )
            loss_dict = compute_loss(model, out, presence, amp_scaled, amp_tier, args, pos_weight, severity_class_weights, epoch=epoch)
            loss = loss_dict["total"]
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        bs = x.shape[0]
        n += bs
        for k in totals:
            totals[k] += float(loss_dict[k].item()) * bs

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def collect_outputs(model, loader, device, amp_scale: float, severity_head: str, severity_target: str, severity_target_values: Optional[np.ndarray] = None, full_eval_amp_values: Optional[np.ndarray] = None):
    model.eval()
    rows = []
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        baseline_features = batch["baseline_features"].to(device, non_blocking=True)
        view_mask = batch["view_mask"].to(device, non_blocking=True)
        presence_true = batch["presence"].cpu().numpy()
        amp_scaled_true = batch["amp_scaled"].cpu().numpy()
        amp_tier_true = batch["amp_tier"].cpu().numpy()
        class_values = np.asarray(severity_target_values, dtype=np.float32) if severity_target_values is not None else np.asarray([], dtype=np.float32)
        full_eval_values = np.asarray(full_eval_amp_values, dtype=np.float32) if full_eval_amp_values is not None else np.asarray([], dtype=np.float32)

        out = model(x, baseline_features, view_mask, training=False, view_dropout=0.0)
        presence_prob = torch.sigmoid(out["presence_logit"]).cpu().numpy()
        amp_pred_scaled = out["amp_pred"].cpu().numpy()
        amp_logvar = out["amp_logvar"].cpu().numpy()
        amp_sigma_scaled = np.sqrt(np.exp(np.clip(amp_logvar, -20, 20)))
        amp_tier_logit = out["amp_tier_logit"].cpu().numpy()
        if severity_head == "cumulative_link":
            amp_tier_pred = (torch.sigmoid(torch.from_numpy(amp_tier_logit)) > 0.5).sum(dim=1).numpy().astype(np.int64)
        else:
            amp_tier_pred = amp_tier_logit.argmax(axis=1) if amp_tier_logit.shape[1] > 1 else np.zeros(len(presence_true), dtype=np.int64)

        if severity_target == "fixed_amp_classes" and len(class_values) > 0:
            amp_class_true = amp_tier_true.astype(np.int64)
            amp_class_pred = np.clip(amp_tier_pred.astype(np.int64), 0, len(class_values) - 1)
            amp_class_true_value = np.full(len(amp_class_true), np.nan, dtype=np.float32)
            valid_true = amp_class_true >= 0
            amp_class_true_value[valid_true] = class_values[amp_class_true[valid_true]]
            amp_class_pred_value = class_values[amp_class_pred]
        else:
            amp_class_true = amp_tier_true.astype(np.int64)
            amp_class_pred = amp_tier_pred.astype(np.int64)
            amp_class_true_value = np.full(len(amp_class_true), np.nan, dtype=np.float32)
            amp_class_pred_value = np.full(len(amp_class_pred), np.nan, dtype=np.float32)

        amp_true_raw = amp_scaled_true * amp_scale
        amp_pred_raw = amp_pred_scaled * amp_scale
        if len(full_eval_values) > 0:
            amp_fullclass_true = np.full(len(amp_true_raw), -1, dtype=np.int64)
            amp_fullclass_pred = np.full(len(amp_pred_raw), -1, dtype=np.int64)
            valid_pos = presence_true > 0.5
            if np.any(valid_pos):
                amp_fullclass_true[valid_pos] = np.argmin(np.abs(amp_true_raw[valid_pos, None] - full_eval_values[None, :]), axis=1).astype(np.int64)
                amp_fullclass_pred[valid_pos] = np.argmin(np.abs(amp_pred_raw[valid_pos, None] - full_eval_values[None, :]), axis=1).astype(np.int64)
        else:
            amp_fullclass_true = np.full(len(amp_true_raw), -1, dtype=np.int64)
            amp_fullclass_pred = np.full(len(amp_pred_raw), -1, dtype=np.int64)

        z_views = out["z_views"].cpu().numpy()
        z_pooled = out["z_pooled"].cpu().numpy()
        z_views_cons = out["z_views_cons"].cpu().numpy()
        z_pooled_cons = out["z_pooled_cons"].cpu().numpy()
        z_fused = out["z_fused"].cpu().numpy()
        attn = out["attention"].cpu().numpy()
        vmask = view_mask.cpu().numpy()

        for i in range(len(presence_true)):
            valid = vmask[i] > 0.5
            zi = z_views_cons[i][valid]
            disp = float(np.mean((zi - z_pooled_cons[i][None, :]) ** 2)) if zi.size else np.nan
            if zi.shape[0] >= 2:
                zz = zi / np.clip(np.linalg.norm(zi, axis=1, keepdims=True), 1e-8, None)
                cos = []
                for a in range(len(zz)):
                    for b in range(a + 1, len(zz)):
                        cos.append(float(np.dot(zz[a], zz[b])))
                cos_mean = float(np.mean(cos)) if cos else np.nan
            else:
                cos_mean = np.nan

            rows.append({
                "presence_true": float(presence_true[i]),
                "amp_true_raw": float(amp_true_raw[i]),
                "amp_tier_true": int(amp_tier_true[i]),
                "amp_class_true": int(amp_class_true[i]),
                "amp_class_true_value": float(amp_class_true_value[i]) if np.isfinite(amp_class_true_value[i]) else np.nan,
                "presence_prob": float(presence_prob[i]),
                "amp_pred_raw": float(amp_pred_raw[i]),
                "amp_pred_sigma_raw": float(amp_sigma_scaled[i] * amp_scale),
                "amp_tier_pred": int(amp_tier_pred[i]),
                "amp_class_pred": int(amp_class_pred[i]),
                "amp_class_pred_value": float(amp_class_pred_value[i]) if np.isfinite(amp_class_pred_value[i]) else np.nan,
                "amp_fullclass_true": int(amp_fullclass_true[i]),
                "amp_fullclass_pred": int(amp_fullclass_pred[i]),
                "embedding_dispersion": disp,
                "cosine_similarity_mean": cos_mean,
                "z_pooled": z_pooled[i],
                "z_fused": z_fused[i],
                "z_views": z_views[i],
                "z_views_cons": z_views_cons[i],
                "view_mask": vmask[i],
                "attention": attn[i],
            })
    return pd.DataFrame(rows)

def choose_threshold_from_val(val_df: pd.DataFrame) -> float:
    y = val_df["presence_true"].values.astype(int)
    s = val_df["presence_prob"].values
    if len(np.unique(y)) < 2:
        return 0.5
    thresholds = np.unique(np.clip(s, 0, 1))
    best_thr, best_j = 0.5, -np.inf
    for thr in thresholds:
        pred = (s >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_thr = float(thr)
    return best_thr

def classification_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    y = df["presence_true"].values.astype(int)
    s = df["presence_prob"].values
    out = {}
    if len(np.unique(y)) > 1:
        out["presence_auc"] = float(roc_auc_score(y, s))
        out["presence_ap"] = float(average_precision_score(y, s))
    else:
        out["presence_auc"] = np.nan
        out["presence_ap"] = np.nan
    out["presence_brier"] = float(brier_score_loss(y, s)) if len(y) else np.nan

    pred = (s >= threshold).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    out["sensitivity"] = tp / max(tp + fn, 1)
    out["specificity"] = tn / max(tn + fp, 1)
    out["accuracy"] = (tp + tn) / max(len(y), 1)
    return out

def regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    pos = df["presence_true"] > 0.5
    if pos.sum() == 0:
        return {"amp_mae": np.nan, "amp_rmse": np.nan, "amp_r2": np.nan, "amp_spearman_r": np.nan, "amp_spearman_p": np.nan}
    yt = df.loc[pos, "amp_true_raw"].values
    yp = df.loc[pos, "amp_pred_raw"].values
    try:
        sr = spearmanr(yt, yp)
        spearman_r = float(sr.statistic)
        spearman_p = float(sr.pvalue)
    except Exception:
        spearman_r = np.nan
        spearman_p = np.nan
    return {
        "amp_mae": float(mean_absolute_error(yt, yp)),
        "amp_rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "amp_r2": float(r2_score(yt, yp)),
        "amp_spearman_r": spearman_r,
        "amp_spearman_p": spearman_p,
    }

def amp_tolerance_metrics(df: pd.DataFrame, tolerances: Tuple[float, ...] = (5.0, 10.0, 15.0)) -> Dict[str, float]:
    pos = df["presence_true"] > 0.5
    out = {}
    if pos.sum() == 0:
        for tol in tolerances:
            out[f"amp_within_{int(tol)}"] = np.nan
        return out
    yt = df.loc[pos, "amp_true_raw"].values
    yp = df.loc[pos, "amp_pred_raw"].values
    err = np.abs(yp - yt)
    for tol in tolerances:
        out[f"amp_within_{int(tol)}"] = float(np.mean(err <= tol))
    return out


def ordinal_metrics_from_columns(df: pd.DataFrame, true_col: str, pred_col: str, prefix: str) -> Dict[str, float]:
    pos = df[true_col] >= 0
    if pos.sum() == 0:
        return {
            f"{prefix}_accuracy": np.nan,
            f"{prefix}_within1_accuracy": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_qwk": np.nan,
        }
    yt = df.loc[pos, true_col].values.astype(int)
    yp = df.loc[pos, pred_col].values.astype(int)
    within1 = np.mean(np.abs(yt - yp) <= 1)
    class_mae = np.mean(np.abs(yt - yp))
    try:
        qwk = float(cohen_kappa_score(yt, yp, weights="quadratic"))
    except Exception:
        qwk = np.nan
    return {
        f"{prefix}_accuracy": float(accuracy_score(yt, yp)),
        f"{prefix}_within1_accuracy": float(within1),
        f"{prefix}_mae": float(class_mae),
        f"{prefix}_qwk": qwk,
    }


def amplitude_class_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out = ordinal_metrics_from_columns(df, "amp_class_true", "amp_class_pred", "amp_class")
    out.update({
        "amp_tier_accuracy": out["amp_class_accuracy"],
        "amp_tier_within1_accuracy": out["amp_class_within1_accuracy"],
        "amp_tier_mae": out["amp_class_mae"],
        "amp_tier_qwk": out["amp_class_qwk"],
    })
    return out


def full_amplitude_class_metrics(df: pd.DataFrame) -> Dict[str, float]:
    return ordinal_metrics_from_columns(df, "amp_fullclass_true", "amp_fullclass_pred", "amp_fullclass")


def low_blunted_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    pos = df["presence_true"] > 0.5
    if pos.sum() == 0:
        return {
            "amp_low_blunted_sensitivity": np.nan,
            "amp_low_blunted_specificity": np.nan,
            "amp_low_blunted_accuracy": np.nan,
        }
    yt = (df.loc[pos, "amp_true_raw"].values < threshold).astype(int)
    yp = (df.loc[pos, "amp_pred_raw"].values < threshold).astype(int)
    tp = ((yp == 1) & (yt == 1)).sum()
    tn = ((yp == 0) & (yt == 0)).sum()
    fp = ((yp == 1) & (yt == 0)).sum()
    fn = ((yp == 0) & (yt == 1)).sum()
    return {
        "amp_low_blunted_sensitivity": tp / max(tp + fn, 1),
        "amp_low_blunted_specificity": tn / max(tn + fp, 1),
        "amp_low_blunted_accuracy": (tp + tn) / max(len(yt), 1),
    }

def pipeline_predictability(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    Xtr, ytr = [], []
    Xte, yte = [], []
    for df, X, y in [(train_df, Xtr, ytr), (test_df, Xte, yte)]:
        for _, row in df.iterrows():
            z_views = row["z_views_cons"]
            mask = row["view_mask"] > 0.5
            valid_idx = np.where(mask)[0]
            for pidx in valid_idx:
                X.append(z_views[pidx])
                y.append(int(pidx))
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return np.nan
    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(np.asarray(Xtr), np.asarray(ytr))
    pred = clf.predict(np.asarray(Xte))
    return float(accuracy_score(np.asarray(yte), pred))


def train_one_fold(
    data: Dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> Dict:
    ensure_dir(fold_dir)

    X = data["X"]
    view_mask = data["view_mask"]
    y_presence = data["y_presence"]
    y_amp_scaled = data["y_amp_scaled"]
    y_amp_raw = data["y_amp_raw"]
    baseline_features = data["baseline_features"]

    view_norm = fit_pipeline_normalizer(X, view_mask, train_idx)
    Xn = apply_pipeline_normalizer(X, view_norm)

    feature_scaler = fit_feature_scaler(baseline_features, train_idx)
    features_scaled = feature_scaler.transform(baseline_features).astype(np.float32)

    y_amp_tier, severity_target_values = build_amp_targets(
        y_amp_raw,
        train_idx,
        severity_target=args.severity_target,
        target_num_tiers=args.num_amp_tiers,
        fixed_amp_class_values=args.fixed_amp_class_values,
    )
    if args.severity_target == "fixed_amp_classes":
        num_amp_tiers = int(len(severity_target_values)) if len(severity_target_values) else 1
    else:
        num_amp_tiers = int(len(np.unique(y_amp_tier[y_amp_tier >= 0]))) if (y_amp_tier >= 0).any() else 1
    num_amp_tiers = max(num_amp_tiers, 1)

    if args.severity_class_balance == "inverse_freq" and num_amp_tiers > 1:
        severity_class_weights_np = inverse_frequency_class_weights(y_amp_tier[train_idx], num_amp_tiers)
    else:
        severity_class_weights_np = np.ones(num_amp_tiers, dtype=np.float32)
    if args.severity_target == "fixed_amp_classes":
        print(f"[{fold_dir.name}] fixed amplitude classes: {severity_target_values.tolist()}", flush=True)
    else:
        print(f"[{fold_dir.name}] quantile tier edges: {severity_target_values.tolist()}", flush=True)

    train_ds = FnirsSampleDataset(Xn, view_mask, features_scaled, y_presence, y_amp_scaled, y_amp_raw, y_amp_tier, train_idx)
    val_ds = FnirsSampleDataset(Xn, view_mask, features_scaled, y_presence, y_amp_scaled, y_amp_raw, y_amp_tier, val_idx)
    test_ds = FnirsSampleDataset(Xn, view_mask, features_scaled, y_presence, y_amp_scaled, y_amp_raw, y_amp_tier, test_idx)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    if args.full_eval_amp_values is not None and len(args.full_eval_amp_values):
        full_eval_amp_values = np.asarray(sorted({float(x) for x in args.full_eval_amp_values if float(x) > 0}), dtype=np.float32)
    else:
        full_eval_amp_values = np.asarray(sorted({float(x) for x in y_amp_raw if float(x) > 0}), dtype=np.float32)

    pos = y_presence[train_idx]
    neg = max(float((pos < 0.5).sum()), 1.0)
    posn = max(float((pos > 0.5).sum()), 1.0)
    pos_weight = torch.tensor([neg / posn], dtype=torch.float32, device=device)

    class_values_scaled = None
    if args.severity_target == "fixed_amp_classes" and len(severity_target_values):
        class_values_scaled = np.asarray(severity_target_values, dtype=np.float32) / float(data["amp_scale"])

    model = HybridMultiviewFnirsNet(
        in_channels=X.shape[2],
        baseline_feat_dim=features_scaled.shape[1],
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        feature_hidden_dim=args.feature_hidden_dim,
        task_hidden_dim=args.task_hidden_dim,
        consistency_proj_dim=args.consistency_proj_dim,
        dropout=args.dropout,
        num_amp_tiers=num_amp_tiers,
        severity_head=args.severity_head,
        amp_regression_loss=args.amp_regression_loss,
        task_weighting=args.task_weighting,
        amp_prediction_mode=args.amp_prediction_mode,
        class_values_scaled=class_values_scaled,
        residual_limit_scaled=args.residual_limit_raw / float(data["amp_scale"]),
        severity_shared_grad_scale=args.severity_shared_grad_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    severity_class_weights = torch.tensor(severity_class_weights_np, dtype=torch.float32, device=device)
    stopper = EarlyStopper(args.patience)
    history = []

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, args, pos_weight, severity_class_weights, epoch=epoch)
        va = run_epoch(model, val_loader, None, device, args, pos_weight, severity_class_weights, epoch=epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[fold {fold_dir.name}] epoch={epoch:02d} "
                f"train={tr['total']:.4f} val={va['total']:.4f} "
                f"(pres={va['presence']:.4f}, amp={va['amp']:.4f}, tier={va['tier']:.4f}, rank={va['rank']:.4f}, orth={va['orth']:.4f}, cons={va['consistency']:.4f}; "
                f"w_pres={va['weight_presence']:.3f}, w_amp={va['weight_amp']:.3f}, "
                f"w_tier={va['weight_tier']:.3f}, w_cons={va['weight_consistency']:.3f})",
                flush=True,
            )
        if stopper.step(va["total"], model):
            print(f"[fold {fold_dir.name}] early stop at epoch {epoch}", flush=True)
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)

    train_df = collect_outputs(model, train_loader, device, data["amp_scale"], args.severity_head, args.severity_target, severity_target_values, full_eval_amp_values)
    val_df = collect_outputs(model, val_loader, device, data["amp_scale"], args.severity_head, args.severity_target, severity_target_values, full_eval_amp_values)
    test_df = collect_outputs(model, test_loader, device, data["amp_scale"], args.severity_head, args.severity_target, severity_target_values, full_eval_amp_values)

    thr = choose_threshold_from_val(val_df)
    metrics = {}
    metrics.update({f"val_{k}": v for k, v in classification_metrics(val_df, thr).items()})
    metrics.update({f"test_{k}": v for k, v in classification_metrics(test_df, thr).items()})
    metrics.update({f"val_{k}": v for k, v in regression_metrics(val_df).items()})
    metrics.update({f"test_{k}": v for k, v in regression_metrics(test_df).items()})
    metrics.update({f"val_{k}": v for k, v in amp_tolerance_metrics(val_df).items()})
    metrics.update({f"test_{k}": v for k, v in amp_tolerance_metrics(test_df).items()})
    metrics.update({f"val_{k}": v for k, v in amplitude_class_metrics(val_df).items()})
    metrics.update({f"test_{k}": v for k, v in amplitude_class_metrics(test_df).items()})
    metrics.update({f"val_{k}": v for k, v in full_amplitude_class_metrics(val_df).items()})
    metrics.update({f"test_{k}": v for k, v in full_amplitude_class_metrics(test_df).items()})
    metrics.update({f"val_{k}": v for k, v in low_blunted_metrics(val_df, args.low_blunted_threshold).items()})
    metrics.update({f"test_{k}": v for k, v in low_blunted_metrics(test_df, args.low_blunted_threshold).items()})

    metrics["val_embedding_dispersion"] = float(val_df["embedding_dispersion"].mean())
    metrics["test_embedding_dispersion"] = float(test_df["embedding_dispersion"].mean())
    metrics["val_cosine_similarity_mean"] = float(val_df["cosine_similarity_mean"].mean())
    metrics["test_cosine_similarity_mean"] = float(test_df["cosine_similarity_mean"].mean())
    metrics["test_pipeline_predictability"] = pipeline_predictability(train_df, test_df)
    metrics["threshold"] = thr
    metrics["n_train"] = int(len(train_idx))
    metrics["n_val"] = int(len(val_idx))
    metrics["n_test"] = int(len(test_idx))
    metrics["num_amp_tiers"] = int(num_amp_tiers)
    metrics["num_amp_classes"] = int(num_amp_tiers) if args.severity_target == "fixed_amp_classes" else np.nan
    metrics["amp_regression_loss"] = args.amp_regression_loss
    metrics["task_weighting"] = args.task_weighting
    metrics["severity_head"] = args.severity_head
    metrics["amp_prediction_mode"] = args.amp_prediction_mode
    metrics["rank_weight"] = args.rank_weight
    metrics["warmup_epochs"] = args.warmup_epochs
    metrics["severity_shared_grad_scale"] = args.severity_shared_grad_scale
    metrics["tower_orth_weight"] = args.tower_orth_weight
    metrics["severity_target_fixed_classes"] = 1 if args.severity_target == "fixed_amp_classes" else 0

    # Classical baselines retained for comparison.
    Xtr = features_scaled[train_idx]
    Xva = features_scaled[val_idx]
    Xte = features_scaled[test_idx]

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_presence[train_idx].astype(int))
    val_prob = clf.predict_proba(Xva)[:, 1]
    test_prob = clf.predict_proba(Xte)[:, 1]
    val_base_df = pd.DataFrame({"presence_true": y_presence[val_idx], "presence_prob": val_prob})
    test_base_df = pd.DataFrame({"presence_true": y_presence[test_idx], "presence_prob": test_prob})
    thr_base = choose_threshold_from_val(val_base_df)
    metrics.update({f"baseline_val_{k}": v for k, v in classification_metrics(val_base_df, thr_base).items()})
    metrics.update({f"baseline_test_{k}": v for k, v in classification_metrics(test_base_df, thr_base).items()})

    pos_train_mask = y_presence[train_idx] > 0.5
    if pos_train_mask.sum() >= 2:
        ridge = Ridge(alpha=1.0)
        ridge.fit(Xtr[pos_train_mask], y_amp_raw[train_idx][pos_train_mask])
        val_pred = ridge.predict(Xva[y_presence[val_idx] > 0.5]) if (y_presence[val_idx] > 0.5).sum() else np.array([])
        test_pred = ridge.predict(Xte[y_presence[test_idx] > 0.5]) if (y_presence[test_idx] > 0.5).sum() else np.array([])
        if len(val_pred):
            yt = y_amp_raw[val_idx][y_presence[val_idx] > 0.5]
            metrics["baseline_val_amp_mae"] = float(mean_absolute_error(yt, val_pred))
        else:
            metrics["baseline_val_amp_mae"] = np.nan
        if len(test_pred):
            yt = y_amp_raw[test_idx][y_presence[test_idx] > 0.5]
            metrics["baseline_test_amp_mae"] = float(mean_absolute_error(yt, test_pred))
        else:
            metrics["baseline_test_amp_mae"] = np.nan
    else:
        metrics["baseline_val_amp_mae"] = np.nan
        metrics["baseline_test_amp_mae"] = np.nan

    np.savez_compressed(
        fold_dir / "predictions.npz",
        train_presence_true=train_df["presence_true"].values,
        train_presence_prob=train_df["presence_prob"].values,
        val_presence_true=val_df["presence_true"].values,
        val_presence_prob=val_df["presence_prob"].values,
        test_presence_true=test_df["presence_true"].values,
        test_presence_prob=test_df["presence_prob"].values,
        val_amp_true_raw=val_df["amp_true_raw"].values,
        val_amp_pred_raw=val_df["amp_pred_raw"].values,
        val_amp_class_true=val_df["amp_class_true"].values,
        val_amp_class_pred=val_df["amp_class_pred"].values,
        val_amp_fullclass_true=val_df["amp_fullclass_true"].values,
        val_amp_fullclass_pred=val_df["amp_fullclass_pred"].values,
        test_amp_true_raw=test_df["amp_true_raw"].values,
        test_amp_pred_raw=test_df["amp_pred_raw"].values,
        test_amp_class_true=test_df["amp_class_true"].values,
        test_amp_class_pred=test_df["amp_class_pred"].values,
        test_amp_fullclass_true=test_df["amp_fullclass_true"].values,
        test_amp_fullclass_pred=test_df["amp_fullclass_pred"].values,
    )
    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    torch.save(model.state_dict(), fold_dir / "model.pt")
    extra_save = {
        "view_mean": view_norm["mean"],
        "view_std": view_norm["std"],
        "feature_mean": feature_scaler.mean_,
        "feature_scale": feature_scaler.scale_,
        "severity_target_values": severity_target_values,
        "full_eval_amp_values": full_eval_amp_values,
    }
    if args.severity_head == "cumulative_link" and hasattr(model.ordinal_head, "thresholds"):
        th = model.ordinal_head.thresholds()
        if th is not None:
            extra_save["ordinal_thresholds"] = th.detach().cpu().numpy()
    np.savez_compressed(fold_dir / "normalizers_and_targets.npz", **extra_save)

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    ensure_dir(outdir)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda was requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    try:
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
        torch.set_num_threads(max(1, omp_threads))
    except Exception:
        pass
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print("Device:", device, flush=True)

    data = load_or_build_cache(args)
    print(
        f"Loaded dataset: n_samples={len(data['X'])}, n_pipelines={len(data['pipeline_names'])}, "
        f"grid=[{data['grid'][0]}, {data['grid'][-1]}], T={len(data['grid'])}",
        flush=True,
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "device_used": str(device), "pipeline_names": list(data["pipeline_names"])},
            f,
            indent=2,
        )

    groups = np.asarray(data["subjects"])
    unique_subjects = np.unique(groups)
    n_folds = min(args.folds, len(unique_subjects))
    gkf = GroupKFold(n_splits=n_folds)

    fold_metrics = []
    fold_iter = list(gkf.split(data["X"], data["y_presence"], groups))
    if args.single_fold_only >= 0:
        fold_iter = [fold_iter[args.single_fold_only]]

    for fold_num, (trainval_idx, test_idx) in enumerate(fold_iter):
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac_within_train, random_state=args.seed + fold_num)
        inner_train_idx, val_idx_rel = next(
            gss.split(np.zeros(len(trainval_idx)), data["y_presence"][trainval_idx], groups[trainval_idx])
        )
        train_idx = trainval_idx[inner_train_idx]
        val_idx = trainval_idx[val_idx_rel]

        fold_dir = outdir / f"fold_{fold_num:02d}"
        metrics = train_one_fold(data, train_idx, val_idx, test_idx, args, device, fold_dir)
        metrics["fold"] = fold_num
        fold_metrics.append(metrics)

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(outdir / "cv_metrics.csv", index=False)

    summary = metrics_df.mean(numeric_only=True).to_dict()
    with open(outdir / "cv_metrics_mean.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CV summary (mean) ===")
    for k in sorted(summary):
        print(f"{k}: {summary[k]:.4f}" if isinstance(summary[k], (float, int)) and not isinstance(summary[k], bool) else f"{k}: {summary[k]}")
    print("\nFinished.")


if __name__ == "__main__":
    main()
