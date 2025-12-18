from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd


NumericBinningMethod = Literal["quantile", "width", "edges"]


@dataclass(frozen=True)
class BinningConfig:
    """
    Controls how a feature is bucketed prior to univariate analysis.

    Notes:
    - For numeric features, default is quantile bins (similar to deciles).
    - Missing values are optionally assigned to a dedicated "MISSING" bin.
    """

    method: NumericBinningMethod = "quantile"
    n_bins: int = 10
    width: float | None = None
    edges: list[float] | None = None
    include_missing: bool = True
    clip_quantiles: tuple[float, float] | None = (0.001, 0.999)

    # Categorical controls (used when feature is non-numeric or treated as category)
    top_n_categories: int = 25
    other_label: str = "OTHER"
    missing_label: str = "MISSING"


@dataclass(frozen=True)
class CredibilityConfig:
    """
    Limited fluctuation style credibility on exposure:
      z = min(1, sqrt(exposure / full_cred_exposure))
    """

    full_cred_exposure: float = 1000.0


def _coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def load_csv(path: str) -> pd.DataFrame:
    """Thin wrapper to load the modeling CSV."""
    return pd.read_csv(path)


def clean_model_data(
    df: pd.DataFrame,
    *,
    fill_projected_at_fault_missing_with_zero: bool = True,
) -> pd.DataFrame:
    """
    Minimal, safe cleaning for the hackathon engine.

    - Does NOT impute core measures (loss/premium/exposure).
    - Optionally fills missing projected at-fault fields with 0 (they are sparse).
    """
    out = df.copy()

    if fill_projected_at_fault_missing_with_zero:
        fault_cols = [
            c
            for c in out.columns
            if "AT_FAULT" in c.upper() and _is_numeric_series(out[c])
        ]
        for c in fault_cols:
            out[c] = out[c].fillna(0.0)

    return out


def _bin_numeric(
    s: pd.Series,
    cfg: BinningConfig,
) -> tuple[pd.Series, pd.Series | None]:
    """
    Returns:
      - bins: a Series of labels (Intervals for numeric bins, plus optional missing label)
      - bin_sort_key: numeric sort key used to order bins on output
    """
    sn = _coerce_numeric(s)
    missing_mask = sn.isna()

    if cfg.clip_quantiles is not None:
        lo_q, hi_q = cfg.clip_quantiles
        lo = sn.quantile(lo_q)
        hi = sn.quantile(hi_q)
        sn = sn.clip(lower=lo, upper=hi)

    # Build bins for non-missing values only
    non_missing = sn[~missing_mask]
    if non_missing.empty:
        labels = pd.Series([cfg.missing_label] * len(sn), index=sn.index, dtype="object")
        sort_key = pd.Series([np.nan] * len(sn), index=sn.index, dtype="float64")
        return labels, sort_key

    if cfg.method == "quantile":
        # duplicates="drop" prevents failures when many ties exist.
        b = pd.qcut(non_missing, q=cfg.n_bins, duplicates="drop")
    elif cfg.method == "width":
        if cfg.width is None or cfg.width <= 0:
            raise ValueError("BinningConfig.width must be set to a positive number for method='width'.")
        vmin = float(non_missing.min())
        vmax = float(non_missing.max())
        edges = np.arange(vmin, vmax + cfg.width, cfg.width)
        if len(edges) < 3:
            edges = np.array([vmin, vmax])
        b = pd.cut(non_missing, bins=edges, include_lowest=True)
    elif cfg.method == "edges":
        if not cfg.edges or len(cfg.edges) < 2:
            raise ValueError("BinningConfig.edges must be a list of >=2 numbers for method='edges'.")
        b = pd.cut(non_missing, bins=sorted(cfg.edges), include_lowest=True)
    else:
        raise ValueError(f"Unknown numeric binning method: {cfg.method}")

    labels = pd.Series(index=sn.index, dtype="object")
    labels.loc[~missing_mask] = b.astype("object")
    if cfg.include_missing:
        labels.loc[missing_mask] = cfg.missing_label
    else:
        labels = labels.loc[~missing_mask]

    # Sort key: interval left edge; missing -> +inf (last)
    def _left(x: Any) -> float:
        if isinstance(x, pd.Interval):
            return float(x.left)
        if x == cfg.missing_label:
            return float("inf")
        return float("inf")

    sort_key = labels.map(_left).astype("float64")
    return labels, sort_key


def _bin_categorical(
    s: pd.Series,
    cfg: BinningConfig,
) -> tuple[pd.Series, pd.Series]:
    sc = s.astype("object")
    if cfg.include_missing:
        sc = sc.where(~sc.isna(), cfg.missing_label)
    else:
        sc = sc.dropna()

    vc = sc.value_counts(dropna=False)
    top = set(vc.head(cfg.top_n_categories).index.tolist())
    binned = sc.where(sc.isin(top), cfg.other_label)

    # Sort by frequency desc; keep missing last to reduce visual noise
    freq = binned.value_counts(dropna=False)
    order = {k: i for i, k in enumerate(freq.index.tolist())}
    if cfg.missing_label in order:
        order[cfg.missing_label] = len(order) + 999
    sort_key = binned.map(lambda x: order.get(x, len(order) + 1)).astype("int64")
    return binned, sort_key


def _compute_credibility(exposure: pd.Series, cfg: CredibilityConfig) -> pd.Series:
    denom = float(cfg.full_cred_exposure)
    if denom <= 0:
        raise ValueError("CredibilityConfig.full_cred_exposure must be > 0.")
    return np.minimum(1.0, np.sqrt(exposure / denom))


def run_univariate(
    df: pd.DataFrame,
    feature: str,
    *,
    loss_col: str = "M3_PROJECTED_AL_LOSS_ADJ",
    premium_col: str = "AL_EARNED_PREMIUM_ADJ_ONLEVEL",
    exposure_col: str = "EARNED_PU_YEARS",
    binning: BinningConfig | None = None,
    credibility: CredibilityConfig | None = None,
    # Backwards-compatible args (from the original hackathon stub)
    n_bins: int | None = None,
    credibility_K: float | None = None,
    min_premium: float = 1e-9,
    min_exposure: float = 1e-9,
    normalize_relativity_to: Literal["premium", "exposure", "none"] = "premium",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Core univariate engine.

    Returns:
      - summary: per-bin LR, raw + credibility-weighted + normalized relativities, displacement
      - lift: bins ordered by normalized relativity, with cumulative exposure/loss (lift chart input)
      - displacement: compact (bin, displacement, abs_displacement) view
    """
    binning = binning or BinningConfig()
    credibility = credibility or CredibilityConfig()
    if n_bins is not None:
        binning = BinningConfig(**{**binning.__dict__, "n_bins": int(n_bins)})
    if credibility_K is not None:
        credibility = CredibilityConfig(full_cred_exposure=float(credibility_K))

    needed = [feature, loss_col, premium_col, exposure_col]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    base = df[needed].copy()

    # Coerce measures to numeric and filter unusable rows.
    base[loss_col] = _coerce_numeric(base[loss_col]).fillna(0.0)
    base[premium_col] = _coerce_numeric(base[premium_col])
    base[exposure_col] = _coerce_numeric(base[exposure_col])

    base = base[(base[premium_col] > min_premium) & (base[exposure_col] > min_exposure)]

    # Bucket the feature (numeric -> numeric bins; otherwise categorical).
    feat = base[feature]
    if _is_numeric_series(feat):
        bins, sort_key = _bin_numeric(feat, binning)
        base = base.join(pd.DataFrame({"bin": bins, "_bin_sort": sort_key}))
    else:
        bins, sort_key = _bin_categorical(feat, binning)
        base = base.join(pd.DataFrame({"bin": bins, "_bin_sort": sort_key}))

    # If missing bins are excluded, some rows can be dropped by binning.
    base = base.dropna(subset=["bin"])

    grouped = (
        base.groupby("bin", dropna=False)
        .agg(
            loss=(loss_col, "sum"),
            premium=(premium_col, "sum"),
            exposure=(exposure_col, "sum"),
            _bin_sort=("_bin_sort", "min"),
            n=("bin", "size"),
        )
        .reset_index()
    )

    grouped["loss_ratio"] = grouped["loss"] / grouped["premium"]
    overall_lr = float(grouped["loss"].sum() / grouped["premium"].sum()) if grouped["premium"].sum() > 0 else np.nan
    grouped["overall_loss_ratio"] = overall_lr

    grouped["raw_relativity"] = grouped["loss_ratio"] / overall_lr if overall_lr and overall_lr > 0 else np.nan
    grouped["credibility"] = _compute_credibility(grouped["exposure"], credibility)
    grouped["cred_relativity"] = 1.0 + grouped["credibility"] * (grouped["raw_relativity"] - 1.0)

    # Normalize relativities so the chosen weighted mean equals 1.0 (for meaningful premium displacement).
    if normalize_relativity_to == "none":
        grouped["adj_relativity"] = grouped["cred_relativity"]
        norm_factor = 1.0
    else:
        w = grouped["premium"] if normalize_relativity_to == "premium" else grouped["exposure"]
        denom = float((grouped["cred_relativity"] * w).sum() / w.sum()) if w.sum() > 0 else 1.0
        norm_factor = denom if denom and denom > 0 else 1.0
        grouped["adj_relativity"] = grouped["cred_relativity"] / norm_factor

    grouped["norm_factor"] = norm_factor

    # Premium displacement (net sums to ~0 if premium-normalized; gross shows churn/impact).
    grouped["indicated_premium"] = grouped["premium"] * grouped["adj_relativity"]
    grouped["displacement"] = grouped["indicated_premium"] - grouped["premium"]
    grouped["abs_displacement"] = grouped["displacement"].abs()

    # Sort bins for presentation (numeric left-edge order; categorical by frequency key).
    grouped = grouped.sort_values(["_bin_sort", "bin"], ascending=[True, True]).reset_index(drop=True)

    # Lift chart input (rank by indication).
    lift = grouped.sort_values("adj_relativity", ascending=False).copy()
    lift["cum_exposure_pct"] = lift["exposure"].cumsum() / lift["exposure"].sum() if lift["exposure"].sum() > 0 else np.nan
    lift["cum_loss_pct"] = lift["loss"].cumsum() / lift["loss"].sum() if lift["loss"].sum() > 0 else np.nan
    lift["cum_premium_pct"] = lift["premium"].cumsum() / lift["premium"].sum() if lift["premium"].sum() > 0 else np.nan

    displacement = grouped[["bin", "displacement", "abs_displacement"]].copy()

    return grouped, lift, displacement
