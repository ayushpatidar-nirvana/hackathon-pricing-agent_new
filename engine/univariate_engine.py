import pandas as pd
import numpy as np

def run_univariate(
    df,
    feature,
    loss_col="M3_PROJECTED_AL_LOSS_ADJ",
    premium_col="AL_EARNED_PREMIUM_ADJ_ONLEVEL",
    exposure_col="EARNED_PU_YEARS",
    n_bins=10,
    credibility_K=1000
):
    df = df[[feature, loss_col, premium_col, exposure_col]].dropna()
    df = df[(df[premium_col] > 0) & (df[exposure_col] > 0)]

    df["bin"] = pd.qcut(df[feature], q=n_bins, duplicates="drop")

    grouped = df.groupby("bin").agg(
        loss=(loss_col, "sum"),
        premium=(premium_col, "sum"),
        exposure=(exposure_col, "sum")
    ).reset_index()

    grouped["loss_ratio"] = grouped["loss"] / grouped["premium"]

    overall_lr = grouped["loss"].sum() / grouped["premium"].sum()
    grouped["raw_relativity"] = grouped["loss_ratio"] / overall_lr

    grouped["credibility"] = np.minimum(
        1.0, np.sqrt(grouped["exposure"] / credibility_K)
    )

    grouped["adj_relativity"] = (
        grouped["credibility"] * grouped["raw_relativity"]
        + (1 - grouped["credibility"]) * 1.0
    )

    # Lift data
    lift = grouped.sort_values("adj_relativity", ascending=False).copy()
    lift["cum_exposure_pct"] = lift["exposure"].cumsum() / lift["exposure"].sum()
    lift["cum_loss_pct"] = lift["loss"].cumsum() / lift["loss"].sum()

    # Premium displacement
    grouped["indicated_premium"] = grouped["premium"] * grouped["adj_relativity"]
    grouped["displacement"] = grouped["indicated_premium"] - grouped["premium"]

    return grouped, lift, grouped[["bin", "displacement"]]
