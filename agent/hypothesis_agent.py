from engine.univariate_engine import run_univariate

def hypothesis_agent(
    df,
    feature,
    expected_direction="increasing",
    min_relativity=1.10,
    min_displacement=100_000,
    displacement_mode="gross",
):
    summary, lift, displacement = run_univariate(df, feature)

    max_rel = summary["adj_relativity"].max()
    # Monotonicity is only meaningful for ordered numeric bins.
    # `run_univariate` sorts bins; we also drop the "MISSING" bin if present.
    monotonic = None
    if "_bin_sort" in summary.columns:
        ordered = summary.sort_values("_bin_sort", ascending=True)
        ordered = ordered[ordered["bin"] != "MISSING"]
        if len(ordered) >= 3:
            monotonic = (
                ordered["loss_ratio"].is_monotonic_increasing
                if expected_direction == "increasing"
                else ordered["loss_ratio"].is_monotonic_decreasing
            )

    net_disp = float(displacement["displacement"].sum())
    gross_disp = float(displacement["abs_displacement"].sum() / 2.0)
    total_disp = gross_disp if displacement_mode == "gross" else abs(net_disp)

    supported = (
        max_rel >= min_relativity
        and (monotonic is None or monotonic)
        and abs(total_disp) >= min_displacement
    )

    return {
        "feature": feature,
        "decision": "ACCEPT" if supported else "REJECT / REVIEW",
        "max_relativity": round(max_rel, 2),
        "monotonic": monotonic,
        "premium_displacement": round(total_disp, 0),
        "premium_displacement_mode": displacement_mode,
        "premium_displacement_net": round(net_disp, 0),
        "premium_displacement_gross": round(gross_disp, 0),
        "summary": summary,
        "lift": lift,
    }
