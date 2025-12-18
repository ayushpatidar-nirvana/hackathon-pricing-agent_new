from engine.univariate_engine import run_univariate

def hypothesis_agent(
    df,
    feature,
    expected_direction="increasing",
    min_relativity=1.10,
    min_displacement=100_000
):
    summary, lift, displacement = run_univariate(df, feature)

    max_rel = summary["adj_relativity"].max()
    monotonic = (
        summary["loss_ratio"].is_monotonic_increasing
        if expected_direction == "increasing"
        else summary["loss_ratio"].is_monotonic_decreasing
    )

    total_disp = displacement["displacement"].sum()

    supported = (
        max_rel >= min_relativity
        and monotonic
        and abs(total_disp) >= min_displacement
    )

    return {
        "feature": feature,
        "decision": "ACCEPT" if supported else "REJECT / REVIEW",
        "max_relativity": round(max_rel, 2),
        "monotonic": monotonic,
        "premium_displacement": round(total_disp, 0),
        "summary": summary
    }
