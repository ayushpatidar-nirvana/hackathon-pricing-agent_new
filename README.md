## Non-fleet hypothesis validation suite

This repo provides a small **univariate pricing / hypothesis validation engine** to go from:

- a hypothesis (e.g. “higher mileage → higher loss ratio”)
- to bucketing + loss ratio / relativity indications
- to lift-style ordering + premium displacement outputs

### Quickstart

Install deps:

```bash
python3 -m pip install -r requirements.txt
```

Run in Python (or a notebook / Cursor scratchpad):

```python
from engine.univariate_engine import (
    load_csv,
    clean_model_data,
    run_univariate,
    BinningConfig,
    CredibilityConfig,
)

# Load + light cleaning
raw = load_csv("data/model_data_nf_v1.csv")
df = clean_model_data(raw)

# Core fields (defaults):
#   loss     = M3_PROJECTED_AL_LOSS_ADJ
#   premium  = AL_EARNED_PREMIUM_ADJ_ONLEVEL
#   exposure = EARNED_PU_YEARS

# Example: Mileage hypothesis
summary, lift, displacement = run_univariate(
    df,
    feature="AVG_ANNUAL_MILEAGE",
    binning=BinningConfig(method="quantile", n_bins=10, include_missing=True),
    credibility=CredibilityConfig(full_cred_exposure=1000.0),
    normalize_relativity_to="premium",  # keeps total premium stable
)

summary.head()
lift.head()
displacement.head()
```

### What you get back

- **`summary`**: per-bin totals and indications
  - `loss`, `premium`, `exposure`, `loss_ratio`
  - `raw_relativity`: bin LR / overall LR
  - `credibility`: \(z = \min(1, \sqrt{exposure / K})\)
  - `cred_relativity`: credibility-weighted relativity
  - `adj_relativity`: normalized relativity (default normalized to premium-weighted mean 1)
  - `indicated_premium`, `displacement`, `abs_displacement`

- **`lift`**: bins ordered by `adj_relativity` with cumulative % columns
  - `cum_exposure_pct`, `cum_loss_pct`, `cum_premium_pct`

- **`displacement`**: compact view of displacement by bin

### Text-driven hypothesis helper

A tiny helper maps common phrases to known columns:

```python
from engine.univariate_engine import load_csv, clean_model_data
from agent.text_agent import run_from_text

df = clean_model_data(load_csv("data/model_data_nf_v1.csv"))

run_from_text(df, "Higher mileage = higher loss ratio")
run_from_text(df, "Better telematics (gps score) should reduce losses")
```

### Dataset notes (as shipped)

- `AVG_ANNUAL_MILEAGE` exists (about ~4% missing)
- `TELEMATICS_SCORE` is **not** present; the closest telematics-like field is `RETRO_10SEC_GPS_SCORE_V1_SCORE`
- Several `*_AT_FAULT_COUNT` columns are very sparse / mostly zero with high missingness; `clean_model_data()` can fill them to 0
