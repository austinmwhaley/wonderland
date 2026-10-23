# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=1009.2s hash=7e32ebb84fc3

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.4479 | +0.0677 |
| orders90 | 0.4547 | 0.5202 | +0.0655 |
| churn90 | 0.9305 | 0.9338 | +0.0033 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4167 | — |
| churn-nn | 0.9722 | — |
| identity channel | 0.5043 | 0.2513 |
| identity category | 0.1721 | 0.2653 |
| shuffle value drop | -0.2785 | 0 |
