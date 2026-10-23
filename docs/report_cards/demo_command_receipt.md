# B/C demo command output

seeds=[0] epochs=5 runtime=352.5s hash=8cc146efbb72

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3732 | +0.0157 |
| orders90 | 0.4503 | 0.4444 | -0.0059 |
| churn90 | 0.9256 | 0.9504 | +0.0249 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4325 | — |
| churn-nn | 0.9778 | — |
| identity channel | 0.3605 | 0.2513 |
| identity category | 0.1641 | 0.2653 |
| shuffle value drop | -0.2496 | 0 |
