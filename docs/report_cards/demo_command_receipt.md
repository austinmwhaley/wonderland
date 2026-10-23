# B/C demo command output

seeds=[0, 0, 1, 2] epochs=12 runtime=1239.0s hash=b17c01c8d397

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3745 | 0.3793 | +0.0048 |
| orders90 | 0.4536 | 0.4485 | -0.0051 |
| churn90 | 0.9292 | 0.9325 | +0.0032 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4180 | — |
| churn-nn | 0.9767 | — |
| identity channel | 0.3131 | 0.2513 |
| identity category | 0.1672 | 0.2653 |
| shuffle value drop | -0.2233 | 0 |
