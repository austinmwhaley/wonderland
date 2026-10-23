# B/C demo command output

seeds=[0] epochs=12 runtime=338.9s hash=6298228fe54c

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3629 | +0.0054 |
| orders90 | 0.4503 | 0.4461 | -0.0042 |
| churn90 | 0.9256 | 0.9468 | +0.0212 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4180 | — |
| churn-nn | 0.9767 | — |
| identity channel | 0.3131 | 0.2513 |
| identity category | 0.1672 | 0.2653 |
| shuffle value drop | -0.2205 | 0 |
