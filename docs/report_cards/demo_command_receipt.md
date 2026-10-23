# B/C demo command output

seeds=[0] epochs=12 runtime=355.4s hash=ad32ed1f4488

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3589 | +0.0014 |
| orders90 | 0.4503 | 0.4258 | -0.0245 |
| churn90 | 0.9256 | 0.9349 | +0.0094 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4021 | — |
| churn-nn | 0.9733 | — |
| identity channel | 0.4457 | 0.2513 |
| identity category | 0.1623 | 0.2653 |
| shuffle value drop | -0.2221 | 0 |
