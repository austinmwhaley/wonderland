# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=662.3s hash=a969f1b0f190

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3869 | +0.0067 |
| orders90 | 0.4547 | 0.4439 | -0.0108 |
| churn90 | 0.9305 | 0.9297 | -0.0008 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4614 | — |
| churn-nn | 0.9811 | — |
| identity channel | 0.4479 | 0.2513 |
| identity category | 0.1618 | 0.2653 |
| shuffle value drop | -0.2557 | 0 |
