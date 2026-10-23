# B/C demo command output

seeds=[0] epochs=12 runtime=354.7s hash=7448b2b5bb45

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3466 | -0.0109 |
| orders90 | 0.4503 | 0.4251 | -0.0252 |
| churn90 | 0.9256 | 0.9358 | +0.0102 |

| probe | E | table |
|---|---|---|
| retrieval | 0.3972 | — |
| churn-nn | 0.9794 | — |
| identity channel | 0.4303 | 0.2513 |
| identity category | 0.1638 | 0.2653 |
| shuffle value drop | -0.2188 | 0 |
