# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=906.2s hash=db23d97a00e0

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3884 | +0.0082 |
| orders90 | 0.4547 | 0.4515 | -0.0032 |
| churn90 | 0.9305 | 0.9294 | -0.0011 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4180 | — |
| churn-nn | 0.9767 | — |
| identity channel | 0.3131 | 0.2513 |
| identity category | 0.1672 | 0.2653 |
| shuffle value drop | -0.2243 | 0 |
