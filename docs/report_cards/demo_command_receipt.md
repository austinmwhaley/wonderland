# B/C demo command output

seeds=[0, 1, 2] epochs=5 runtime=875.9s hash=e94ab16cbf2b

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3780 | -0.0022 |
| orders90 | 0.4547 | 0.4393 | -0.0154 |
| churn90 | 0.9305 | 0.9321 | +0.0017 |

| probe | E | table |
|---|---|---|
| retrieval | 0.3965 | — |
| churn-nn | 0.9806 | — |
| identity channel | 0.3996 | 0.2513 |
| identity category | 0.1704 | 0.2653 |
| shuffle value drop | -0.2355 | 0 |
