# B/C demo command output

seeds=[0] epochs=5 runtime=199.3s hash=05f7c8a80eda

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3777 | 0.3504 | -0.0273 |
| orders90 | 0.3924 | 0.3830 | -0.0093 |
| churn90 | 0.9602 | 0.9562 | -0.0041 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4182 | — |
| churn-nn | 0.9800 | — |
| identity channel | 0.3357 | 0.2487 |
| identity category | 0.1760 | 0.2600 |
| shuffle value drop | -0.2537 | 0 |
