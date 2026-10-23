# B/C demo command output

seeds=[0] epochs=5 runtime=187.6s hash=8686fae636f6

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3777 | 0.3397 | -0.0380 |
| orders90 | 0.3924 | 0.3886 | -0.0038 |
| churn90 | 0.9602 | 0.9558 | -0.0044 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4164 | — |
| churn-nn | 0.9800 | — |
| identity channel | 0.3807 | 0.2487 |
| identity category | 0.1642 | 0.2600 |
| shuffle value drop | -0.2753 | 0 |
