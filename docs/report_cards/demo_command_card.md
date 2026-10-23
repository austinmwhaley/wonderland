# B/C demo command output

seeds=[0] epochs=5 runtime=331.0s hash=152e28537839

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3777 | 0.4037 | +0.0260 |
| orders90 | 0.3924 | 0.4085 | +0.0161 |
| churn90 | 0.9602 | 0.9481 | -0.0122 |

| probe | E | table |
|---|---|---|
| retrieval | 0.3838 | — |
| churn-nn | 0.9733 | — |
| identity channel | 0.7278 | 0.2487 |
| identity category | 0.2167 | 0.2600 |
| shuffle value drop | -0.0840 | 0 |
