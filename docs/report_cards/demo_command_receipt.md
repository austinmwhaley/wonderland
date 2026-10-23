# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=983.4s hash=01c6564f5c29

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3884 | +0.0082 |
| orders90 | 0.4547 | 0.4490 | -0.0057 |
| churn90 | 0.9305 | 0.9228 | -0.0077 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4255 | — |
| churn-nn | 0.9794 | — |
| identity channel | 0.4007 | 0.2513 |
| identity category | 0.1652 | 0.2653 |
| shuffle value drop | -0.2252 | 0 |
