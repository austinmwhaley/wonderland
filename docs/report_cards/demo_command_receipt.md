# B/C demo command output

seeds=[3, 4] epochs=12 runtime=540.2s hash=d30a922af0e0

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3677 | 0.3376 | -0.0300 |
| orders90 | 0.4293 | 0.4187 | -0.0107 |
| churn90 | 0.9417 | 0.9347 | -0.0070 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4089 | — |
| churn-nn | 0.9733 | — |
| identity channel | 0.4751 | 0.2532 |
| identity category | 0.1744 | 0.2735 |
| shuffle value drop | -0.2439 | 0 |
