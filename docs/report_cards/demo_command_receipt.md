# B/C demo command output

seeds=[0] epochs=12 runtime=351.0s hash=738880279c0b

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3577 | +0.0002 |
| orders90 | 0.4503 | 0.4281 | -0.0221 |
| churn90 | 0.9256 | 0.9320 | +0.0064 |

| probe | E | table |
|---|---|---|
| retrieval | 0.3997 | — |
| churn-nn | 0.9744 | — |
| identity channel | 0.4480 | 0.2513 |
| identity category | 0.1642 | 0.2653 |
| shuffle value drop | -0.2316 | 0 |
