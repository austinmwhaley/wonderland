# B/C demo command output

seeds=[0] epochs=12 runtime=606.9s hash=3e1409d62863

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3575 | 0.3438 | -0.0137 |
| orders90 | 0.4503 | 0.4281 | -0.0221 |
| churn90 | 0.9256 | 0.9352 | +0.0097 |

| probe | E | table |
|---|---|---|
| retrieval | 0.3442 | — |
| churn-nn | 0.9739 | — |
| identity channel | 0.3415 | 0.2513 |
| identity category | 0.1649 | 0.2653 |
| shuffle value drop | -0.2341 | 0 |
