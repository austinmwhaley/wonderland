# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=868.7s hash=378ef7bc69f7

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3901 | +0.0099 |
| orders90 | 0.4547 | 0.4500 | -0.0047 |
| churn90 | 0.9305 | 0.9328 | +0.0023 |

| probe | E | table |
|---|---|---|
| retrieval | 0.4178 | — |
| churn-nn | 0.9767 | — |
| identity channel | 0.3129 | 0.2513 |
| identity category | 0.1671 | 0.2653 |
| shuffle value drop | -0.2526 | 0 |
