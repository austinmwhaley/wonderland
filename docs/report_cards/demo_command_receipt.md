# B/C demo command output

seeds=[0, 1, 2] epochs=12 runtime=685.0s hash=8086995cad1d

| target | table (XGBoost GPU) | frozen E | margin |
|---|---|---|---|
| value90 | 0.3802 | 0.3445 | -0.0357 |
| orders90 | 0.4547 | 0.3977 | -0.0570 |
| churn90 | 0.9305 | 0.9217 | -0.0088 |

| probe | E | table |
|---|---|---|
| retrieval | 0.2673 | — |
| churn-nn | 0.9217 | — |
| identity channel | 0.2646 | 0.2513 |
| identity category | 0.1671 | 0.2653 |
| shuffle value drop | -0.1718 | 0 |
