# Exploratory analysis E1: ranking versus operating point

_Post hoc and outside the Holm family (plan: `docs/protocol/exploratory_analyses.md`, written after the confirmatory results). Frozen proposed detector, 5 seeds; estimate = seed mean, 95 % CI = 300 s block bootstrap (B = 10 000). **Oracle** columns choose the threshold from target labels: upper bounds that no deployed detector could reach. Budget columns alert on the top-scored rows and use no labels. Oracle, budget and "ties split" figures split the rows tied at the cut-off score in proportion (the straight segment of the ROC curve); the frozen threshold and the first FPR at 95 % detection do not, as scored._

## Ranking and operating points

| evaluation / space | attack share | PR-AUC | AP-lift | ROC-AUC | recall at frozen threshold | its FPR | oracle recall, FPR 0.1 % | oracle recall, FPR 1 % | oracle recall, FPR 5 % | recall, top 1 % | recall, top 5 % | FPR at 95 % detection | same, ties split | attacks undetectable below 100 % FPR |
|---|---:|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| cicids2017_test / cic | 6.74 % | 1.000 [0.991, 1.000] | 0.932 [0.849, 0.990] | 1.000 [1.000, 1.000] | 0.997 [0.956, 0.999] | 0.018 % | 0.999 [0.988, 1.000] | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 0.148 [0.066, 0.999] | 0.741 [0.332, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| cicids2017_test / unsw_canonical | 6.74 % | 1.000 [0.981, 1.000] | 0.932 [0.849, 0.989] | 1.000 [0.998, 1.000] | 0.996 [0.955, 0.999] | 0.016 % | 0.997 [0.971, 1.000] | 1.000 [0.992, 1.000] | 1.000 [0.996, 1.000] | 0.148 [0.066, 0.998] | 0.741 [0.332, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.004] |
| unsw_nb15 / unsw_canonical | 4.36 % | 0.236 [0.207, 0.266] | 0.193 [0.164, 0.222] | 0.696 [0.676, 0.716] | 0.313 [0.286, 0.342] | 9.057 % | 0.059 [0.051, 0.067] | 0.227 [0.199, 0.257] | 0.267 [0.240, 0.296] | 0.156 [0.137, 0.184] | 0.260 [0.233, 0.288] | 0.774 [0.766, 0.777] | 0.751 [0.745, 0.757] | 0.027 [0.024, 0.030] |
| csecicids2018 / cic | 7.66 % | 0.238 [0.185, 0.384] | 0.161 [0.085, 0.346] | 0.652 [0.575, 0.843] | 0.087 [0.047, 0.195] | 0.328 % | 0.004 [0.002, 0.009] | 0.179 [0.101, 0.388] | 0.262 [0.151, 0.541] | 0.106 [0.063, 0.215] | 0.221 [0.134, 0.453] | 1.000 [1.000, 1.000] | 0.917 [0.737, 0.934] | 0.408 [0.143, 0.516] |

## Attack families

| evaluation / space | family | rows | recall at frozen threshold | oracle recall, FPR 1 % | recall, top 5 % | ROC-AUC against benign | undetectable below 100 % FPR |
|---|---|---:|---|---|---|---|---|
| cicids2017_test / cic | Bot | 33 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.588 [0.258, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | BruteForce | 418 | 0.975 [0.000, 1.000] | 0.998 [0.988, 1.000] | 0.713 [0.108, 1.000] | 0.999 [0.995, 1.000] | 0.001 [0.000, 0.009] |
| cicids2017_test / cic | DDoS | 5,911 | 0.997 [0.997, 0.997] | 1.000 [1.000, 1.000] | 0.754 [0.319, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | DoS | 7,483 | 0.999 [0.560, 1.000] | 1.000 [0.990, 1.000] | 0.754 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | Heartbleed | 2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | Infiltration | 902 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.639 [0.209, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | PortScan | 149 | 0.895 [0.600, 0.897] | 1.000 [1.000, 1.000] | 0.369 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / cic | WebAttack | 12 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.458 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | Bot | 33 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.599 [0.261, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | BruteForce | 418 | 0.978 [0.000, 1.000] | 0.998 [0.061, 1.000] | 0.715 [0.085, 1.000] | 0.999 [0.591, 1.000] | 0.002 [0.000, 0.800] |
| cicids2017_test / unsw_canonical | DDoS | 5,911 | 0.996 [0.996, 0.996] | 1.000 [1.000, 1.000] | 0.752 [0.318, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | DoS | 7,483 | 1.000 [0.840, 1.000] | 1.000 [1.000, 1.000] | 0.756 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | Heartbleed | 2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | Infiltration | 902 | 0.998 [0.989, 0.998] | 1.000 [1.000, 1.000] | 0.635 [0.175, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| cicids2017_test / unsw_canonical | PortScan | 149 | 0.858 [0.400, 0.861] | 0.972 [0.938, 1.000] | 0.334 [0.000, 1.000] | 0.993 [0.992, 1.000] | 0.013 [0.000, 0.014] |
| cicids2017_test / unsw_canonical | WebAttack | 12 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.473 [0.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| unsw_nb15 / unsw_canonical | Analysis | 2,033 | 0.045 [0.032, 0.058] | 0.007 [0.002, 0.016] | 0.027 [0.019, 0.036] | 0.454 [0.437, 0.467] | 0.025 [0.014, 0.049] |
| unsw_nb15 / unsw_canonical | Backdoor | 1,881 | 0.119 [0.089, 0.181] | 0.042 [0.024, 0.079] | 0.082 [0.058, 0.131] | 0.525 [0.500, 0.577] | 0.055 [0.034, 0.104] |
| unsw_nb15 / unsw_canonical | DoS | 5,502 | 0.171 [0.146, 0.200] | 0.064 [0.051, 0.080] | 0.115 [0.097, 0.137] | 0.593 [0.568, 0.622] | 0.039 [0.032, 0.049] |
| unsw_nb15 / unsw_canonical | Exploits | 27,436 | 0.136 [0.124, 0.147] | 0.031 [0.027, 0.036] | 0.080 [0.071, 0.090] | 0.596 [0.586, 0.607] | 0.033 [0.030, 0.037] |
| unsw_nb15 / unsw_canonical | Fuzzers | 20,965 | 0.328 [0.296, 0.363] | 0.203 [0.171, 0.237] | 0.238 [0.206, 0.273] | 0.740 [0.719, 0.762] | 0.014 [0.012, 0.016] |
| unsw_nb15 / unsw_canonical | Generic | 18,860 | 0.776 [0.724, 0.820] | 0.757 [0.701, 0.804] | 0.768 [0.714, 0.813] | 0.873 [0.844, 0.898] | 0.015 [0.012, 0.019] |
| unsw_nb15 / unsw_canonical | Reconnaissance | 10,118 | 0.081 [0.074, 0.088] | 0.010 [0.008, 0.012] | 0.015 [0.012, 0.018] | 0.678 [0.656, 0.701] | 0.047 [0.042, 0.053] |
| unsw_nb15 / unsw_canonical | Shellcode | 1,458 | 0.273 [0.255, 0.291] | 0.120 [0.104, 0.137] | 0.194 [0.176, 0.213] | 0.717 [0.705, 0.730] | 0.023 [0.017, 0.029] |
| unsw_nb15 / unsw_canonical | Worms | 171 | 0.150 [0.106, 0.197] | 0.123 [0.079, 0.170] | 0.130 [0.086, 0.178] | 0.644 [0.615, 0.674] | 0.008 [0.003, 0.014] |
| csecicids2018 / cic | Bot | 142,921 | 0.360 [0.353, 0.367] | 0.908 [0.893, 0.927] | 0.943 [0.930, 0.959] | 0.991 [0.991, 0.993] | 0.000 [0.000, 0.000] |
| csecicids2018 / cic | BruteForce | 94,197 | 0.600 [0.600, 0.600] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.998 [0.998, 0.998] | 0.000 [0.000, 0.000] |
| csecicids2018 / cic | DDoS | 1,374,148 | 0.128 [0.062, 0.241] | 0.269 [0.163, 0.444] | 0.368 [0.272, 0.527] | 0.815 [0.763, 0.872] | 0.172 [0.107, 0.241] |
| csecicids2018 / cic | DoS | 1,834,209 | 0.006 [0.002, 0.320] | 0.009 [0.004, 0.370] | 0.011 [0.006, 0.415] | 0.484 [0.475, 0.931] | 0.640 [0.000, 0.655] |
| csecicids2018 / cic | Infiltration | 14,671 | 0.553 [0.528, 0.574] | 0.735 [0.715, 0.754] | 0.769 [0.751, 0.784] | 0.950 [0.945, 0.953] | 0.021 [0.017, 0.026] |
| csecicids2018 / cic | WebAttack | 283 | 0.871 [0.804, 0.931] | 0.934 [0.880, 0.980] | 0.940 [0.888, 0.983] | 0.992 [0.986, 0.997] | 0.000 [0.000, 0.000] |

## Ties at the lowest score

| evaluation / space | lowest-score share, attack rows | lowest-score share, benign rows |
|---|---|---|
| cicids2017_test / cic | 0.000 [0.000, 0.001] | 0.985 [0.980, 0.988] |
| cicids2017_test / unsw_canonical | 0.000 [0.000, 0.004] | 0.981 [0.973, 0.985] |
| unsw_nb15 / unsw_canonical | 0.027 [0.024, 0.030] | 0.226 [0.223, 0.228] |
| csecicids2018 / cic | 0.408 [0.143, 0.516] | 0.663 [0.661, 0.666] |

## Checks (every one must hold, or nothing is written)

| evaluation / space | rows | blocks | distinct scores per seed | PR-AUC vs unit record | FPR at 95 % vs unit record |
|---|---:|---:|---|---:|---:|
| cicids2017_test / cic | 221,066 | 75 | 183, 173, 175, 174, 169 | 0.0e+00 | 0.0e+00 |
| cicids2017_test / unsw_canonical | 221,066 | 75 | 194, 191, 191, 198, 185 | 0.0e+00 | 0.0e+00 |
| unsw_nb15 / unsw_canonical | 2,027,278 | 293 | 448, 383, 427, 439, 391 | 0.0e+00 | 0.0e+00 |
| csecicids2018 / cic | 45,147,292 | 1579 | 624, 743, 703, 581, 722 | 0.0e+00 | 0.0e+00 |

Scores SHA-256, row order, groups and alerts at the frozen threshold reproduce the confirmatory records exactly for every seed; PR-AUC and FPR at 95 % detection are recomputed from the float32 scores and differ from the float64 unit records by at most the figures shown.
