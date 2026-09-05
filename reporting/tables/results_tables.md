# Locked Results Tables

Generated only from the committed final-evaluation and SHAP summary artifacts.
Lower values are better for RMSE, MAE, and QLIKE.

## Table 1. Final common test sample

| Quantity | Value |
|---|---:|
| Security-date forecasts | 1,206,821 |
| Securities | 682 |
| Forecast-origin dates | 2,495 |
| Origin range | 2015-01-02 to 2024-11-29 |
| SHAP sample rows | 10,000 |
| SHAP sample securities | 665 |

## Table 2. Overall out-of-sample performance

| Model | RMSE [95% CI] | MAE [95% CI] | QLIKE [95% CI] | RMSE vs GARCH | QLIKE vs GARCH |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.010497 [0.008612, 0.012743] | 0.006351 [0.005653, 0.007291] | 0.590578 [0.441757, 0.849437] | -12.67% | -55.50% |
| EWMA | 0.009769 [0.008012, 0.012122] | 0.005929 [0.005279, 0.006827] | 0.469894 [0.343917, 0.696641] | -4.85% | -23.73% |
| GARCH(1,1) | 0.009317 [0.007636, 0.011727] | 0.005906 [0.005369, 0.006751] | 0.379788 [0.282894, 0.564896] | — | — |
| Random Forest | 0.008530 [0.006578, 0.011248] | 0.004897 [0.004310, 0.005801] | 0.404345 [0.267574, 0.655924] | +8.45% | -6.47% |
| XGBoost | 0.008438 [0.006546, 0.011082] | 0.004859 [0.004286, 0.005750] | 0.396821 [0.265832, 0.638268] | +9.43% | -4.48% |

Positive relative changes indicate improvement over GARCH. Overall changes are descriptive; no paired overall superiority test was preregistered.

## Table 3. Pooled QLIKE by VIX regime

| Model | Low VIX | Middle VIX | High VIX |
|---|---:|---:|---:|
| Persistence | 0.632099 | 0.568786 | 0.569110 |
| EWMA | 0.481763 | 0.461217 | 0.467160 |
| GARCH(1,1) | 0.361689 | 0.356784 | 0.434317 |
| Random Forest | 0.385422 | 0.399702 | 0.434383 |
| XGBoost | 0.384528 | 0.392026 | 0.418805 |

These pooled subgroup losses are descriptive. H1 and H2 decisions use the preregistered date-equal paired differences below.

## Table 4. Regime hypothesis tests against GARCH

| ML model | High-VIX QLIKE advantage [95% CI] | NW p | Low-VIX relative advantage [95% CI] | NW p |
|---|---:|---:|---:|---:|
| Random Forest | 0.00025 [-0.06053, 0.03436] | 0.992 | -6.47% [-22.54%, 9.41%] | 0.454 |
| XGBoost | 0.01563 [-0.01518, 0.04221] | 0.268 | -6.22% [-22.71%, 10.09%] | 0.481 |

H1 decision: **not supported**. H2 decision: **full support under the locked rule**.

## Table 5. Earnings-window QLIKE penalty

| Model | Date-matched penalty | 95% block-bootstrap CI | NW p |
|---|---:|---:|---:|
| Persistence | 0.58337 | [0.36892, 0.97079] | 0.0002156 |
| EWMA | 0.30814 | [0.23715, 0.39990] | 4.601e-13 |
| GARCH(1,1) | 0.07267 | [0.03166, 0.12587] | 0.001951 |
| Random Forest | 0.28099 | [0.19777, 0.41235] | 1.72e-07 |
| XGBoost | 0.26000 | [0.19776, 0.34340] | 4.253e-12 |

H3 decision: **full support**. Positive values mean higher QLIKE for earnings-window forecasts.

## Table 6. Grouped SHAP importance

| Feature group | Random Forest | XGBoost |
|---|---:|---:|
| Trailing volatility | 83.27% | 74.31% |
| VIX changes | 6.98% | 11.67% |
| Compounded returns | 7.04% | 10.11% |
| VIX level | 2.72% | 3.91% |

## Table 7. Leading individual SHAP features

| Model | Rank | Feature | Importance share |
|---|---:|---:|---:|
| Random Forest | 1 | Mean \|return\| (63d) | 26.30% |
| Random Forest | 2 | Mean \|return\| (21d) | 13.80% |
| Random Forest | 3 | Return stdev (63d) | 10.46% |
| Random Forest | 4 | EWMA volatility | 10.33% |
| Random Forest | 5 | Return stdev (126d) | 6.86% |
| XGBoost | 1 | Mean \|return\| (63d) | 38.81% |
| XGBoost | 2 | Mean \|return\| (21d) | 13.23% |
| XGBoost | 3 | Compounded return (63d) | 6.93% |
| XGBoost | 4 | VIX change (21d) | 6.55% |
| XGBoost | 5 | Return stdev (252d) | 5.89% |

## Table 8. SHAP stability across subperiods

| Model | Feature-rank Spearman | Shared top-five features | Group-rank Spearman |
|---|---:|---:|---:|
| Random Forest | 0.965 | 5 | 0.800 |
| XGBoost | 0.968 | 5 | 1.000 |

SHAP values explain log-volatility predictions. Importance is descriptive, and individual rankings among correlated volatility measures should not be interpreted causally.

Source SHA-256 (evaluation): `49d4ff213ddf871678e378f0c048860edfa528e563ccdc24c310ce02fe179c0d`  
Source SHA-256 (SHAP): `89aa3ac576866de9f8ff97d9de1128355e8873ca1ba9910a384144032227e98e`
