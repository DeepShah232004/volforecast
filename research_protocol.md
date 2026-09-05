# Research Protocol

## When Does Machine Learning Improve Equity-Volatility Forecasts?

**Status:** Version 1.5; final evaluation rules locked on 2026-09-05.

This document consolidates the original design and the amendments made while
building the data and classical baselines. The three hypotheses remain
confirmatory. Any unlisted model, feature, subgroup, or alternative definition
must be reported as exploratory.

## 1. Research question and contribution

Under what market conditions do tree-based machine-learning models improve on
classical volatility forecasts for historical S&P 500 constituents?

This is a replication-and-extension study, not a claim to invent a new model.
Its contribution is a point-in-time, reproducible comparison focused on
*when* model classes differ: high versus low VIX regimes and forecast windows
with versus without earnings announcements.

## 2. Confirmatory hypotheses

- **H1:** Random Forest and XGBoost outperform GARCH during high-VIX regimes.
- **H2:** GARCH remains practically comparable to ML during low-VIX regimes.
- **H3:** Every model's forecast loss is higher for forward windows containing
  an earnings announcement than for windows without one. Model rankings are
  also reported for both groups.

QLIKE is the primary loss; RMSE and MAE are secondary. For H1, each ML model is
compared separately with GARCH. A model supports H1 when its mean high-regime
QLIKE is lower and the 95% block-bootstrap confidence interval for its loss
improvement excludes zero.
H1 is fully supported only if both ML models satisfy this rule; one qualifying
model is reported as partial support.

For H2, define the relative ML improvement as
`(QLIKE_GARCH - QLIKE_ML) / QLIKE_GARCH`. GARCH is practically comparable
to an ML model when the upper endpoint of that model's 95% confidence interval
for improvement is below 5%; a GARCH win also qualifies. H2 is fully supported
only if this holds against both ML models.

For H3, each model is tested separately using the difference between mean
earnings-window and non-earnings-window QLIKE. Full support requires a positive
difference whose 95% confidence interval excludes zero for every model. The
model-ranking clause is secondary and descriptive.

## 3. Data and population

| Source | Role | Locked implementation |
|---|---|---|
| CRSP | Prices, total returns, membership | `crsp_a_stock.stkdlysecuritydata`; 2005-2024 |
| CRSP indexes | Historical universe | `crsp_a_indexes.dsp500list_v2` |
| CCM | Date-correct PERMNO-GVKEY links | Link types LC/LU; primary links P/C |
| IBES | Quarterly EPS announcement dates | `ibes.act_epsus`; QTR/EPS; date-correct score 1-2 CRSP link |
| Cboe via WRDS | Daily VIX close | `cboe.cboe`; origin-date value only |

The universe contains 957 unique historical S&P 500 PERMNOs across 979
membership spells. CRSP provides usable price history for 947 of them:
3,517,630 unique security-date rows, of which 2,510,573 occur during actual
membership.

Full pre-membership price history may be used to compute a member's features
and per-company model state. Training and evaluation origins, however, must
have `in_sp500_membership == True`. This prevents scoring non-members or
training the pooled ML models on firms selected only because they join later.

Multiple membership spells are treated as a union of intervals and may never
duplicate a security-date row. Features and targets are computed only after
enforcing unique `(PERMNO, date)` keys.

Current auxiliary data:

- 56,308 linked IBES records covering 954 universe PERMNOs.
- 5,052 unique VIX closes from 2005-01-03 through 2024-12-31.
- All 5,033 CRSP trading dates have an observed VIX close; no VIX values are
  imputed.
- Compustat fundamentals, Fama-French factors, sector splits, and market-cap
  splits are out of scope for the confirmatory Version 1 analysis.

Raw WRDS data and generated Parquet artifacts are not committed because of
licensing and size. Pull and transformation code is version-controlled.

## 4. Target and forecast timing

The target is **daily-return-based volatility**, not intraday realized
volatility:

> The sample standard deviation of 21 CRSP daily total returns, expressed in
> daily decimal-return units and not annualized.

At each security-date origin `t`, the target uses the next 21 observed
security trading rows, `(t, t+21]`. Every feature and model input must be
known by the close of `t`.

The trailing feature window ending at `t` does not overlap its own forward
target. Targets from consecutive origins do overlap, so observations cannot be
treated as independent for validation or inference.

Current target data contain 3,497,736 valid trailing-volatility observations.
The March 2020 cross-sectional average rose from 0.01698 in 2019 Q4 to a peak
of 0.08767, a 5.16x increase.

## 5. Regime and event definitions

### VIX regimes

Thresholds are the 33rd and 67th percentiles of VIX close calculated using only
2005-2014:

- Low: VIX <= 14.306667
- Middle: 14.306667 < VIX < 20.756667
- High: VIX >= 20.756667

The VIX close is joined on forecast origin `t`. Thresholds are never
recomputed using 2015-2024 data. H1 and H2 use the high and low groups;
middle-regime results are descriptive.

### Earnings windows

A forecast is an earnings window when the firm has at least one unique IBES
announcement date in `(t, target_end_date]`. An announcement on `t` is
excluded; one on the target end date is included. This is an ex-post
evaluation label and is never an ML feature.

Same-firm, same-day IBES records are collapsed before counting. A missing
target end date produces a missing event label, not `False`.

## 6. Models

### Classical baselines

1. Persistence: trailing 21-day volatility at `t`.
2. EWMA: RiskMetrics daily decay, lambda = 0.94.
3. Zero-mean Gaussian GARCH(1,1), estimated separately by PERMNO.

GARCH is re-estimated each calendar year using all observations through the
prior year, with at least 252 valid training returns. Fits must satisfy
`omega > 0`, non-negative ARCH/GARCH coefficients, and `alpha + beta < 1`.
A fit is rejected when its unconditional variance is outside 0.05-20 times
the training sample variance.

Each new calendar year is initialized with the one-step-ahead conditional
variance from that year's expanding-window fit. The filtered state and all 21
forecast steps then use the same parameter set. Invalid fits yield missing
forecasts; outputs are not economically clipped.

### Machine-learning models

- Random Forest
- XGBoost

Both are pooled panel models predicting log forward volatility. Predictions
are exponentiated, guaranteeing positive volatility forecasts. GARCH is a
competitor, not an ML input.

Locked predictors, all measured through `t`:

- Trailing return standard deviation: 5, 21, 63, 126, and 252 days.
- Mean absolute return: 5, 21, and 63 days.
- Compounded return: 5, 21, and 63 days.
- EWMA volatility with lambda = 0.94.
- Origin-date VIX close.
- Log VIX change over the previous 1, 5, and 21 CRSP trading dates.

Firm identifiers, future earnings labels, GARCH forecasts, prices,
fundamentals, sectors, and market capitalization are excluded. Rows missing
any locked predictor or target are excluded consistently from both ML models
and their final classical comparison sample; no feature imputation is used.

### Tuning

All model selection minimizes validation QLIKE. Random seed is 42.

| Model | Locked search |
|---|---|
| Random Forest | 400 trees; max depth {8, 16, unlimited}; min leaf {20, 100}; max features {sqrt, 0.5}; bootstrap enabled |
| XGBoost | squared-error objective on log volatility; histogram trees; 1,500-tree cap with 75-round early stopping; learning rate {0.03, 0.08}; max depth {3, 6}; min child weight {5, 20}; subsample 0.8; column subsample 0.8; L2 = 1 |

XGBoost early stopping minimizes validation RMSE on the log-volatility target,
which matches its squared-error training objective. The retained boosting round
from each candidate is then scored in volatility units, and the candidate with
the lowest validation QLIKE is selected. Random Forest candidates are likewise
selected by validation QLIKE. Exact ties are broken by the fixed grid order.

No additional model or feature is added because of test-period performance.

## 7. Splits and leakage controls

- Training origins: 2005-2012.
- Validation origins: 2013-2014.
- Final test origins: 2015-2024.

Training rows whose target ends in 2013 or later are removed. Validation rows
whose target ends in 2015 or later are removed. Thus no forward target crosses
a split boundary during tuning. Hyperparameters are selected once on
validation data and then refit on all finite member-origin rows whose target
ends before 2015. The train-validation boundary purge is no longer needed for
this combined refit, so otherwise-valid late-2012 rows are re-admitted; labels
reaching 2015 remain excluded. The refitted models then produce one final
2015-2024 forecast set. For XGBoost, the boosting-round count selected by
validation early stopping is held fixed during the final refit.

The test period is never used for feature selection, tuning, early stopping,
missing-value decisions, or regime thresholds.

## 8. Evaluation and inference

All models are compared on the same valid `(PERMNO, origin)` rows. Metrics
are pooled equally across eligible constituent-day forecasts.

Because firms share market shocks and consecutive 21-day targets overlap,
inference operates on the cross-sectional mean loss differential for each
forecast date:

- Primary uncertainty method: moving-block bootstrap over dates, block length
  21 trading days, 2,000 replicates, seed 42.
- Secondary robustness: Diebold-Mariano-style mean-loss test on the
  date-aggregated differential with Newey-West lag 20.

The same date-block bootstrap produces 95% confidence intervals for RMSE, MAE,
QLIKE, H1-H2 model differences, and H3 event/non-event differences.

Headline metrics are pooled equally across constituent-day rows. Their
bootstrap intervals resample non-circular blocks of 21 consecutive dates from
the full test trading calendar and recompute the pooled statistic using all
rows on each sampled date. VIX and earnings conditions are applied inside each
replicate, preserving the clustering and spacing of those conditions.

For H1 and H2, security-level QLIKE differences are averaged within date and
then equally across eligible dates. Positive `GARCH loss - ML loss` favors ML.
H2 divides this date-equal mean difference by date-equal mean GARCH QLIKE on
low-VIX dates.
For H3, the daily series is the within-date mean earnings-window QLIKE minus
the within-date mean non-earnings QLIKE. Newey-West inference with lag 20 is
applied to each finite conditional daily-difference series.

## 9. Interpretability

SHAP is computed only after final model selection on a fixed 10,000-row test
sample stratified by calendar year and VIX regime, seed 42. Report global SHAP
importance and stability between 2015-2019 and 2020-2024. Interpret correlated
volatility features as groups; do not claim causal effects.

## 10. Current checkpoint

Phase 2 is complete. The common 2015-2024 classical sample has 1,207,514 rows,
682 PERMNOs, and forecast origins through 2024-11-29.

| Model | RMSE | MAE | QLIKE |
|---|---:|---:|---:|
| Persistence | 0.010503220 | 0.006352679 | 0.590950275 |
| EWMA | 0.009776251 | 0.005932379 | 0.470315067 |
| GARCH(1,1) | **0.009321981** | **0.005909793** | **0.379852437** |

The evaluation panel contains 413,797 low-VIX, 461,617 middle-VIX, and
332,100 high-VIX common rows. It contains 404,778 earnings-window and 802,736
non-earnings-window rows. Conditional losses have not been inspected before
locking the Phase 3 specification.

The Phase 3 feature matrix is also complete. After requiring all locked
predictors, a valid log target, actual membership at the forecast origin, and
the split-boundary purge, it contains 859,540 training rows, 238,982 validation
rows, and 1,234,643 test rows. Intersecting the test rows with valid forecasts
from all three classical models leaves 1,206,821 final-comparison rows across
682 PERMNOs.

All 12 Random Forest and eight XGBoost candidates have been evaluated on the
locked validation sample. The QLIKE-selected specifications are now frozen:

| Model | Selected specification | Validation RMSE | Validation MAE | Validation QLIKE |
|---|---|---:|---:|---:|
| Random Forest | 400 trees; depth 8; min leaf 20; max features sqrt | 0.005315932 | 0.003426376 | **0.272853127** |
| XGBoost | learning rate 0.03; depth 6; min child weight 20; 205 rounds | 0.005325615 | 0.003404988 | **0.283479163** |

Random Forest's validation QLIKE is 3.748% lower than XGBoost's. This is a
model-selection result only, not evidence for H1 or H2. Test-period losses and
conditional hypothesis results remain unopened.

Both selected specifications have now been refit on 1,108,846 combined
pre-2015 rows. Each produced 1,234,643 positive, finite test forecasts with
identical keys and targets. The final models and forecast files passed
round-trip validation; test-period losses remain unopened.

Next: build the 1,206,821-row all-model comparison panel, run the locked final
evaluation once, and report H1-H3 without specification changes.

## 11. Amendment log

| Version | Locked amendment |
|---|---|
| 1.0 | Removed options platform, dashboard, crypto extension, and LSTM; fixed H1-H3, model families, and three losses. |
| 1.1 | Replaced the leakage-adjacent next trailing-window task with the genuine forward 21-return target; aligned H3 to that window. |
| 1.2 | Retained pre-membership history for state/features while scoring only membership rows; adopted annual expanding GARCH fits and fit-time variance diagnostics. |
| 1.3 | Fixed repeat-membership row expansion; made GARCH origin parameters and annual state initialization internally consistent; locked training-only VIX tertiles, event boundaries, ML features, splits, and inference. |
| 1.4 | Fixed the ML execution rule: log-RMSE early stopping, QLIKE hyperparameter selection, deterministic tie-breaking, and a pre-2015 final refit that re-admits eligible train-boundary rows. |
| 1.5 | Locked full-calendar non-circular date-block resampling, pooled-metric intervals, date-equal H1-H2 loss differences, and within-date H3 event differences before test evaluation. |

## 12. Success criteria and limitations

Success means evaluating all three hypotheses with a reproducible pipeline and
reporting the result regardless of whether ML wins.

Pre-specified limitations:

- The target is daily-return volatility, not intraday realized volatility.
- Consecutive 21-day targets overlap.
- Results concern historical S&P 500 constituents, not the full US market or
  other asset classes.
- The study produces point forecasts, not calibrated prediction intervals.
- Only tree-based ML is studied; no deep sequence model is included.
- Actual earnings dates are used only for ex-post stratification.

Out of scope for Version 1: additional model families, alternative horizons,
sector/size subgroup searches, fundamentals, Fama-French factors, prediction
intervals, dashboards, options-surface work, and crypto comparisons.
