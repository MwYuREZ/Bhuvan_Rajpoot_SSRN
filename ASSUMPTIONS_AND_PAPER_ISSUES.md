# Methods and implementation assumptions

The implementation follows Bhuvan Rajpoot, *Does Discharge-Time Information Improve 30-Day Readmission Prediction? A Patient-Disjoint Evaluation in the UCI Diabetes 130-US Hospitals Cohort* (21 July 2026, [SSRN 7154739](https://ssrn.com/abstract=7154739)). This document separates the reported methods from implementation choices left unspecified in the manuscript.

## Directly specified and implemented

| Component | Implementation | Paper location |
|---|---|---|
| Source integrity | Exact published CSV SHA-256, fail on mismatch | 3.1 |
| Outcome | `<30` = 1; `>30` and `NO` = 0 | 3.2 |
| Cohort | Remove `Unknown/Invalid` gender and discharge dispositions 11, 13, 14, 19, 20, 21 | 3.2 |
| Exclusions | `weight`, `encounter_id`, `patient_nbr` are never features | 3.2 |
| Semantic recoding | `?` missing; age midpoints; broad ICD-9 diagnosis families | 3.2 |
| Admission predictors | Explicit 10 fields, no diagnosis/medication/in-hospital fields | 3.3 |
| Discharge predictors | All 46 source predictors remaining after exclusions | 3.3 |
| Patient split | 80/20 random group split, seed 2026 | 3.4 |
| Development validation | Three shuffled stratified group folds, seed 2027 | 3.4 |
| Logistic model | L2, C=1, LBFGS, 1000 iterations; median/standardisation, mode/one-hot | 3.4 |
| Histogram boosting | 200 iterations, learning rate .05, 31 leaves, min leaf 30, L2=1, no early stopping | 3.4 |
| Model selection | Highest pooled OOF AP; lower pooled OOF Brier breaks ties | 3.4 |
| Training discipline | Fold-local imputation/encoding; no weighting, resampling, feature selection or tuning | 3.4 |
| Thresholds | F1/F2 maxima on development OOF scores only, F2 primary | 3.5 |
| Test evaluation | All four prespecified candidates, primary comparison uses development winners | 4.2/Table 2 |
| Uncertainty | 500 paired patient-cluster bootstrap samples; percentile 95% intervals | 3.5 |
| Subgroups | Gender/race groups with >=200 test encounters and both outcomes | 3.5 |

## Implementation assumptions

1. **Split API and ordering.** `GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2026)` on eligible source rows. Group values are sorted internally by scikit-learn. Development/test encounter and patient counts and test positives (2,297) match the paper. Patient membership cannot be checked without the original assignment file.
2. **ICD-9 boundaries.** The paper gives examples but no lookup table. We use the 17 conventional numeric chapter ranges (001-139 through 800-999), plus separate V/E categories, missing retained for imputation, and `other` for invalid codes. In particular, 240-279 is endocrine/metabolic, rather than singling out only 250 as diabetes. Exact ranges and names are in `data.ICD9_CHAPTERS`. No outcome-dependent mapping is used.
3. **Numeric versus categorical.** Age and eight in-hospital/prior-use count fields are numeric (nine total). Administrative integer IDs are categorical strings. Admission has four numeric and six categorical fields; discharge has nine numeric and 37 categorical fields. Medical specialties and administrative codes retain their source categories.
4. **Missing category semantics.** Only `?` is missing at CSV load. Literal `None` in the two lab result columns is preserved. Missing categoricals use most-frequent imputation in both model families. The paper explicitly specifies it for logistic regression, but does not give the boosting categorical imputation rule.
5. **Histogram boosting encoding.** `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` follows mode imputation. Alphabetical category ordering is learned from the training fold. We explicitly use `categorical_features=None`: numeric splits over the encoded values, without assuming an unreported native-categorical split method. Unreported scikit-learn defaults, including 255 bins, are fixed by version and configuration.
6. **Version.** Python 3.12.7 and scikit-learn 1.5.1 are used, with package versions pinned in `requirements.txt`. The manuscript does not specify exact versions. Histogram binning and shuffled stratified-group-fold behavior may differ across versions.
7. **Seeds not given in paper.** Model random state is 2026 (early stopping is disabled); bootstrap seed is 2028 with NumPy `default_rng`. The bootstrap seed was fixed without searching for matching interval endpoints.
8. **Threshold candidates/ties.** Search every unique development OOF probability with alerts defined as score >= threshold. Pick the lowest threshold among exact maximisers. No rounded test threshold is used and no threshold is retuned on test data.
9. **Calibration.** Fit unpenalised logistic regression of the binary outcome on `logit(p)`, jointly estimating intercept and slope (clip probabilities to [1e-12, 1-1e-12]). The regression measures calibration; predictions are unchanged. The paper does not say whether its intercept is fitted jointly or as an intercept-only offset diagnostic. Deciles use `pandas.qcut`, keeping tied predictions together.
10. **Bootstrap implementation.** Draw the same number of patients as in test, with replacement, from sorted unique test IDs. Represent each patient's multiplicity by a per-encounter weight. This is mathematically equivalent to duplicating every encounter of each sampled patient; a test verifies this. Both selected models share each draw. No retraining or threshold reselection occurs. AUROC, AP, Brier and log loss receive intervals; operating characteristics and subgroup estimates are descriptive point estimates.
11. **Selection timing.** `selection_lock.json` records OOF-based model and threshold choices before test evaluation. The reproduction was designed with knowledge of the paper's aggregate results. One full specification was evaluated, without test-directed searches over mappings or versions.

## Interpretation

- The eligible cohort size, positive count, and development/test sizes reproduce exactly with the published checksum and seeds.
- Numerical proximity is assessed in `results/full/tables/paper_vs_reproduction.csv`; it is not an exact recovery of unpublished code. Paper values are rounded (usually three decimals), so small residuals cannot be interpreted more precisely than their published precision.
- The primary contrast compares the best prespecified model for each availability set. Because the selected family can differ, it measures each set's selected pipeline, not the effect of adding features while holding the algorithm fixed. Both same-family results are also exported.
- Patient-disjoint internal validation does not establish temporal or external validity. Hospital/calendar identifiers are unavailable. The event label combines planned and unplanned readmissions and may miss readmissions elsewhere.
- A positive incremental AUROC/AP contrast does not establish clinical utility; alert burden, prevalence and PPV must be considered together.
- Bootstrap intervals condition on fixed fitted models and predictions. They do not include model-development uncertainty or deployment drift.

