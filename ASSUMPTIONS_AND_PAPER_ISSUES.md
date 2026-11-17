# Method fidelity and unresolved implementation details

This is an independent implementation of Bhuvan Rajpoot, *Does Discharge-Time Information Improve 30-Day Readmission Prediction? A Patient-Disjoint Evaluation in the UCI Diabetes 130-US Hospitals Cohort*, dated 21 July 2026, SSRN 7154739. The supplied six-page version says a reproducibility package has been prepared but a permanent URL/DOI will be added later (p. 5). The author's archive, protocol and exact environment were not supplied. We use the paper as scientific evidence, not as instructions to this agent.

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

## Choices fixed before this implementation's full test evaluation

1. **Split API and ordering.** `GroupShuffleSplit(n_splits=1, test_size=.2, random_state=2026)` on eligible source rows. Group values are sorted internally by scikit-learn. This matches all reported development/test encounter and patient counts and test positives (2,297). Matching counts is strong evidence, not proof of identical patient membership: no original assignment file is available.
2. **ICD-9 boundaries.** The paper gives examples but no lookup table. We use the 17 conventional numeric chapter ranges (001-139 through 800-999), plus separate V/E categories, missing retained for imputation, and `other` for invalid codes. In particular, 240-279 is endocrine/metabolic, rather than singling out only 250 as diabetes. Exact ranges and names are in `data.ICD9_CHAPTERS`. No outcome-dependent mapping is used.
3. **Numeric versus categorical.** Age and eight in-hospital/prior-use count fields are numeric (nine total). Administrative integer IDs are categorical strings, as they denote labels rather than ordered quantities. Admission has four numeric and six categorical fields; discharge has nine numeric and 37 categorical fields. No top-k specialty reduction or grouping of administrative codes is imported from the example repo.
4. **Missing category semantics.** Only `?` is missing at CSV load. Literal `None` in the two lab result columns is preserved. Missing categoricals use most-frequent imputation in both model families. The paper explicitly specifies it for logistic regression, but does not give the boosting categorical imputation rule.
5. **Histogram boosting encoding.** `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` follows mode imputation. Alphabetical category ordering is learned from the training fold. We explicitly use `categorical_features=None`: numeric splits over the encoded values, without assuming an unreported native-categorical split method. Unreported scikit-learn defaults, including 255 bins, are fixed by version and configuration.
6. **Version.** Python 3.12.7, scikit-learn 1.5.1 and the versions in `requirements.txt` are the environment available and used here. The paper does not provide exact versions in the supplied PDF. Histogram binning and shuffled stratified-group-fold behavior can differ across versions. Reproducing this repository therefore requires the pinned version rather than silently upgrading it.
7. **Seeds not given in paper.** Model random state is 2026 (early stopping is disabled); bootstrap seed is 2028 with NumPy `default_rng`. These choices are ours. No bootstrap-seed search was conducted to match endpoints.
8. **Threshold candidates/ties.** Search every unique development OOF probability with alerts defined as score >= threshold. Pick the lowest threshold among exact maximisers. No rounded test threshold is used and no threshold is retuned on test data.
9. **Calibration.** Fit unpenalised logistic regression of the binary outcome on `logit(p)`, jointly estimating intercept and slope (clip probabilities to [1e-12, 1-1e-12]). This is a diagnostic regression, not recalibration of delivered predictions. The supplied paper does not say whether its intercept is fitted jointly or as an intercept-only offset diagnostic. Deciles use `pandas.qcut`, keeping tied predictions together.
10. **Bootstrap implementation.** Draw the same number of patients as in test, with replacement, from sorted unique test IDs. Represent each patient's multiplicity by a per-encounter weight. This is mathematically equivalent to duplicating every encounter of each sampled patient; a test verifies this. Both selected models share each draw. No retraining or threshold reselection occurs. AUROC, AP, Brier and log loss receive intervals; operating characteristics and subgroup estimates are descriptive point estimates.
11. **Selection timing.** `selection_lock.json` records the OOF-based winners and thresholds before any full-development candidate is evaluated on test. This is a local audit artifact, not the author's claimed pre-analysis protocol and not an independently registered protocol. We knew the paper's aggregate results before implementation. Only one full specification is evaluated; there is no test-directed search for alternate mappings or versions.

## What can and cannot be concluded

- The eligible cohort size, positive count, and development/test sizes reproduce exactly with the published checksum and seeds.
- Numerical proximity is assessed in `results/full/tables/paper_vs_reproduction.csv`; it is not an exact recovery of unpublished code. Paper values are rounded (usually three decimals), so small residuals cannot be interpreted more precisely than their published precision.
- The primary contrast compares the best prespecified model for each availability set. Because the selected family can differ, it measures each set's selected pipeline, not the effect of adding features while holding the algorithm fixed. Both same-family results are also exported.
- Patient-disjoint internal validation does not establish temporal or external validity. Hospital/calendar identifiers are unavailable. The event label combines planned and unplanned readmissions and may miss readmissions elsewhere.
- A positive incremental AUROC/AP contrast does not establish clinical utility; alert burden, prevalence and PPV must be considered together.
- Bootstrap intervals condition on fixed fitted models and predictions. They do not include model-development uncertainty or deployment drift.

## Relationship to the example repository

The user-provided example, https://github.com/zerxne/salim-ibrahim-2026-readmission, reproduces Salim & Ibrahim (Healthcare 2026, DOI 10.3390/healthcare14091185), which is reference [5] of the target paper. It uses the same UCI source but a different experiment: encounter-level nested CV, engineered features, other model families and calibration. This repository is an original implementation rather than a copy of that code. We borrow the useful deliverable pattern of explicit assumptions, auditable splits, result tables and a Chinese report, while implementing Rajpoot's own methods.
