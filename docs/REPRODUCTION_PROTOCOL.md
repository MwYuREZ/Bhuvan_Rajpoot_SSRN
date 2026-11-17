# Reproduction protocol

This protocol follows the methods in Rajpoot (2026), SSRN 7154739. Implementation assumptions are documented separately. The paper's aggregate results were known when this reproduction was designed.

1. Verify the UCI CSV against the checksum published in the paper.
2. Apply only the reported death/hospice and invalid-gender exclusions.
3. Convert age intervals and broad ICD-9 families deterministically without labels.
4. Keep all 10 admission fields and all 46 eligible discharge fields.
5. Make the seed-2026 patient-group holdout. Check published sample counts; assert zero overlap.
6. Make seed-2027 shuffled three-fold stratified group validation within development.
7. Fit the two prespecified candidates separately for each feature set. Fit imputation, encoding and scaling using training-fold data only. Pool OOF predictions.
8. Select by OOF AP, with Brier as the tie breaker. For each winner, select F1/F2 thresholds from OOF predictions. Save selection, thresholds, configuration, OOF hash and code hashes before test evaluation.
9. Refit all four candidates on full development, evaluate each once on test, and save predictions and fitted pipelines.
10. For the two development winners, compute paired cluster-bootstrap intervals from 500 fixed-prediction patient resamples with seed 2028. Use the same draws for both models and retain patient encounter clusters.
11. Export operating characteristics, calibration deciles, subgroup estimates and figures.
12. Compare against transcribed paper values without changing modeling choices to improve agreement.

Unreported choices are fixed in `configs/paper.json` and explained in `ASSUMPTIONS_AND_PAPER_ISSUES.md`. A changed specification is a new sensitivity analysis and should use a separate output directory. `--quick` is only a workflow check and cannot support a reproduction claim.
