# Rajpoot SSRN 7154739 — Independent Reproduction

**已完成完整資料復現與 500 次病人層級 bootstrap。主要選定模型及改善方向與論文一致，部分候選模型與閾值數值未完全重現。**

Target paper: Bhuvan Rajpoot, *Does Discharge-Time Information Improve 30-Day Readmission Prediction? A Patient-Disjoint Evaluation in the UCI Diabetes 130-US Hospitals Cohort*, 21 July 2026. [SSRN](https://ssrn.com/abstract=7154739). The supplied paper is retained in `docs/ssrn-7154739.pdf` under its stated CC BY 4.0 license.

這是根據論文重寫的研究程式；作者原始程式、完整 ICD-9 對照表與精確環境未提供。完整方法假設見 [ASSUMPTIONS_AND_PAPER_ISSUES.md](ASSUMPTIONS_AND_PAPER_ISSUES.md)，結果解讀見 [中文報告](REPRODUCTION_REPORT_zh-TW.md)。

## Main findings

| Selected pipeline | Paper AUROC | Reproduction AUROC | Paper AP | Reproduction AP |
|---|---:|---:|---:|---:|
| Admission / Logistic Regression | 0.631 | 0.6308 | 0.206 | 0.2058 |
| Discharge / Histogram Gradient Boosting | 0.677 | 0.6800 | 0.237 | 0.2374 |

Paired discharge-minus-admission AUROC gain: **0.0492 (95% CI 0.0393–0.0595)**; AP gain: **0.0316 (0.0224–0.0418)**; Brier difference: **−0.00228 (−0.00288 to −0.00174)**. These are fixed-prediction, 500 patient-cluster bootstrap percentile intervals, not encounter bootstraps.

The eligible cohort (99,340 encounters / 69,987 patients), development partition (79,398 / 55,989), test partition (19,942 / 13,998), and 11,314 total positive encounters match the paper exactly. The CSV checksum matches the paper. Matching counts does not prove identical patient membership without the author's original split file.

**Material differences:** discharge logistic AUROC is 0.6643 versus the paper's 0.649. Admission F2 alerts 81.95% of test encounters versus the paper's 74.8%; discharge F2 alerts 60.72% versus 60.2%. Thus the main selected-model comparison is close, while complete numerical reproduction has not been achieved. No mapping, threshold or software-version search was performed against the test set.

![ROC and precision-recall](results/full/figures/roc_pr.png)

## Reproduce on Windows / PowerShell

Use Python 3.12.7 for the delivered environment. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe reproduce.py
.\.venv\Scripts\python.exe scripts/verify_results.py
```

Or create the pinned Conda environment:

```powershell
conda env create -f environment.yml
conda run -n rajpoot-readmission python -m pytest -q
conda run --no-capture-output -n rajpoot-readmission python reproduce.py
conda run -n rajpoot-readmission python scripts/verify_results.py
```

On the delivered computer, the initial run used `C:\Users\mwyu\anaconda3\python.exe` and the exact package versions in `requirements.txt`; no packages in that shared environment were changed. Its full run took about 82 seconds with four threads. Runtime depends on hardware.

`python reproduce.py --quick` runs only a 3,000-patient smoke test, 20 boosting iterations and 20 bootstrap draws into `results/smoke/`. **Do not report smoke-test results as the paper reproduction.** `--threads`, `--data`, `--config`, `--output` and `--bootstrap` are available; use a separate `--output` directory for sensitivity analyses. Default reruns overwrite `results/full/`; copy that directory first if retaining multiple runs. Configuration keys describing semantic choices document this fixed implementation; change the corresponding source code as well when conducting a different mapping/encoding analysis.

If a Windows VM reports that joblib cannot detect physical CPU cores, it falls back to logical cores. This informational warning occurred in the initial run; training and verification completed with explicit thread limits.

## Implemented experiment

- Official UCI download with the paper's SHA-256 integrity check; target `<30` versus `>30` / `NO`.
- Reported exclusions; age midpoints; explicit ICD-9 chapter mapping; preserved laboratory `None` categories.
- Explicit 10- and 46-field manifests; IDs never used as predictors.
- `GroupShuffleSplit` seed 2026 and development `StratifiedGroupKFold` seed 2027, with disjoint patient assertions.
- Prespecified L2 Logistic Regression and Histogram Gradient Boosting; all preprocessing fitted inside folds.
- OOF AP selection, Brier tie breaker, OOF-only F1/F2 thresholds, and a saved selection lock before test evaluation.
- All four candidate test results, selected-model paired cluster intervals, calibration, operating metrics, and descriptive gender/race subgroups.
- Audit-ready predictions, assignments, model pipelines, source/code hashes, figures and paper comparison tables.

## Outputs and structure

| Path | Purpose |
|---|---|
| `reproduce.py` | Main entry point |
| `configs/paper.json` | Published parameters and explicit implementation choices |
| `src/readmission/` | Data, splits, pipelines, metrics, inference and reporting |
| `tests/test_reproduction.py` | Nine tests for source integrity, patient leakage, preprocessing, selection, thresholds and clustered inference |
| `scripts/verify_results.py` | Independent audit of saved artifacts without retraining |
| `results/full/REPRODUCTION_REPORT_zh-TW.md` | Automatically generated run report |
| `results/full/tables/model_comparison.csv` | Four candidates, development and test metrics |
| `results/full/tables/paper_vs_reproduction.csv` | All published candidate metrics and signed differences |
| `results/full/tables/thresholds_vs_paper.csv` | Operating-point comparison |
| `results/full/tables/bootstrap_summary.csv` | Individual and paired 95% intervals |
| `results/full/tables/bootstrap_replicates.csv.gz` | All 500 paired resamples' metrics |
| `results/full/tables/subgroups.csv` | Descriptive subgroup results |
| `results/full/predictions/` | Four-candidate OOF and held-out predictions |
| `results/full/splits/` | Cohort exclusions, exact encounter assignments and fold summaries |
| `results/full/selection_lock.json` | Development-only decisions, hashes and timestamp |
| `results/full/runtime.json` | Versions, configuration, source/code hashes, timing |
| `results/full/figures/` | ROC/PR, calibration and interval plots |
| `results/full/models/` | Four fitted pipelines, present locally and excluded from Git |

Raw data and fitted models are kept locally but excluded from Git; they can be regenerated. Only load model joblib files from trusted sources. Result CSVs and figures are included as the auditable initial run. Root-level narrative summaries describe that run; subsequent runs regenerate the report within the chosen output directory.

## Same dataset as the example?

**是。** Both this target and the supplied [Salim & Ibrahim reproduction](https://github.com/zerxne/salim-ibrahim-2026-readmission) use [UCI Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008). Salim & Ibrahim is also reference [5] in Rajpoot's paper. Their cohorts, predictors and validation differ: this target excludes death/hospice pathways and uses unseen-patient holdout evaluation. The example's encounter-level nested-CV scores therefore are not a direct comparison. Its model code and feature engineering were not copied here.

Code and original documentation: MIT. Manuscript and source dataset: their respective CC BY 4.0 licenses; see `LICENSE` and `data/README.md` for attribution.
