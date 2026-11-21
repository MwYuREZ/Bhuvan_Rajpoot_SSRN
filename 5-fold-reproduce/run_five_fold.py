"""Evaluate HGB preprocessing methods with stratified, patient-disjoint five-fold cross-validation."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from threadpoolctl import threadpool_limits

from make_folds import FOLD_FILE, INNER, N_FOLDS, PUBLISHED_TEST_ROWS, SEED, SPLIT_VERSION, published_cohort
from run import (
    ADMIN_IDS, BASELINE, HERE, KEYS, MIN_FREQUENCY, ROOT, ROY_COMMIT,
    average_precision_score, build_model, fit_roy, load_source, make_features,
    normalize_model_frame, roc_auc_score, roy_hgb_preprocessor, sha256,
)

ROY_HGB = "roy_hgb_46"
METHODS = [BASELINE, ROY_HGB]
JOIN_KEYS = ["encounter_id", "patient_nbr", "y_true", "Fold"]


def check_reference(assignments, path):
    reference = pd.read_csv(path, usecols=JOIN_KEYS)
    actual = assignments[JOIN_KEYS].sort_values("encounter_id", ignore_index=True)
    expected = reference[JOIN_KEYS].sort_values("encounter_id", ignore_index=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)


def load_folds(path, cohort):
    assignments = pd.read_csv(path)
    expected = cohort[KEYS].rename(columns={"target": "y_true"})
    pd.testing.assert_frame_equal(assignments[expected.columns], expected, check_dtype=False, check_exact=True)
    if not assignments.encounter_id.is_unique or assignments.groupby("patient_nbr").Fold.nunique().max() != 1:
        raise ValueError("The assignment has duplicate encounters or patients split across outer folds")
    sizes = assignments.Fold.value_counts().sort_index()
    if sizes.index.tolist() != list(range(1, N_FOLDS + 1)) or sizes.tolist() != PUBLISHED_TEST_ROWS:
        raise ValueError("The assignment does not match the expected outer fold sizes")
    for fold, column in enumerate(INNER, 1):
        if set(assignments[column]) != {0, 1} or assignments.loc[assignments.Fold.eq(fold), column].any():
            raise ValueError(f"Invalid inner validation membership in fold {fold}")
        if assignments.groupby("patient_nbr")[column].nunique().max() != 1:
            raise ValueError(f"Inner validation leaks patients in fold {fold}")
    return assignments


def fit_fold(base, roy, discharge, config, assignments, fold):
    test = assignments.Fold.eq(fold)
    train = ~test
    labels = assignments.loc[train, "y_true"]
    predictions = assignments.loc[test, JOIN_KEYS].copy()

    baseline = build_model("hgb", discharge, config).fit(base.loc[train], labels)
    predictions[BASELINE] = baseline.predict_proba(base.loc[test])[:, 1]
    print(f"  Baseline HGB AP: {average_precision_score(predictions.y_true, predictions[BASELINE]):.6f}", flush=True)

    roy_train, roy_test = roy.loc[train], roy.loc[test]
    preprocessor = roy_hgb_preprocessor(roy_train, discharge["columns"])
    roy_model = build_model("hgb", discharge, config)["model"]
    predictions[ROY_HGB], encoded = fit_roy(preprocessor, roy_model, roy_train, roy_test, labels)
    return predictions, baseline, len(encoded)


def run(source, fold_file, output, threads, reference):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    cohort = published_cohort(load_source(source))
    assignments = load_folds(fold_file, cohort)
    if reference is not None:
        check_reference(assignments, reference)
    config_path = ROOT / "configs" / "paper.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base, manifests = make_features(cohort)
    discharge = manifests["discharge"]
    columns = discharge["columns"]
    roy = pd.read_csv(source, na_values=["?"], low_memory=False).iloc[cohort.source_row].reset_index(drop=True)
    pd.testing.assert_frame_equal(roy[KEYS[:2]], cohort[KEYS[:2]], check_exact=True)
    roy = normalize_model_frame(roy[columns].astype({name: object for name in ADMIN_IDS}), columns)

    records, oof_parts, baseline_models = [], [], []
    with threadpool_limits(limits=threads):
        for fold in range(1, N_FOLDS + 1):
            print(f"Fold {fold}/{N_FOLDS}", flush=True)
            part, baseline, encoded_count = fit_fold(base, roy, discharge, config, assignments, fold)
            baseline_models.append(baseline)
            oof_parts.append(part)
            for method, encoded in zip(METHODS, [len(columns), encoded_count]):
                records.append({
                    "Fold": fold, "method": method,
                    "train_encounters": len(assignments) - len(part), "test_encounters": len(part),
                    "test_patients": part.patient_nbr.nunique(), "test_positives": int(part.y_true.sum()),
                    "encoded_features": encoded,
                    "test_ap": average_precision_score(part.y_true, part[method]),
                    "test_auroc": roc_auc_score(part.y_true, part[method]),
                })
            print(f"  One-hot HGB AP: {records[-1]['test_ap']:.6f}", flush=True)

    oof = pd.concat(oof_parts).sort_index(ignore_index=True)
    pd.testing.assert_frame_equal(oof[JOIN_KEYS], assignments[JOIN_KEYS], check_exact=True)
    metrics = pd.DataFrame(records)
    baseline_ap = metrics.loc[metrics.method.eq(BASELINE)].set_index("Fold").test_ap
    metrics["ap_delta_vs_bhuvan"] = metrics.test_ap - metrics.Fold.map(baseline_ap)
    summary = metrics.groupby("method", sort=False).agg(
        mean_ap=("test_ap", "mean"), std_ap=("test_ap", "std"), mean_auroc=("test_auroc", "mean"),
        mean_ap_delta_vs_bhuvan=("ap_delta_vs_bhuvan", "mean"),
    )
    summary["pooled_oof_ap"] = [average_precision_score(oof.y_true, oof[name]) for name in summary.index]
    run_info = {
        "source_sha256": sha256(source), "fold_assignments_sha256": sha256(fold_file),
        "roy_commit": ROY_COMMIT, "splitter": "StratifiedGroupKFold", "splitter_sklearn": SPLIT_VERSION,
        "split_seed": SEED, "n_splits": N_FOLDS, "shuffle": True, "group_column": "patient_nbr",
        "training_sklearn": sklearn_version, "config_sha256": sha256(config_path), "threads": threads,
        "cohort_encounters": len(cohort), "cohort_patients": cohort.patient_nbr.nunique(),
        "cohort_positives": int(cohort.target.sum()), "hgb_parameters": baseline_models[0]["model"].get_params(),
        "input_features": columns, "roy_hgb_min_frequency": MIN_FREQUENCY,
        "roy_reference_oof_sha256": sha256(reference) if reference is not None else None,
    }
    output.mkdir(parents=True)
    models = output / "models"
    models.mkdir()
    for fold, model in enumerate(baseline_models, 1):
        joblib.dump(model, models / f"bhuvan_hgb_fold_{fold}.joblib", compress=3)
    metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv")
    oof.to_csv(output / "oof_predictions.csv.gz", index=False)
    (output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "diabetic_data.csv")
    parser.add_argument("--folds-file", type=Path, default=FOLD_FILE)
    parser.add_argument("--output", type=Path, default=HERE / "runs" / f"five_fold_{datetime.now():%Y%m%d_%H%M%S_%f}")
    parser.add_argument("--reference-oof", type=Path, help="Check encounters, labels and fold assignments against a reference OOF file")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.input, args.folds_file, args.output, args.threads, args.reference_oof)
