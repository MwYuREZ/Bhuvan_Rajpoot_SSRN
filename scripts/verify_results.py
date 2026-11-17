"""Audit exported full-run artifacts without refitting or retuning any model."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from readmission.data import eligible_cohort, load_source, make_features, sha256
from readmission.metrics import choose_threshold, operating_metrics, probability_metrics
from readmission.models import select_candidate


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "full"
    runtime = json.loads((out / "runtime.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))
    assert runtime["mode"] == "full"
    for relative, expected in runtime["code_sha256"].items():
        assert sha256(ROOT / relative) == expected, f"Changed scientific source: {relative}"
    assert sha256(out / "selection_lock.json") == runtime["selection_lock_sha256"]
    assert sha256(out / "predictions" / "test_predictions.csv.gz") == runtime["predictions_sha256"]
    assert sha256(out / "predictions" / "development_oof.csv.gz") == lock["oof_file_sha256"]
    assignments = pd.read_csv(out / "splits" / "encounter_assignments.csv.gz")
    assert assignments.encounter_id.is_unique
    assert len(assignments) == 99340
    assert assignments.groupby("patient_nbr").partition.nunique().max() == 1
    dev_rows = assignments[assignments.partition.eq("development")]
    assert dev_rows.groupby("patient_nbr").cv_validation_fold.nunique().max() == 1
    assert set(dev_rows.cv_validation_fold) == {0, 1, 2}
    oof = pd.read_csv(out / "predictions" / "development_oof.csv.gz")
    test = pd.read_csv(out / "predictions" / "test_predictions.csv.gz")
    assert len(oof) == 79398 and len(test) == 19942
    assert set(oof.patient_nbr).isdisjoint(test.patient_nbr)
    cohort, _ = eligible_cohort(load_source(ROOT / "data" / "raw" / "diabetic_data.csv"))
    features, _ = make_features(cohort)
    cohort_index = cohort.set_index("encounter_id").index
    test_positions = cohort_index.get_indexer(test.encounter_id)
    assert np.all(test_positions >= 0)
    candidates = pd.read_csv(out / "tables" / "model_comparison.csv")
    operating = pd.read_csv(out / "tables" / "threshold_metrics.csv")
    for feature_set in ["admission", "discharge"]:
        dev_metrics = []
        for name in ["logistic", "hgb"]:
            key = f"{feature_set}_{name}"
            metrics = probability_metrics(oof.target.to_numpy(), oof[key].to_numpy(), calibration=False)
            dev_metrics.append({"model": name, **metrics})
            row = candidates.loc[candidates.features.eq(feature_set) & candidates.model.eq(name)].iloc[0]
            assert np.isclose(row.dev_ap, metrics["ap"], atol=1e-12)
            model = joblib.load(out / "models" / f"{key}.joblib")
            recomputed = model.predict_proba(features.iloc[test_positions])[:, 1]
            np.testing.assert_allclose(recomputed, test[key], rtol=1e-10, atol=1e-12)
            metrics = probability_metrics(test.target.to_numpy(), test[key].to_numpy())
            for metric, value in metrics.items():
                assert np.isclose(row[f"test_{metric}"], value, atol=1e-10), (key, metric)
        selected = select_candidate(dev_metrics)
        assert lock["selection"][feature_set]["model"] == selected
        for beta in [1, 2]:
            criterion = f"F{beta}"
            threshold = choose_threshold(oof.target, oof[f"{feature_set}_{selected}"], beta)
            assert np.isclose(threshold, lock["selection"][feature_set]["thresholds"][criterion], atol=1e-12)
            metrics = operating_metrics(test.target, test[f"{feature_set}_{selected}"], threshold)
            row = operating.loc[operating.features.eq(feature_set) & operating.criterion.eq(criterion)].iloc[0]
            for metric, value in metrics.items():
                assert np.isclose(row[metric], value, atol=1e-10), (feature_set, criterion, metric)
    samples = pd.read_csv(out / "tables" / "bootstrap_replicates.csv.gz")
    assert samples.replicate.nunique() == 500
    for metric in ["auroc", "ap", "brier", "log_loss"]:
        pivot = samples.pivot(index="replicate", columns="features", values=metric)
        np.testing.assert_allclose(pivot.difference, pivot.discharge - pivot.admission, atol=1e-12)
        summary = pd.read_csv(out / "tables" / "bootstrap_summary.csv")
        for name in pivot:
            row = summary.loc[summary.features.eq(name) & summary.metric.eq(metric)].iloc[0]
            np.testing.assert_allclose([row.lower, row.upper], np.quantile(pivot[name], [.025, .975]), atol=1e-12)
    print("PASS: source/code hashes, patient boundaries, OOF selection/thresholds, saved-model predictions,")
    print("      all reported test metrics, and 500 paired bootstrap differences/percentiles.")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
