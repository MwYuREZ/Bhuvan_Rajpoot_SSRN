"""Run from raw UCI data through one locked patient-disjoint evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .data import (download_data, eligible_cohort, load_source, make_features,
                   make_splits, sha256)
from .metrics import (choose_threshold, operating_metrics, paired_cluster_bootstrap,
                      probability_metrics)
from .models import build_model, select_candidate
from .reporting import paper_comparison, write_figures, write_report

ROOT = Path(__file__).resolve().parents[2]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def describe(cohort):
    return {"encounters": len(cohort), "patients": int(cohort.patient_nbr.nunique()),
            "positives": int(cohort.target.sum()), "prevalence": float(cohort.target.mean())}


def code_hashes():
    files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "configs").glob("*.json"))
    files += list((ROOT / "tests").glob("*.py"))
    files += [ROOT / "reproduce.py", ROOT / "requirements.txt"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(files)}


def run(args):
    started = time.monotonic()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.quick:
        config["hgb"]["max_iter"] = 20
        config["bootstrap_replicates"] = 20
    if args.bootstrap is not None:
        if args.bootstrap < 2:
            raise ValueError("At least two bootstrap replicates are required")
        config["bootstrap_replicates"] = args.bootstrap
    out = args.output or ROOT / "results" / ("smoke" if args.quick else "full")
    out.mkdir(parents=True, exist_ok=True)
    for folder in ["tables", "predictions", "splits", "models"]:
        (out / folder).mkdir(exist_ok=True)
    source_path = args.data or download_data(ROOT / "data" / "raw")
    runtime = {"started_utc": datetime.now(timezone.utc).isoformat(),
               "mode": "smoke" if args.quick else "full", "config": config,
               "python": platform.python_version(), "platform": platform.platform(),
               "packages": {p: importlib.metadata.version(p) for p in [
                   "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
               "source_sha256": sha256(source_path), "code_sha256": code_hashes(),
               "threads": args.threads}
    dump(out / "runtime.json", runtime)
    source = load_source(source_path)
    cohort, audit = eligible_cohort(source)
    audit.to_csv(out / "splits" / "cohort_audit.csv.gz", index=False)
    if args.quick:
        patients = np.random.default_rng(2026).choice(cohort.patient_nbr.unique(), size=3000, replace=False)
        cohort = cohort.loc[cohort.patient_nbr.isin(patients)].reset_index(drop=True)
    x, manifest = make_features(cohort)
    dump(out / "feature_manifest.json", manifest)
    dev, test, folds = make_splits(cohort, config)
    summary = {"eligible": describe(cohort), "development": describe(cohort.iloc[dev]),
               "test": describe(cohort.iloc[test])}
    reference = json.loads((ROOT / "docs" / "paper_reference.json").read_text(encoding="utf-8"))
    if not args.quick:
        expected = reference["cohort"]
        actual = {"source_encounters": len(source), "eligible_encounters": len(cohort),
                  "eligible_patients": summary["eligible"]["patients"],
                  "positives": summary["eligible"]["positives"],
                  "development_encounters": len(dev), "development_patients": summary["development"]["patients"],
                  "test_encounters": len(test), "test_patients": summary["test"]["patients"]}
        if actual != expected:
            raise AssertionError(f"Cohort/split counts differ from paper: {actual}")
    dump(out / "cohort_summary.json", summary)
    assignments = cohort[["source_row", "encounter_id", "patient_nbr", "target"]].copy()
    assignments["partition"] = "development"
    assignments.loc[test, "partition"] = "test"
    assignments["cv_validation_fold"] = -1
    fold_rows = []
    for fold, (train, val) in enumerate(folds):
        assignments.loc[dev[val], "cv_validation_fold"] = fold
        for part, idx in [("train", train), ("validation", val)]:
            fold_rows.append({"fold": fold, "partition": part, **describe(cohort.iloc[dev[idx]])})
    assignments.to_csv(out / "splits" / "encounter_assignments.csv.gz", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "splits" / "fold_summary.csv", index=False)
    y_dev = cohort.target.to_numpy()[dev]
    y_test = cohort.target.to_numpy()[test]
    oof = assignments.iloc[dev].copy()
    test_predictions = assignments.iloc[test].copy()
    dev_records = []
    selection = {}

    print(f"{runtime['mode']}: {len(cohort):,} encounters; development {len(dev):,}; test {len(test):,}", flush=True)
    # Complete ALL development selection before evaluating ANY test prediction.
    for feature_set, feature_info in manifest.items():
        for name in ["logistic", "hgb"]:
            key = f"{feature_set}_{name}"
            predictions = np.full(len(dev), np.nan)
            for fold, (train, val) in enumerate(folds):
                print(f"OOF {key} fold {fold + 1}/{len(folds)}", flush=True)
                model = build_model(name, feature_info, config)
                model.fit(x.iloc[dev[train]], y_dev[train])
                predictions[val] = model.predict_proba(x.iloc[dev[val]])[:, 1]
            if not np.all(np.isfinite(predictions)):
                raise AssertionError("Missing/nonfinite OOF predictions")
            oof[key] = predictions
            metrics = probability_metrics(y_dev, predictions)
            dev_records.append({"features": feature_set, "model": name, **metrics})
            print(f"  development AP={metrics['ap']:.6f}, AUROC={metrics['auroc']:.6f}", flush=True)
        winner = select_candidate([r for r in dev_records if r["features"] == feature_set])
        selected_oof = oof[f"{feature_set}_{winner}"].to_numpy()
        selection[feature_set] = {"model": winner,
            "thresholds": {f"F{beta}": choose_threshold(y_dev, selected_oof, beta) for beta in [1, 2]}}
    oof.to_csv(out / "predictions" / "development_oof.csv.gz", index=False)
    selection_lock = {"locked_utc": datetime.now(timezone.utc).isoformat(), "selection": selection,
                      "development_metrics": dev_records, "config": config,
                      "oof_file_sha256": sha256(out / "predictions" / "development_oof.csv.gz"),
                      "code_sha256": runtime["code_sha256"],
                      "note": "Written after OOF selection and before fitting/evaluating test models. This is a reproduction lock, not the author's unavailable protocol."}
    dump(out / "selection_lock.json", selection_lock)
    print(f"Development selections locked: {selection}", flush=True)

    records = []
    for row in dev_records:
        feature_set, name = row["features"], row["model"]
        key = f"{feature_set}_{name}"
        print(f"Full development fit and held-out evaluation: {key}", flush=True)
        model = build_model(name, manifest[feature_set], config)
        model.fit(x.iloc[dev], y_dev)
        p = model.predict_proba(x.iloc[test])[:, 1]
        test_predictions[key] = p
        joblib.dump(model, out / "models" / f"{key}.joblib", compress=3)
        records.append({"features": feature_set, "model": name,
            "selected": selection[feature_set]["model"] == name,
            **{f"dev_{k}": v for k, v in row.items() if k not in {"features", "model"}},
            **{f"test_{k}": v for k, v in probability_metrics(y_test, p).items()}})
    test_predictions.to_csv(out / "predictions" / "test_predictions.csv.gz", index=False)
    candidates = pd.DataFrame(records)
    candidates.to_csv(out / "tables" / "model_comparison.csv", index=False)
    selected_predictions = {name: test_predictions[f"{name}_{decision['model']}"].to_numpy()
                            for name, decision in selection.items()}
    operating = pd.DataFrame([{"features": name, "model": selection[name]["model"], "criterion": criterion,
        **operating_metrics(y_test, p, threshold)} for name, p in selected_predictions.items()
        for criterion, threshold in selection[name]["thresholds"].items()])
    operating.to_csv(out / "tables" / "threshold_metrics.csv", index=False)
    print(f"Paired patient-cluster bootstrap: {config['bootstrap_replicates']} replicates", flush=True)
    samples, intervals = paired_cluster_bootstrap(y_test, selected_predictions,
        cohort.patient_nbr.to_numpy()[test], config["bootstrap_replicates"], config["bootstrap_seed"])
    samples.to_csv(out / "tables" / "bootstrap_replicates.csv.gz", index=False)
    intervals.to_csv(out / "tables" / "bootstrap_summary.csv", index=False)
    subgroup_rows = []
    test_cohort = cohort.iloc[test]
    for column in ["gender", "race"]:
        for value in test_cohort[column].dropna().unique():
            mask = test_cohort[column].eq(value).to_numpy()
            if mask.sum() < config["subgroup_min_encounters"] or len(np.unique(y_test[mask])) < 2:
                continue
            for name, p in selected_predictions.items():
                subgroup_rows.append({"attribute": column, "group": value, "features": name,
                    "encounters": int(mask.sum()), "patients": int(test_cohort.loc[mask, "patient_nbr"].nunique()),
                    "positives": int(y_test[mask].sum()), "prevalence": float(y_test[mask].mean()),
                    **probability_metrics(y_test[mask], p[mask], calibration=False),
                    **operating_metrics(y_test[mask], p[mask], selection[name]["thresholds"]["F2"])})
    pd.DataFrame(subgroup_rows).to_csv(out / "tables" / "subgroups.csv", index=False)
    if not args.quick:
        paper_comparison(out, candidates, operating, intervals, reference)
    write_figures(out, y_test, selected_predictions, intervals)
    runtime["elapsed_seconds"] = time.monotonic() - started
    runtime["completed_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["selection_lock_sha256"] = sha256(out / "selection_lock.json")
    runtime["predictions_sha256"] = sha256(out / "predictions" / "test_predictions.csv.gz")
    dump(out / "runtime.json", runtime)
    write_report(out, candidates, operating, intervals, summary, selection, runtime, args.quick)
    print(candidates[["features", "model", "selected", "dev_ap", "test_auroc", "test_ap", "test_brier"]].to_string(index=False), flush=True)
    print(intervals.loc[intervals.features.eq("difference")].to_string(index=False), flush=True)
    print(f"Completed in {runtime['elapsed_seconds']:.1f}s: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.json")
    parser.add_argument("--data", type=Path, help="An existing unmodified diabetic_data.csv; checksum required")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, help="Override replicates; paper default is 500")
    parser.add_argument("--quick", action="store_true", help="3000 patients, 20 HGB iterations, 20 bootstraps; NOT paper results")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    warnings.filterwarnings("error", category=ConvergenceWarning)
    with threadpool_limits(limits=args.threads):
        run(args)
