"""Re-fit the selected HGB/CatBoost configurations on the paper's development set."""

from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256 as hash_bytes
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from expanded_features import DIAGNOSIS, Encoding, build_expanded
from readmission.data import eligible_cohort, load_source, make_features, sha256
from readmission.metrics import cluster_weights, probability_metrics

PAPER = ROOT / "results" / "full"
SOURCE = ROOT / "data" / "raw" / "diabetic_data.csv"
ASSIGNMENTS = PAPER / "splits" / "encounter_assignments.csv.gz"
BASELINE = PAPER / "predictions" / "test_predictions.csv.gz"
ASSIGNMENT_CONTENT_SHA256 = "8a915bb0c46a3c384b1ed939afa1a7d093a77c0f949312bdfce324933e545d50"
KEYS = ["encounter_id", "patient_nbr", "target"]
THREADS = 4
SEED = 2026
BOOTSTRAP_SEED = 2028
REPETITIONS = 1000
SKLEARN = {"46_fields": "1.5.1", "expanded": "1.8.0"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_split():
    runtime = read_json(PAPER / "runtime.json")
    assert sha256(SOURCE) == runtime["source_sha256"]
    assert sha256(BASELINE) == runtime["predictions_sha256"]
    assignment = pd.read_csv(ASSIGNMENTS)
    # A regenerated gzip file can have a different timestamp but the same split.
    assignment_content = assignment.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assert hash_bytes(assignment_content).hexdigest() == ASSIGNMENT_CONTENT_SHA256
    assert len(assignment) == 99340 and assignment.encounter_id.is_unique
    assert assignment.groupby("patient_nbr").partition.nunique().max() == 1
    dev = assignment.loc[assignment.partition.eq("development")].copy()
    test = assignment.loc[assignment.partition.eq("test")].reset_index(drop=True)
    assert (len(dev), dev.patient_nbr.nunique(), int(dev.target.sum())) == (79398, 55989, 9017)
    assert (len(test), test.patient_nbr.nunique(), int(test.target.sum())) == (19942, 13998, 2297)
    assert set(dev.cv_validation_fold) == {0, 1, 2}
    assert dev.groupby("patient_nbr").cv_validation_fold.nunique().max() == 1
    baseline = pd.read_csv(BASELINE, float_precision="round_trip")
    pd.testing.assert_frame_equal(test[KEYS], baseline[KEYS])
    np.testing.assert_allclose(average_precision_score(test.target, baseline.discharge_hgb),
                               0.23737169718809753, rtol=0, atol=1e-13)
    test["paper_hgb"] = baseline.discharge_hgb
    return assignment, dev, test


def fit_catboost(features, categorical, dev, test, params, model_path):
    train_pool = Pool(features.loc[dev.encounter_id], dev.target, cat_features=categorical)
    test_pool = Pool(features.loc[test.encounter_id], cat_features=categorical)
    model = CatBoostClassifier(**params, loss_function="Logloss", random_seed=SEED,
                               thread_count=THREADS, allow_writing_files=False, verbose=False)
    model.fit(train_pool)
    model.save_model(str(model_path))
    probabilities = model.predict_proba(test_pool, thread_count=THREADS)[:, 1]
    reloaded = CatBoostClassifier().load_model(str(model_path))
    np.testing.assert_allclose(reloaded.predict_proba(test_pool, thread_count=THREADS)[:, 1],
                               probabilities, rtol=0, atol=1e-12)
    return probabilities


def fit_46_fields(assignment, dev, test, lock, models):
    cohort, _ = eligible_cohort(load_source(SOURCE))
    pd.testing.assert_frame_equal(cohort[KEYS], assignment[KEYS], check_dtype=False, check_exact=True)
    features, manifests = make_features(cohort)
    features.index = assignment.encounter_id
    categorical = manifests["discharge"]["categorical"]
    raw = features.copy()
    raw[DIAGNOSIS] = cohort[DIAGNOSIS].set_axis(features.index)
    params = {"iterations": lock["trees"], "depth": lock["depth"], "l2_leaf_reg": {4: 5., 6: 15.}[lock["depth"]],
              "learning_rate": .035, "task_type": "CPU"}
    test["hgb"] = test.pop("paper_hgb")
    for method, matrix in [("catboost", features), ("catboost_raw_icd9", raw)]:
        matrix[categorical] = matrix[categorical].fillna("__MISSING__").astype(str)
        test[method] = fit_catboost(matrix, categorical, dev, test, params, models / f"{method}.cbm")
    weight = lock["blend_catboost_weight"]
    test["blend"] = (1 - weight) * test.hgb + weight * test.catboost
    test["blend_raw_icd9"] = (1 - weight) * test.hgb + weight * test.catboost_raw_icd9
    methods = ["hgb", "catboost", "blend", "catboost_raw_icd9", "blend_raw_icd9"]
    for name, coefficients in lock["platt"].items():
        calibrated = name + "_platt"
        logits = logit(test[name].clip(1e-12, 1 - 1e-12))
        test[calibrated] = expit(coefficients["intercept"] + coefficients["slope"] * logits)
        methods.append(calibrated)
    return methods, [("catboost", "hgb"), ("blend", "hgb"), ("blend", "catboost"),
                     ("catboost_raw_icd9", "catboost")]


def fit_expanded(assignment, dev, test, lock, models, device):
    assert lock["final_training_seed"] == SEED
    hgb, cat = build_expanded(SOURCE, assignment)
    hgb_encoder = Encoding("native").fit(hgb.loc[dev.encounter_id])
    hgb_model = HistGradientBoostingClassifier(**lock["hgb"]["params"],
                                               max_iter=lock["hgb"]["trees"],
                                               random_state=SEED, warm_start=True)
    hgb_model.fit(hgb_encoder.transform(hgb.loc[dev.encounter_id]), dev.target)
    test["native_hgb"] = hgb_model.predict_proba(hgb_encoder.transform(hgb.loc[test.encounter_id]))[:, 1]
    joblib.dump({"model": hgb_model, "encoder": hgb_encoder}, models / "native_hgb.joblib", compress=3)
    cat_encoder = Encoding("catboost").fit(cat.loc[dev.encounter_id])
    cat_matrix = cat_encoder.transform(cat)
    params = {**lock["catboost"]["params"], "iterations": lock["catboost"]["trees"],
              "bootstrap_type": "Bernoulli", "subsample": .85, "random_strength": .5,
              "border_count": 128, "max_ctr_complexity": 1, "task_type": device}
    if device == "GPU":
        params["devices"] = "0"
    test["catboost"] = fit_catboost(cat_matrix, cat_encoder.cat, dev, test, params, models / "catboost.cbm")
    joblib.dump(cat_encoder, models / "catboost_encoder.joblib", compress=3)
    weights = lock["blend_weights"]
    test["blend_50_50"] = weights["native_hgb"] * test.native_hgb + weights["catboost"] * test.catboost
    methods = ["paper_hgb", "native_hgb", "catboost", "blend_50_50"]
    return methods, [("blend_50_50", name) for name in methods[:-1]]


def paired_ap(test, metrics, comparisons):
    names = sorted(set(name for pair in comparisons for name in pair))
    patients, inverse = np.unique(test.patient_nbr, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = np.empty((REPETITIONS, len(comparisons)))
    for index in range(REPETITIONS):
        weights = cluster_weights(inverse, len(patients), rng)
        ap = {name: average_precision_score(test.target, test[name], sample_weight=weights) for name in names}
        differences[index, :] = [ap[treatment] - ap[control] for treatment, control in comparisons]
    points = metrics.set_index("method").ap
    rows = []
    for index, (treatment, control) in enumerate(comparisons):
        lower, upper = np.quantile(differences[:, index], [.025, .975])
        rows.append({"treatment": treatment, "control": control,
                     "AP_difference": points[treatment] - points[control], "CI95_low": lower, "CI95_high": upper,
                     "patient_bootstrap_repetitions": REPETITIONS, "seed": BOOTSTRAP_SEED})
    return pd.DataFrame(rows)


def run(variant, output, device):
    if version("scikit-learn") != SKLEARN[variant] or version("catboost") != "1.2.8":
        raise RuntimeError(f"{variant} requires scikit-learn {SKLEARN[variant]} and CatBoost 1.2.8")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    lock_path = HERE / "reference" / variant / "selection_lock.json"
    lock = read_json(lock_path)
    assignment, dev, test = load_split()
    models = output / "models"
    models.mkdir(parents=True)
    print(f"{variant}: {len(dev):,} development, {len(test):,} test encounters", flush=True)
    if variant == "46_fields":
        methods, comparisons = fit_46_fields(assignment, dev, test, lock, models)
    else:
        methods, comparisons = fit_expanded(assignment, dev, test, lock, models, device)
    assert np.isfinite(test[methods]).all().all()
    predictions = output / "test_predictions.csv.gz"
    test.to_csv(predictions, index=False)
    metrics = pd.DataFrame([{"method": name, **probability_metrics(test.target, test[name])} for name in methods])
    metrics.to_csv(output / "test_metrics.csv", index=False)
    paired_ap(test, metrics, comparisons).to_csv(output / "paired_ap.csv", index=False)
    record = {"completed_utc": datetime.now(timezone.utc).isoformat(), "variant": variant, "catboost_device": device,
              "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost"]},
              "source_sha256": sha256(SOURCE), "assignments_sha256": sha256(ASSIGNMENTS),
              "reference_hgb_predictions_sha256": sha256(BASELINE), "selection_lock_sha256": sha256(lock_path),
              "test_predictions_sha256": sha256(predictions)}
    (output / "runtime.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(metrics[["method", "ap", "auroc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=SKLEARN)
    parser.add_argument("--output", type=Path, help="New directory for predictions, models, and scores")
    parser.add_argument("--device", choices=["CPU", "GPU"], help="CatBoost device (46_fields uses CPU)")
    args = parser.parse_args()
    if args.variant == "46_fields" and args.device == "GPU":
        parser.error("46_fields was run on CPU; use CPU for this comparison")
    device = args.device or ("CPU" if args.variant == "46_fields" else "GPU")
    with threadpool_limits(limits=THREADS):
        run(args.variant, args.output or HERE / "runs" / args.variant, device)
