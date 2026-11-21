"""Compare HGB preprocessing methods on a fixed, patient-disjoint holdout."""

import argparse
import json
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from readmission.data import eligible_cohort, load_source, make_features, sha256
from readmission.metrics import cluster_weights
from readmission.models import build_model
from upstream.data import feature_columns, load_uci_diabetes
from upstream.features import (
    add_clinical_features, infer_column_types, make_preprocessor, normalize_model_frame,
)

KEYS = ["encounter_id", "patient_nbr", "target"]
ADMIN_IDS = ["admission_type_id", "admission_source_id", "discharge_disposition_id"]
ROY_COMMIT = "0625563bf2362865346cb6df0f15183709189254"
BASELINE = "bhuvan_hgb"
MIN_FREQUENCY = 100


def load_assignments(path, cohort):
    assignments = pd.read_csv(path)
    pd.testing.assert_frame_equal(assignments[KEYS], cohort[KEYS], check_dtype=False)
    if not assignments.encounter_id.is_unique or assignments[KEYS].isna().any().any():
        raise ValueError("The saved split has duplicate encounters or missing identifiers/labels")
    if set(assignments.partition) != {"development", "test"}:
        raise ValueError("Expected development and test partitions")
    if assignments.groupby("patient_nbr").partition.nunique().max() != 1:
        raise ValueError("The saved split has patients in both partitions")
    dev = assignments.index[assignments.partition.eq("development")]
    test = assignments.index[assignments.partition.eq("test")]
    if any(set(assignments.loc[rows, "target"]) != {0, 1} for rows in (dev, test)):
        raise ValueError("Both partitions must contain both target classes")
    return assignments, dev, test


def roy_hgb_preprocessor(train, columns):
    """Build a one-hot encoder with infrequent category grouping and median imputation."""
    categorical, numeric = infer_column_types(train, columns)
    return ColumnTransformer([
        ("categorical", make_pipeline(
            SimpleImputer(strategy="constant", fill_value="Missing"),
            OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_FREQUENCY,
                          sparse_output=False, dtype=np.float32),
        ), categorical),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ], sparse_threshold=0)


def hgb_matrix(matrix):
    """Convert sparse inputs to dense arrays and cast dense inputs to float32."""
    if sparse.issparse(matrix):
        return matrix.toarray(order="F")
    return matrix.astype(np.float32, copy=False)


def fit_roy(preprocessor, model, train, test, labels):
    matrix = hgb_matrix(preprocessor.fit_transform(train))
    print(f"  Training matrix: {matrix.shape[0]:,} rows, {matrix.shape[1]:,} columns", flush=True)
    model.fit(matrix, labels)
    del matrix
    matrix = hgb_matrix(preprocessor.transform(test))
    return model.predict_proba(matrix)[:, 1], preprocessor.get_feature_names_out().tolist()


def evaluate(predictions, replicates, seed):
    labels = predictions.target.to_numpy()
    methods = [name for name in predictions if name not in KEYS]
    baseline = methods.index(BASELINE)
    scores = predictions[methods].to_numpy().T
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected finite probabilities between zero and one")
    ap = np.array([average_precision_score(labels, p) for p in scores])
    patients, inverse = np.unique(predictions.patient_nbr, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas = np.empty((replicates, len(methods)))
    for draw in range(replicates):
        weights = cluster_weights(inverse, len(patients), rng)
        draw_ap = np.array([average_precision_score(labels, p, sample_weight=weights) for p in scores])
        deltas[draw] = draw_ap - draw_ap[baseline]
    low, high = np.quantile(deltas, [.025, .975], axis=0)
    return pd.DataFrame({
        "method": methods,
        "test_ap": ap,
        "test_auroc": [roc_auc_score(labels, p) for p in scores],
        "ap_delta_vs_bhuvan": ap - ap[baseline],
        "ap_delta_ci95_low": low,
        "ap_delta_ci95_high": high,
    })


def run(source, output, threads):
    if output.exists():
        raise FileExistsError(f"Choose a new --output directory: {output}")
    config_path = ROOT / "configs" / "paper.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_path = ROOT / "results" / "full" / "splits" / "encounter_assignments.csv.gz"
    cohort, _ = eligible_cohort(load_source(source))
    assignments, dev, test = load_assignments(split_path, cohort)
    roy = load_uci_diabetes(source)
    pd.testing.assert_frame_equal(roy[KEYS], assignments[KEYS], check_dtype=False)

    base, manifest = make_features(cohort)
    discharge = manifest["discharge"]
    base_model = build_model("hgb", discharge, config)
    labels = assignments.loc[dev, "target"]
    predictions = assignments.loc[test, KEYS].copy()
    features = {BASELINE: {"input": discharge["columns"]}}

    print(f"Development: {len(dev):,} encounters. Test: {len(test):,} encounters.", flush=True)
    with threadpool_limits(limits=threads):
        print("Fitting Bhuvan discharge HGB", flush=True)
        base_model.fit(base.loc[dev], labels)
        predictions[BASELINE] = base_model.predict_proba(base.loc[test])[:, 1]
        features[BASELINE]["encoded"] = base_model["preprocess"].get_feature_names_out().tolist()
        del base, cohort

        print("Fitting HGB with raw ICD-9 codes and one-hot encoding", flush=True)
        columns = discharge["columns"]
        roy_hgb = roy[columns].astype({name: object for name in ADMIN_IDS})
        roy_hgb = normalize_model_frame(roy_hgb, columns)
        train, holdout = roy_hgb.loc[dev], roy_hgb.loc[test]
        model = build_model("hgb", discharge, config)["model"]
        predictions["roy_hgb_46"], encoded = fit_roy(
            roy_hgb_preprocessor(train, columns), model, train, holdout, labels,
        )
        features["roy_hgb_46"] = {"input": columns, "encoded": encoded, "min_frequency": MIN_FREQUENCY}
        del roy_hgb, train, holdout

        print("Fitting HGB with derived clinical features and one-hot encoding", flush=True)
        roy, clinical = add_clinical_features(roy)
        columns = feature_columns(roy)
        train = normalize_model_frame(roy.loc[dev], columns)
        holdout = normalize_model_frame(roy.loc[test], columns)
        model = build_model("hgb", discharge, config)["model"]
        predictions["roy_shared_clinical"], encoded = fit_roy(
            make_preprocessor(train, columns), model, train, holdout, labels,
        )
        features["roy_shared_clinical"] = {"input": columns, "encoded": encoded, "clinical": clinical}

    metrics = evaluate(predictions, config["bootstrap_replicates"], config["bootstrap_seed"])
    run_info = {
        "roy_commit": ROY_COMMIT,
        "source_sha256": sha256(source),
        "split_sha256": sha256(split_path),
        "config_sha256": sha256(config_path),
        "development_encounters": len(dev),
        "development_patients": int(assignments.loc[dev, "patient_nbr"].nunique()),
        "test_encounters": len(test),
        "test_patients": int(predictions.patient_nbr.nunique()),
        "test_positives": int(predictions.target.sum()),
        "hgb_parameters": base_model["model"].get_params(),
        "bootstrap_replicates": config["bootstrap_replicates"],
        "bootstrap_seed": config["bootstrap_seed"],
        "threads": threads,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")},
        "features": features,
    }
    output.mkdir(parents=True)
    predictions.to_csv(output / "test_predictions.csv.gz", index=False)
    metrics.to_csv(output / "test_metrics.csv", index=False)
    (output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "diabetic_data.csv")
    default_output = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args.input, args.output, args.threads)
