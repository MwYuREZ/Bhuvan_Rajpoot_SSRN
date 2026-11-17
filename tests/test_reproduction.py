"""Checks for leakage, semantic missingness, and clustered inference correctness."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, fbeta_score, roc_auc_score

from readmission.data import (ADMISSION, NEVER_PREDICTORS, diagnosis_family,
    eligible_cohort, load_source, make_features, make_splits, verify_source)
from readmission.metrics import (choose_threshold, cluster_weights, operating_metrics,
                                 paired_cluster_bootstrap)
from readmission.models import build_model, select_candidate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs" / "paper.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source():
    path = ROOT / "data" / "raw" / "diabetic_data.csv"
    if not path.exists():
        pytest.skip("Run python scripts/download_data.py for source-data integration checks")
    return load_source(path)


def test_source_checksum_fails_closed(tmp_path):
    path = tmp_path / "modified.csv"
    path.write_text("modified source", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_source(path)


def test_real_cohort_and_patient_boundaries(source):
    cohort, audit = eligible_cohort(source)
    assert (len(cohort), cohort.patient_nbr.nunique(), cohort.target.sum()) == (99340, 69987, 11314)
    assert (~audit.included).sum() == 2426
    dev, test, folds = make_splits(cohort, CONFIG)
    assert (len(dev), len(test)) == (79398, 19942)
    assert set(cohort.iloc[dev].patient_nbr).isdisjoint(cohort.iloc[test].patient_nbr)
    assert cohort.iloc[test].target.sum() == 2297
    assert np.array_equal(np.sort(np.concatenate([v for _, v in folds])), np.arange(len(dev)))
    for train, val in folds:
        assert set(cohort.iloc[dev[train]].patient_nbr).isdisjoint(cohort.iloc[dev[val]].patient_nbr)


def test_semantic_categories_and_feature_exclusions(source):
    assert "None" in source.A1Cresult.unique()
    assert "None" in source.max_glu_serum.unique()
    assert source.race.isna().any()
    cohort, _ = eligible_cohort(source)
    x, manifest = make_features(cohort)
    assert len(manifest["admission"]["columns"]) == 10
    assert len(manifest["discharge"]["columns"]) == 46
    assert not NEVER_PREDICTORS.intersection(x.columns)
    assert list(manifest["admission"]["columns"]) == ADMISSION
    assert "admission_type_id" in manifest["admission"]["categorical"]
    assert "discharge_disposition_id" in manifest["discharge"]["categorical"]
    assert set(x.age.dropna().unique()) <= set(range(5, 100, 10))
    assert diagnosis_family("250.13") == "endocrine_metabolic"
    assert diagnosis_family("414") == "circulatory"
    assert diagnosis_family("V45") == "supplemental_V"
    assert diagnosis_family("E999") == "external_E"
    assert pd.isna(diagnosis_family("?"))


@pytest.mark.parametrize("name", ["logistic", "hgb"])
def test_fold_local_preprocessing_handles_unseen_category(name):
    train = pd.DataFrame({"age": [10., 20., np.nan, 40., 50., 60.],
                          "context": ["a", "b", np.nan, "a", "b", "a"]})
    validation = pd.DataFrame({"age": [10000., np.nan], "context": ["VALIDATION_ONLY", np.nan]})
    manifest = {"numeric": ["age"], "categorical": ["context"]}
    model = build_model(name, manifest, CONFIG)
    model.fit(train, [0, 1, 0, 1, 0, 1])
    preprocess = model.named_steps["preprocess"]
    encoder = preprocess.named_transformers_["categorical"].named_steps["encode"]
    assert "VALIDATION_ONLY" not in encoder.categories_[0]
    assert preprocess.named_transformers_["numeric"].named_steps["impute"].statistics_[0] == 40.
    predictions = model.predict_proba(validation)[:, 1]
    assert np.all(np.isfinite(predictions))
    assert "VALIDATION_ONLY" not in encoder.categories_[0]


def test_selection_uses_development_ap_then_brier():
    metrics = [{"model": "logistic", "ap": .20, "brier": .10, "test_ap": .99},
               {"model": "hgb", "ap": .21, "brier": .11, "test_ap": .01}]
    assert select_candidate(metrics) == "hgb"
    metrics[0]["ap"] = .21
    assert select_candidate(metrics) == "logistic"


@pytest.mark.parametrize("beta", [1, 2])
def test_threshold_matches_exhaustive_development_optimum(beta):
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    p = np.array([.1, .1, .3, .3, .8, .9, .9, .95])
    threshold = choose_threshold(y, p, beta)
    candidates = np.unique(p)
    scores = [fbeta_score(y, p >= t, beta=beta) for t in candidates]
    assert threshold == candidates[np.argmax(scores)]
    assert operating_metrics(y, p, threshold)[f"f{beta}"] == pytest.approx(max(scores))


def test_paired_cluster_bootstrap_matches_explicit_patient_duplication():
    # Each cluster contains one positive and one negative, ensuring valid draws.
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([10, 10, 20, 20, 30, 30])
    predictions = {"admission": np.array([.1, .2, .2, .3, .3, .4]),
                   "discharge": np.array([.05, .6, .4, .5, .2, .7])}
    samples, summary = paired_cluster_bootstrap(y, predictions, groups, 3, 71)
    rng = np.random.default_rng(71)
    inverse = np.repeat(np.arange(3), 2)
    weights = cluster_weights(inverse, 3, rng)
    assert weights[0] == weights[1] and weights[2] == weights[3] and weights[4] == weights[5]
    duplicated = np.repeat(np.arange(6), weights)
    first = samples[samples.replicate.eq(0)].set_index("features")
    for name, p in predictions.items():
        assert first.loc[name, "auroc"] == pytest.approx(roc_auc_score(y[duplicated], p[duplicated]))
        assert first.loc[name, "ap"] == pytest.approx(average_precision_score(y[duplicated], p[duplicated]))
        assert first.loc[name, "brier"] == pytest.approx(brier_score_loss(y[duplicated], p[duplicated]))
    assert first.loc["difference", "ap"] == pytest.approx(first.loc["discharge", "ap"] - first.loc["admission", "ap"])
    assert len(summary) == 12
