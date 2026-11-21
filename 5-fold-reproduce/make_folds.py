"""Generate assignments for stratified, patient-disjoint five-fold cross-validation."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn import __version__ as sklearn_version

from run import HERE, KEYS, ROOT, load_source
from upstream.data import EXCLUDED_DISCHARGE_IDS
from upstream.splitting import make_patient_folds

SEED = 777
N_FOLDS = 5
SPLIT_VERSION = "1.8.0"
FOLD_FILE = HERE / "results" / "five_fold" / "fold_assignments.csv.gz"
PUBLISHED_TEST_ROWS = [19869, 19869, 19868, 19869, 19868]
INNER = [f"inner_val_{fold}" for fold in range(1, N_FOLDS + 1)]


def published_cohort(source):
    # Retain encounters with Unknown/Invalid gender in the five-fold cohort.
    rows = source.dropna(subset=["patient_nbr", "readmitted"])
    keep = rows.readmitted.isin(["<30", ">30", "NO"]) & ~rows.discharge_disposition_id.isin(EXCLUDED_DISCHARGE_IDS)
    cohort = rows.loc[keep].copy().reset_index(names="source_row")
    cohort["target"] = cohort.readmitted.eq("<30").astype(int)
    expected = (99343, 69990, 11314)
    if (len(cohort), cohort.patient_nbr.nunique(), cohort.target.sum()) != expected:
        raise ValueError(f"The five-fold cohort differs from the expected counts {expected}")
    return cohort


def create_folds(source, output):
    if output.exists():
        raise FileExistsError(f"Choose a new output file: {output}")
    if sklearn_version != SPLIT_VERSION:
        raise RuntimeError(f"Use scikit-learn {SPLIT_VERSION} to regenerate the five-fold split")
    cohort = published_cohort(load_source(source))
    folds = make_patient_folds(cohort, n_splits=N_FOLDS, seed=SEED)
    if [len(fold.test_idx) for fold in folds] != PUBLISHED_TEST_ROWS:
        raise ValueError("Outer fold sizes differ from the expected five-fold test counts")
    assignment = cohort[KEYS].rename(columns={"target": "y_true"}).copy()
    assignment["Fold"] = 0
    for fold, column in zip(folds, INNER):
        assignment.loc[fold.test_idx, "Fold"] = fold.number
        assignment[column] = 0
        assignment.loc[fold.val_idx, column] = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(output, index=False, mode="x")
    print(assignment.groupby("Fold").agg(encounters=("encounter_id", "size"),
                                         patients=("patient_nbr", "nunique"),
                                         positives=("y_true", "sum")).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "diabetic_data.csv")
    parser.add_argument("--output", type=Path, default=FOLD_FILE)
    args = parser.parse_args()
    create_folds(args.input, args.output)
