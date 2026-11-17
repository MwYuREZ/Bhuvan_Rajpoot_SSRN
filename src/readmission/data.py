"""Source integrity, deterministic semantic recoding, and patient-disjoint splits."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

SOURCE_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
SOURCE_SHA256 = "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97"
EXCLUDED_DISPOSITIONS = [11, 13, 14, 19, 20, 21]
ADMISSION = ["race", "gender", "age", "admission_type_id", "admission_source_id",
             "payer_code", "medical_specialty", "number_outpatient",
             "number_emergency", "number_inpatient"]
NUMERIC = ["age", "time_in_hospital", "num_lab_procedures", "num_procedures",
           "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "number_diagnoses"]
NEVER_PREDICTORS = {"encounter_id", "patient_nbr", "weight", "readmitted"}
ICD9_CHAPTERS = [
    (1, 139, "infectious_parasitic"), (140, 239, "neoplasms"),
    (240, 279, "endocrine_metabolic"), (280, 289, "blood"),
    (290, 319, "mental"), (320, 389, "nervous_sense"),
    (390, 459, "circulatory"), (460, 519, "respiratory"),
    (520, 579, "digestive"), (580, 629, "genitourinary"),
    (630, 679, "pregnancy_childbirth"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"),
    (760, 779, "perinatal"), (780, 799, "symptoms_ill_defined"),
    (800, 999, "injury_poisoning"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(path: Path) -> str:
    actual = sha256(path)
    if actual != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual}")
    return actual


def download_data(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    csv = directory / "diabetic_data.csv"
    if csv.exists():
        verify_source(csv)
        return csv
    archive = directory / "diabetes.zip"
    with urllib.request.urlopen(SOURCE_URL, timeout=90) as response:
        archive.write_bytes(response.read())
    # Extract only the two expected flat files, never archive-provided paths.
    with zipfile.ZipFile(archive) as z:
        for name in ["diabetic_data.csv", "IDS_mapping.csv"]:
            matches = [m for m in z.namelist() if Path(m).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in source archive")
            (directory / name).write_bytes(z.read(matches[0]))
    verify_source(csv)
    (directory / "provenance.json").write_text(json.dumps({
        "url": SOURCE_URL, "csv_sha256": sha256(csv),
        "zip_sha256": sha256(archive), "license": "CC BY 4.0",
        "citation": "Clore et al. (2014), UCI, https://doi.org/10.24432/C5230J"
    }, indent=2) + "\n", encoding="utf-8")
    return csv


def load_source(path: Path) -> pd.DataFrame:
    verify_source(path)
    # The literal 'None' denotes no glucose/HbA1c measurement; it is a category.
    return pd.read_csv(path, keep_default_na=False, na_values=["?"], low_memory=False)


def eligible_cohort(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid_gender = source["gender"].eq("Unknown/Invalid")
    excluded_discharge = source["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)
    keep = ~invalid_gender & ~excluded_discharge
    audit = source[["encounter_id", "patient_nbr"]].copy()
    audit["invalid_gender"] = invalid_gender
    audit["death_hospice_discharge"] = excluded_discharge
    audit["included"] = keep
    cohort = source.loc[keep].copy().reset_index(names="source_row")
    valid_labels = {"<30", ">30", "NO"}
    if not set(cohort["readmitted"]).issubset(valid_labels):
        raise ValueError("Unexpected outcome label")
    cohort["target"] = cohort["readmitted"].eq("<30").astype(int)
    return cohort, audit


def diagnosis_family(value):
    """ICD-9 chapter families, including separate supplemental V and external E codes."""
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return np.nan
    text = str(value).strip().upper()
    if text.startswith("V"):
        return "supplemental_V"
    if text.startswith("E"):
        return "external_E"
    try:
        code = int(float(text))
    except ValueError:
        return "other"
    for lo, hi, family in ICD9_CHAPTERS:
        if lo <= code <= hi:
            return family
    return "other"


def age_midpoint(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.fullmatch(r"\[(\d+)-(\d+)\)", str(value))
    if not match:
        raise ValueError(f"Unexpected age interval {value!r}")
    return (float(match[1]) + float(match[2])) / 2


def make_features(cohort: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [c for c in cohort if c not in NEVER_PREDICTORS | {"source_row", "target"}]
    if len(columns) != 46:
        raise ValueError(f"Expected 46 discharge predictors, got {len(columns)}")
    x = cohort[columns].copy()
    x["age"] = x["age"].map(age_midpoint)
    for col in ["diag_1", "diag_2", "diag_3"]:
        x[col] = x[col].map(diagnosis_family)
    # Administrative IDs are categories despite their integer representation.
    for col in x:
        if col in NUMERIC:
            x[col] = pd.to_numeric(x[col], errors="raise")
        else:
            x[col] = x[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
    sets = {"admission": ADMISSION, "discharge": columns}
    manifest = {name: {"columns": cols, "numeric": [c for c in cols if c in NUMERIC],
                       "categorical": [c for c in cols if c not in NUMERIC]}
                for name, cols in sets.items()}
    return x, manifest


def assert_disjoint(groups, train, test):
    if set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]):
        raise AssertionError("Patient leakage across an assessment boundary")


def make_splits(cohort: pd.DataFrame, config: dict):
    groups = cohort["patient_nbr"].to_numpy()
    y = cohort["target"].to_numpy()
    dev, test = next(GroupShuffleSplit(n_splits=1, test_size=config["test_size"],
                    random_state=config["split_seed"]).split(cohort, y, groups))
    assert_disjoint(groups, dev, test)
    splitter = StratifiedGroupKFold(n_splits=config["cv_folds"], shuffle=True,
                                   random_state=config["cv_seed"])
    folds = list(splitter.split(cohort.iloc[dev], y[dev], groups[dev]))
    seen = np.zeros(len(dev), dtype=int)
    for train, val in folds:
        assert_disjoint(groups[dev], train, val)
        seen[val] += 1
    if not np.all(seen == 1):
        raise AssertionError("Every development encounter must have exactly one OOF prediction")
    return dev, test, folds
