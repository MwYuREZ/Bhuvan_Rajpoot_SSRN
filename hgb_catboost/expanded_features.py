"""UCI-only features for the 85/88-field experiment."""

import re

import numpy as np
import pandas as pd

from readmission.data import EXCLUDED_DISPOSITIONS, diagnosis_family


# Adapted from Roy4016/-diabetes-project, commit 0625563bf2362865346cb6df0f15183709189254.
# The accompanying TEAM_LICENSE preserves the upstream MIT notice.
DIAGNOSIS = ["diag_1", "diag_2", "diag_3"]
ADMIN = {"admission_type_id", "admission_source_id", "discharge_disposition_id"}
DIAG_GROUPS = {
    "clinical_hypertension": (401, 405),
    "clinical_lipid_disorder": (272, 272),
    "clinical_ischemic_heart": (410, 414),
    "clinical_heart_failure": (428, 428),
    "clinical_cerebrovascular": (430, 438),
    "clinical_renal_disorder": (580, 589),
    "clinical_copd_related": (490, 496),
    "clinical_liver_disorder": (570, 573),
    "clinical_neoplasm": (140, 208),
}
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]


def _age_midpoint(value):
    interval = re.match(r"^\[(\d+)-(\d+)\)$", str(value).strip())
    return sum(map(int, interval.groups())) / 2 if interval else np.nan


def build_expanded(source, assignments):
    """Keep the historical team reader, then select exactly the paper's encounters."""
    frame = pd.read_csv(source, low_memory=False).replace("?", np.nan)
    frame.columns = frame.columns.str.strip()
    frame = frame.dropna(subset=["patient_nbr", "readmitted"])
    frame = frame.loc[frame.readmitted.isin(["<30", ">30", "NO"])]
    dispositions = pd.to_numeric(frame.discharge_disposition_id, errors="coerce")
    frame = frame.loc[~dispositions.isin(EXCLUDED_DISPOSITIONS)].copy()
    frame["target"] = frame.readmitted.eq("<30").astype(int)
    frame = frame.set_index("encounter_id").loc[assignments.encounter_id]
    keys = ["patient_nbr", "target"]
    pd.testing.assert_frame_equal(frame[keys], assignments.set_index("encounter_id")[keys],
                                  check_dtype=False, check_exact=True)
    raw = [column for column in frame if column not in {"patient_nbr", "readmitted", "target", "weight"}]
    assert len(raw) == 46

    diagnoses = frame[DIAGNOSIS].astype("string").apply(lambda column: column.str.strip())
    prefixes = diagnoses.apply(lambda column: pd.to_numeric(
        column.str.extract(r"^(\d{3})(?:\.|$)", expand=False), errors="coerce"))
    clinical = pd.DataFrame(index=frame.index)
    for name, (low, high) in DIAG_GROUPS.items():
        clinical[name] = ((prefixes >= low) & (prefixes <= high)).fillna(False).any(axis=1).astype("int8")
    complications = diagnoses.apply(lambda column: column.str.match(r"^250\.[1-9][0-9]?$", na=False))
    clinical["clinical_diabetes_complication"] = complications.any(axis=1).astype("int8")
    clinical["clinical_comorbidity_count"] = clinical.sum(axis=1).astype("float32")
    clinical["clinical_unique_diagnoses"] = (frame[DIAGNOSIS].astype("string")
        .replace({"?": pd.NA, "": pd.NA}).nunique(axis=1).astype("float32"))

    medications = frame[MEDICATIONS].fillna("No").astype(str)
    for name, values in {
        "active_diabetes_drugs": ["Steady", "Up", "Down"],
        "drugs_increased": ["Up"], "drugs_decreased": ["Down"], "drugs_adjusted": ["Up", "Down"],
    }.items():
        clinical["clinical_" + name] = medications.isin(values).sum(axis=1).astype("float32")

    visits = ["number_inpatient", "number_emergency", "number_outpatient"]
    utilization = frame[visits].apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    for column in visits:
        clinical["clinical_log_" + column] = np.log1p(utilization[column]).astype("float32")
    clinical["clinical_total_prior_visits"] = utilization.sum(axis=1).astype("float32")
    clinical["clinical_prior_inpatient_flag"] = utilization.number_inpatient.gt(0).astype("int8")
    clinical["clinical_repeated_inpatient_flag"] = utilization.number_inpatient.ge(2).astype("int8")
    clinical["clinical_prior_emergency_flag"] = utilization.number_emergency.gt(0).astype("int8")
    clinical["clinical_comorbidity_x_prior_visits"] = (clinical.clinical_comorbidity_count *
        np.log1p(clinical.clinical_total_prior_visits)).astype("float32")
    assert len(clinical.columns) == 24

    cat = pd.concat([frame[raw], clinical], axis=1)
    chapters = ["chapter_" + column for column in DIAGNOSIS]
    for column, chapter in zip(DIAGNOSIS, chapters):
        cat[chapter] = frame[column].map(diagnosis_family)
    cat["age_midpoint"] = frame.age.map(_age_midpoint)
    stay = frame.time_in_hospital.clip(lower=1).astype(float)
    for column in ["num_lab_procedures", "num_medications", "num_procedures", "number_diagnoses"]:
        cat[column + "_per_day"] = frame[column] / stay
    visits_total = frame[["number_inpatient", "number_outpatient", "number_emergency"]].sum(axis=1)
    cat["inpatient_share"] = frame.number_inpatient / (1 + visits_total)
    cat["emergency_share"] = frame.number_emergency / (1 + visits_total)
    cat["prior_inpatient_x_stay"] = np.log1p(frame.number_inpatient) * stay
    cat["prior_emergency_x_age"] = np.log1p(frame.number_emergency) * cat.age_midpoint
    cat["a1c_measured"] = frame.A1Cresult.notna().astype(float)
    cat["glucose_measured"] = frame.max_glu_serum.notna().astype(float)
    cat["a1c_high"] = frame.A1Cresult.isin([">7", ">8"]).astype(float)
    cat["glucose_high"] = frame.max_glu_serum.isin([">200", ">300"]).astype(float)
    cat["insulin_adjusted"] = frame.insulin.isin(["Up", "Down"]).astype(float)
    cat["missing_field_count"] = frame[raw].isna().sum(axis=1).astype(float)

    hgb = cat.drop(columns=chapters)
    hgb[DIAGNOSIS] = cat[chapters].set_axis(DIAGNOSIS, axis=1)
    assert (hgb.shape[1], cat.shape[1]) == (85, 88)
    return hgb, cat


class Encoding:
    """Fit medians and native HGB category levels on development patients."""

    def __init__(self, kind):
        self.kind = kind

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.cat = [column for column in frame if column in ADMIN or
                    pd.api.types.is_object_dtype(frame[column]) or pd.api.types.is_string_dtype(frame[column])]
        self.num = [column for column in frame if column not in self.cat]
        self.medians = frame[self.num].median().fillna(0)
        self.keep = {}
        self.levels = {}
        if self.kind == "native":
            for column in self.cat:
                values = self._category(frame[column])
                counts = values.value_counts()
                self.keep[column] = set(counts[counts.ge(10)].index)
                self.levels[column] = sorted(values.where(values.isin(self.keep[column]), "__RARE__").unique())
        return self

    @staticmethod
    def _category(values):
        return values.astype(object).where(values.notna(), "__MISSING__").astype(str)

    def transform(self, frame):
        result = frame[self.columns].copy()
        for column in self.cat:
            values = self._category(frame[column])
            if self.kind == "native":
                values = values.where(values.isin(self.keep[column]), "__RARE__")
                result[column] = pd.Categorical(values, categories=self.levels[column])
            else:
                result[column] = values
        numeric = frame[self.num].apply(pd.to_numeric, errors="coerce").fillna(self.medians).astype(np.float32)
        for column in self.num:
            result[column] = numeric[column]
        return result
