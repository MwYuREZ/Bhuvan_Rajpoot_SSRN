"""Exactly two prespecified candidates with fold-local preprocessing."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def build_model(name: str, manifest: dict, config: dict) -> Pipeline:
    numeric = [("impute", SimpleImputer(strategy="median"))]
    categorical = [("impute", SimpleImputer(strategy=config["categorical_imputation"]))]
    if name == "logistic":
        numeric.append(("scale", StandardScaler()))
        categorical.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)))
        estimator = LogisticRegression(penalty="l2", **config["logistic"])
    elif name == "hgb":
        categorical.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=config["ordinal_unknown_value"])))
        # The paper specifies ordinal encoding, but does not specify native
        # categorical splits. Explicitly keep numeric splits on encoded values.
        estimator = HistGradientBoostingClassifier(categorical_features=None,
                    random_state=config["model_seed"], **config["hgb"])
    else:
        raise ValueError(name)
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(numeric), manifest["numeric"]),
        ("categorical", Pipeline(categorical), manifest["categorical"]),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def select_candidate(dev_metrics: list[dict]) -> str:
    """Selection accepts development metrics only; test results cannot enter it."""
    return sorted(dev_metrics, key=lambda r: (-r["ap"], r["brier"], r["model"]))[0]["model"]
