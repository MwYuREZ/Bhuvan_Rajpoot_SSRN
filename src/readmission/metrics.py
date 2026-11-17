"""Probability metrics, development thresholds, and paired cluster uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (average_precision_score, brier_score_loss,
    confusion_matrix, log_loss, precision_recall_curve, roc_auc_score)


def calibration_parameters(y, p):
    z = logit(np.clip(p, 1e-12, 1 - 1e-12))
    design = np.column_stack([np.ones(len(z)), z])
    y = np.asarray(y)

    def objective(beta):
        linear = design @ beta
        loss = np.mean(np.logaddexp(0, linear) - y * linear)
        gradient = design.T @ (expit(linear) - y) / len(y)
        return loss, gradient

    fit = minimize(objective, np.array([0., 1.]), jac=True, method="BFGS",
                   options={"gtol": 1e-8, "maxiter": 1000})
    # BFGS may report precision loss after reaching a sufficiently small gradient.
    if not fit.success and np.max(np.abs(fit.jac)) > 1e-6:
        raise RuntimeError(f"Calibration regression failed: {fit.message}")
    return float(fit.x[0]), float(fit.x[1])


def probability_metrics(y, p, calibration=True):
    result = {"auroc": roc_auc_score(y, p), "ap": average_precision_score(y, p),
              "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p, labels=[0, 1])}
    if calibration:
        result["calibration_intercept"], result["calibration_slope"] = calibration_parameters(y, p)
    return result


def choose_threshold(y, p, beta):
    precision, recall, thresholds = precision_recall_curve(y, p)
    numerator = (1 + beta * beta) * precision[:-1] * recall[:-1]
    denominator = beta * beta * precision[:-1] + recall[:-1]
    f = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    # sklearn thresholds ascend, so argmax takes the lowest threshold on a tie.
    return float(thresholds[np.argmax(f)])


def operating_metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def divide(a, b):
        return float(a / b) if b else 0.0

    return {"threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp), "sensitivity": divide(tp, tp + fn),
            "specificity": divide(tn, tn + fp), "ppv": divide(tp, tp + fp),
            "npv": divide(tn, tn + fn), "f1": divide(2 * tp, 2 * tp + fp + fn),
            "f2": divide(5 * tp, 5 * tp + fp + 4 * fn),
            "alert_rate": float(pred.mean())}


def cluster_weights(inverse, n_groups, rng):
    """Multiplicity weights exactly represent sampling all patients with replacement."""
    draws = rng.integers(0, n_groups, size=n_groups)
    counts = np.bincount(draws, minlength=n_groups)
    return counts[np.asarray(inverse)]


def paired_cluster_bootstrap(y, predictions: dict, groups, replicates, seed):
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = inverse.max() + 1
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        weights = cluster_weights(inverse, n_groups, rng)
        positive = np.sum(weights * y)
        if positive == 0 or positive == weights.sum():
            raise ValueError("Single-class cluster bootstrap sample; cannot compute AUROC")
        metrics = {}
        for name, p in predictions.items():
            metrics[name] = {
                "auroc": roc_auc_score(y, p, sample_weight=weights),
                "ap": average_precision_score(y, p, sample_weight=weights),
                "brier": np.average((y - p) ** 2, weights=weights),
                "log_loss": log_loss(y, p, sample_weight=weights, labels=[0, 1]),
            }
            rows.append({"replicate": replicate, "features": name, **metrics[name]})
        rows.append({"replicate": replicate, "features": "difference",
                     **{k: metrics["discharge"][k] - metrics["admission"][k]
                        for k in metrics["admission"]}})
    samples = pd.DataFrame(rows)
    summary = []
    point = {k: probability_metrics(y, p, calibration=False) for k, p in predictions.items()}
    point["difference"] = {k: point["discharge"][k] - point["admission"][k]
                           for k in point["admission"]}
    for name, frame in samples.groupby("features", sort=False):
        for metric in ["auroc", "ap", "brier", "log_loss"]:
            lo, hi = np.quantile(frame[metric], [0.025, 0.975])
            summary.append({"features": name, "metric": metric,
                            "estimate": point[name][metric], "lower": lo, "upper": hi,
                            "replicates": replicates, "seed": seed})
    return samples, pd.DataFrame(summary)


def calibration_deciles(y, p):
    # qcut keeps tied scores in the same bin, dropping duplicate edges if needed.
    frame = pd.DataFrame({"observed": np.asarray(y), "predicted": np.asarray(p)})
    frame["bin"] = pd.qcut(frame["predicted"], q=10, duplicates="drop")
    return frame.groupby("bin", observed=True).agg(
        encounters=("observed", "size"), mean_predicted=("predicted", "mean"),
        event_rate=("observed", "mean")).reset_index()
