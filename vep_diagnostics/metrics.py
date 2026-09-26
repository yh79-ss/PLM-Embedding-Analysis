"""Assay-macro metrics and paired conditional cluster uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import rankdata, spearmanr


def rho(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(y) < 3 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        return float("nan")
    if np.ptp(y) == 0 or np.ptp(prediction) == 0:
        return float("nan")
    # Index zero is shared by SciPy 1.9's SpearmanrResult and newer result types.
    return float(spearmanr(y, prediction)[0])


def permutation_pvalue(null, observed):
    """Plus-one upper tail, including numerical ties between correlation routes.

    Rank-dot and SciPy correlation, or vector and batched BLAS products, can
    round a mathematically identical statistic differently. The absolute floor
    is 100 float64 eps for these bounded [-1, 1] statistics, not a scientific
    effect-size tolerance. Undefined hypotheses stay undefined, not p=0.
    """
    null = np.asarray(null, dtype=float)
    if null.ndim != 1:
        raise ValueError("Permutation null must be one-dimensional")
    if not len(null) or not np.isfinite(observed) or not np.isfinite(null).all():
        return float("nan")
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(float(observed)))
    return float((1 + np.count_nonzero(null >= observed - tolerance)) / (len(null) + 1))


def ranks(y):
    y = np.asarray(y, dtype=float)
    return (rankdata(y, method="average") - 1) / (len(y) - 1) if len(y) > 1 else np.full(len(y), 0.5)


def assay_metrics(frame, prediction_columns):
    rows = []
    for assay, group in frame.groupby("assay_id", sort=True):
        for col in prediction_columns:
            value = rho(group.score.to_numpy(dtype=float), group[col].to_numpy(dtype=float))
            rows.append({"assay_id": assay, "task": group.task.iloc[0],
                         "protein_id": group.protein_id.iloc[0], "super_cluster": group.super_cluster.iloc[0],
                         "fold": group.fold.iloc[0], "model": col, "n_variants": len(group), "rho": value,
                         "status": "ok" if np.isfinite(value) else "undefined_constant_or_too_small"})
    return pd.DataFrame(rows)


def macro_summary(metrics):
    return metrics.groupby(["task", "model"], as_index=False).agg(
        n_assays=("assay_id", "size"), n_valid=("rho", "count"), mean_rho=("rho", "mean"))


def paired_cluster_interval(frame, n_bootstrap, seed):
    """Resample clusters with multiplicity; preserve assay-macro point estimand."""
    if frame.empty:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_clusters": 0}
    grouped = frame.groupby("super_cluster").delta.agg(["sum", "count"])
    estimate = float(frame.delta.mean())
    if len(grouped) < 2:
        return {"delta": estimate, "ci_low": np.nan, "ci_high": np.nan, "n_clusters": len(grouped)}
    totals, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_bootstrap):
        chosen = rng.integers(len(grouped), size=len(grouped))
        draws.append(totals[chosen].sum() / counts[chosen].sum())
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"delta": estimate, "ci_low": float(low), "ci_high": float(high), "n_clusters": len(grouped)}


def mmd_squared(x, y):
    """Biased multi-scale RBF MMD², including diagonals; pair-adaptive bandwidth."""
    distances = pdist(np.vstack([x, y]), metric="sqeuclidean")
    positive = distances[distances > 0]
    base = float(np.median(positive)) if len(positive) else 1.0
    xx, yy = squareform(pdist(x, "sqeuclidean")), squareform(pdist(y, "sqeuclidean"))
    xy = cdist(x, y, "sqeuclidean")
    values = []
    for factor in (0.5, 1.0, 2.0, 4.0):
        denominator = 2 * max(base * factor, 1e-12)
        values.append(np.exp(-xx / denominator).mean() + np.exp(-yy / denominator).mean()
                      - 2 * np.exp(-xy / denominator).mean())
    return float(max(0.0, np.mean(values)))


def moment_distances(x, y):
    """Dimension-normalized mean and covariance distances in declared coordinates."""
    dimension = x.shape[1]
    centroid = float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)) / np.sqrt(dimension))
    if min(len(x), len(y)) < 2:
        return centroid, float("nan")
    a, b = x - x.mean(axis=0), y - y.mean(axis=0)
    if dimension <= len(x) + len(y):
        value = np.linalg.norm(a.T @ a / (len(a) - 1) - b.T @ b / (len(b) - 1), "fro")
    else:
        # Gram identity avoids allocating two D-by-D matrices for high-dimensional inputs.
        aa = np.sum((a @ a.T) ** 2) / (len(a) - 1) ** 2
        bb = np.sum((b @ b.T) ** 2) / (len(b) - 1) ** 2
        cross = np.sum((a @ b.T) ** 2) / ((len(a) - 1) * (len(b) - 1))
        squared = aa + bb - 2 * cross
        tolerance = 16 * np.finfo(float).eps * max(aa + bb, 2 * abs(cross))
        value = np.sqrt(squared) if squared > tolerance else 0.0
    return centroid, float(value / np.sqrt(dimension))


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, not its square root."""
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    midpoint = (p + q) / 2
    return float(sum(np.sum(a[a > 0] * np.log2(a[a > 0] / midpoint[a > 0])) for a in (p, q)) / 2)
