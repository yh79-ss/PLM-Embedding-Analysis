"""Strict seed-paired contrasts and conditional assay-macro cluster intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IDENTITY, KEY, Run, load_metadata, read_csv, require_columns, scores, stable_seed
from .metrics import paired_cluster_interval, rho


def declared_seeds(seeds):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Declare a nonempty set of unique seeds")
    return set(int(seed) for seed in seeds)


def checked_predictions(path, metadata, columns, seeds, protocol, require_seed=False):
    """Validate identities before joining; metadata remains the score authority."""
    expected = declared_seeds(seeds)
    forbidden = set(KEY + IDENTITY + ["score", "seed", "n_evaluation", "analysis_scope", "label_protocol"])
    if set(columns) & forbidden or len(columns) != len(set(columns)):
        raise ValueError("Prediction columns must be distinct from reserved identity, score, count and provenance fields")
    prediction = read_csv(path)
    require_columns(prediction, KEY + columns + (["seed"] if require_seed else []))
    if "seed" in prediction:
        numeric = pd.to_numeric(prediction.seed, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
            raise ValueError("Prediction seeds must be finite integers")
        prediction["seed"] = numeric.astype(int)
        if not set(prediction.seed).issubset(expected):
            raise ValueError("Prediction file contains undeclared seeds")
    join_key = KEY + (["seed"] if "seed" in prediction else [])
    if prediction.duplicated(join_key).any():
        raise ValueError(f"Duplicate prediction keys: {join_key}")
    if protocol == "no-target-label" and "label_protocol" in prediction:
        if not prediction.label_protocol.str.startswith("no-target-label").all():
            raise ValueError("Cannot relabel target-support/descriptive predictions as no-target-label")
    metadata_index = pd.MultiIndex.from_frame(metadata[KEY])
    indices = metadata_index.get_indexer(pd.MultiIndex.from_frame(prediction[KEY]))
    if np.any(indices < 0):
        raise ValueError("Prediction keys absent from metadata")
    matched = metadata.iloc[indices].reset_index(drop=True)
    for column in IDENTITY[1:]:
        if column not in prediction:
            continue
        observed = prediction[column]
        if column == "fold":
            equal = pd.to_numeric(observed, errors="raise").to_numpy() == matched[column].to_numpy()
        else:
            equal = observed.to_numpy(dtype=str) == matched[column].to_numpy(dtype=str)
        if not np.all(equal):
            raise ValueError(f"Prediction {column} disagrees with metadata")
    if "score" in prediction and not np.allclose(scores(prediction), scores(matched), atol=1e-12, rtol=1e-12):
        raise ValueError("Prediction score disagrees with metadata")
    for column in columns:
        prediction[column] = scores(prediction, column)
    return prediction[join_key + columns]


def attach_comparator(evaluation, comparator, column, output="prediction_source"):
    """Join a frozen source score by variant, and by seed when supplied."""
    keys = KEY + (["seed"] if "seed" in comparator else [])
    if comparator.duplicated(keys).any():
        raise ValueError("Duplicate source-comparator keys")
    index = pd.MultiIndex.from_frame(comparator[keys])
    indices = index.get_indexer(pd.MultiIndex.from_frame(evaluation[keys]))
    if np.any(indices < 0):
        raise ValueError("Frozen source comparator is missing evaluation rows")
    result = evaluation.copy()
    result[output] = comparator.iloc[indices][column].to_numpy(dtype=float)
    return result


def seed_correlations(predictions, columns):
    """One correlation per model on identical assay/seed evaluation rows."""
    rows = []
    for (assay, seed), group in predictions.groupby(["assay_id", "seed"], sort=True):
        rows.append({"assay_id": assay, **{c: group[c].iloc[0] for c in IDENTITY[1:]},
                     "seed": seed, "n_evaluation": len(group),
                     **{column: rho(group.score.to_numpy(dtype=float), group[column].to_numpy(dtype=float))
                        for column in columns}})
    return pd.DataFrame(rows, columns=IDENTITY + ["seed", "n_evaluation"] + columns)


def paired_seed_summary(per_seed, contrasts, seeds, n_bootstrap, seed, inventory=None, required_columns=None):
    """Difference first, seed-average second, cluster-resample assays last.

    Every declared seed must exist and be finite on both sides. Optional
    required_columns freezes a common valid population across related contrasts.
    Missing assays/seeds are retained as undefined, never silently seed-dropped.
    """
    expected = declared_seeds(seeds)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if per_seed.duplicated(["assay_id", "seed"]).any():
        raise ValueError("Duplicate assay/seed metric rows")
    if not set(per_seed.seed).issubset(expected):
        raise ValueError("Metrics contain undeclared seeds")
    if inventory is None:
        inventory = per_seed[IDENTITY].drop_duplicates()
    inventory = inventory[IDENTITY].drop_duplicates().copy()
    if inventory.duplicated("assay_id").any():
        raise ValueError("Conflicting assay identity in paired inventory")
    for column in IDENTITY[1:]:
        identity = inventory.set_index("assay_id")[column]
        if not per_seed[column].eq(per_seed.assay_id.map(identity)).all():
            raise ValueError(f"Inconsistent {column} in paired seed metrics")
    groups = {assay: group for assay, group in per_seed.groupby("assay_id", sort=True)}
    records = []
    for identity in inventory.to_dict("records"):
        group = groups.get(identity["assay_id"], per_seed.iloc[:0])
        complete = set(group.seed) == expected and len(group) == len(expected)
        for name, (left, right) in contrasts.items():
            check_columns = list(dict.fromkeys([left, right] + list(required_columns or [])))
            valid_rows = np.isfinite(group[check_columns].to_numpy(dtype=float)).all(axis=1)
            valid = bool(complete and valid_rows.all())
            # Paired values are evaluated before averaging; never compare separate
            # available-case seed means or bootstrap seeds as independent assays.
            delta = group[left].to_numpy(dtype=float) - group[right].to_numpy(dtype=float)
            records.append({**identity, "contrast": name, "left": left, "right": right,
                            "n_declared_seeds": len(expected), "n_observed_seeds": len(group),
                            "n_valid_pairs": int(valid_rows.sum()),
                            "status": "ok" if valid else "incomplete_or_undefined_seed",
                            "left_mean": float(group[left].mean()) if valid else np.nan,
                            "right_mean": float(group[right].mean()) if valid else np.nan,
                            "delta": float(delta.mean()) if valid else np.nan})
    per_assay = pd.DataFrame(records)
    rows = []
    for (task, contrast), group in per_assay.groupby(["task", "contrast"], sort=True):
        valid = group[group.status.eq("ok")]
        # Use common resampling draws for contrasts sharing the same valid
        # cluster population (e.g. source improvement and oracle-gap reduction).
        interval = paired_cluster_interval(valid, n_bootstrap, stable_seed(seed, task, "paired-seeds"))
        rows.append({"task": task, "contrast": contrast, "left": group.left.iloc[0], "right": group.right.iloc[0],
                     "n_assays": len(group), "n_valid_paired_assays": len(valid), "n_declared_seeds": len(expected),
                     "left_mean": valid.left_mean.mean(), "right_mean": valid.right_mean.mean(), **interval})
    return per_assay, pd.DataFrame(rows)


def run_paired_seeds(args):
    frame = load_metadata(args.metadata)
    frame["score"] = scores(frame)
    columns = [args.left, args.right]
    prediction = checked_predictions(args.predictions, frame, columns, args.seeds, args.label_protocol, require_seed=True)
    inventory = []
    for declared in args.seeds:
        present = prediction[prediction.seed.eq(declared)]
        missing = len(frame) - len(present)
        inventory.append({"seed": declared, "n_metadata_rows": len(frame), "n_prediction_rows": len(present), "n_missing_rows": missing})
        if missing and not args.allow_evaluation_subset:
            raise ValueError("Prediction population differs from metadata; declare --allow-evaluation-subset for fixed evaluation subsets")
    metadata_columns = list(dict.fromkeys(KEY + IDENTITY)) + ["score"]
    aligned = prediction.merge(frame[metadata_columns], on=KEY, how="left", validate="many_to_one", sort=False)
    metrics = seed_correlations(aligned, columns)
    per_assay, summary = paired_seed_summary(metrics, {"left_minus_right": tuple(columns)}, args.seeds,
                                            args.bootstrap, args.seed, frame[IDENTITY])
    run = Run(args, args.label_protocol + "; paired frozen-prediction multi-seed evaluation", [args.metadata, args.predictions])
    run.table("population.csv", pd.DataFrame(inventory))
    run.table("per_seed.csv", metrics)
    run.table("per_assay.csv", per_assay)
    run.table("summary.csv", summary)
    run.finish(aggregation="within-assay/seed Spearman difference; mean declared seeds within assay; equal-assay task macro",
               uncertainty="95% conditional percentile supercluster bootstrap; fixed predictions, split assignments and selections; no multiplicity adjustment",
               complete_seeds_required=True, evaluation_subset=bool(args.allow_evaluation_subset))
