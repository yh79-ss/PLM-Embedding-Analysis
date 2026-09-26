"""Matched source/oracle evaluation and explicitly labeled weight geometry."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .data import IDENTITY, KEY, Run, load_embeddings, load_metadata, scores, stable_seed
from .metrics import ranks, rho
from .probes import fit_probe, rank_assays, select_source_alpha


def weight_cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not np.isfinite(na) or not np.isfinite(nb) or na == 0 or nb == 0:
        return float("nan")
    return float(np.clip((a / na) @ (b / nb), -1, 1))


def oracle_split(n, fraction, seed):
    """Random-row support/evaluation assignment independent of effect values."""
    n_support = int(round(n * fraction))
    support = np.sort(np.random.default_rng(seed).choice(n, size=n_support, replace=False))
    evaluation = np.setdiff1d(np.arange(n), support)
    return support, evaluation


def matched_summary(per_seed, seeds):
    """Use complete paired seeds/assays for both rho columns and their difference."""
    rows = []
    for assay, group in per_seed.groupby("assay_id", sort=True):
        pair_valid = bool(len(group) == len(seeds) and group.status.eq("ok").all())
        cosine_valid = bool(len(group) == len(seeds) and np.isfinite(group.weight_cosine_source_oracle).all())
        record = {"assay_id": assay, **{c: group[c].iloc[0] for c in IDENTITY[1:]},
                  "n_seeds": len(group), "n_valid_pairs": int(group.status.eq("ok").sum()),
                  "paired_status": "ok" if pair_valid else "incomplete_or_undefined_seed",
                  "rho_source": group.rho_source.mean() if pair_valid else np.nan,
                  "rho_oracle": group.rho_oracle.mean() if pair_valid else np.nan,
                  "oracle_gap": group.oracle_gap.mean() if pair_valid else np.nan,
                  "weight_cosine_source_oracle": group.weight_cosine_source_oracle.mean() if cosine_valid else np.nan}
        rows.append(record)
    per_assay = pd.DataFrame(rows)
    summaries = []
    for task, group in per_assay.groupby("task", sort=True):
        paired = group[group.paired_status.eq("ok")]
        summaries.append({"task": task, "n_assays": len(group), "n_valid_paired_assays": len(paired),
                          "rho_source": paired.rho_source.mean(), "rho_oracle": paired.rho_oracle.mean(),
                          "oracle_gap": paired.oracle_gap.mean(),
                          "n_valid_cosine_assays": int(group.weight_cosine_source_oracle.notna().sum()),
                          "weight_cosine_source_oracle": group.weight_cosine_source_oracle.mean()})
    return per_assay, pd.DataFrame(summaries)


def run_diagnose(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    frame["score"] = scores(frame)
    run = Run(args, "target-support diagnostic comparison; source fitting uses no target labels; oracle uses support only",
              [args.metadata, args.embeddings])
    plans, split_rows, inventory = {}, [], []
    for assay, group in frame.groupby("assay_id", sort=True):
        for seed in args.seeds:
            support, ev = oracle_split(len(group), args.support_fraction, stable_seed(seed, assay, "random-row-oracle"))
            plans[assay, seed] = support, ev
            eligible = len(support) >= 3 and len(ev) >= 3
            inventory.append({"assay_id": assay, "seed": seed, "n_support": len(support), "n_evaluation": len(ev), "eligible": eligible})
            for role, idx in (("support", support), ("evaluation", ev)):
                part = group.iloc[idx][KEY].copy()
                part["seed"], part["role"] = seed, role
                split_rows.append(part)
    inventory = pd.DataFrame(inventory)
    run.table("eligibility.csv", inventory)
    run.table("oracle_splits.csv", pd.concat(split_rows, ignore_index=True))
    eligible = inventory.groupby("assay_id").eligible.all()
    predictions, metrics, selections = [], [], []
    for (task, fold), target in frame.groupby(["task", "fold"], sort=True):
        source_idx = np.flatnonzero(frame.task.eq(task) & frame.fold.ne(fold))
        if len(source_idx) < 3:
            raise ValueError(f"Insufficient source rows for {task}, outer fold {fold}")
        source = frame.iloc[source_idx].copy().reset_index(drop=True)
        alpha = args.fixed_alpha
        if args.source_selection == "nested":
            alpha, selection = select_source_alpha(source, x[source_idx], args.alphas)
            selection["outer_fold"], selection["task"] = fold, task
            selections.append(selection)
        source_model = fit_probe(x[source_idx], rank_assays(source), alpha)
        print(f"diagnose: {task}, fold {fold}, shared alpha {alpha:g}", flush=True)
        for assay, group in target.groupby("assay_id", sort=True):
            features = x[group.index.to_numpy()]
            for seed in args.seeds:
                support, ev = plans[assay, seed]
                record = {"assay_id": assay, **{c: group[c].iloc[0] for c in IDENTITY[1:]},
                          "seed": seed, "alpha": alpha, "n_support": len(support), "n_evaluation": len(ev),
                          "rho_source": np.nan, "rho_oracle": np.nan, "oracle_gap": np.nan,
                          "weight_cosine_source_oracle": np.nan, "status": "insufficient_support_or_evaluation"}
                if eligible.loc[assay]:
                    support_y = group.iloc[support].score.to_numpy(dtype=float)
                    oracle_model = fit_probe(features[support], ranks(support_y), alpha)
                    p_source = source_model.predict(features[ev])
                    p_oracle = oracle_model.predict(features[ev])
                    # Evaluation effects do not enter either fit or alpha selection.
                    evaluation_y = group.iloc[ev].score.to_numpy(dtype=float)
                    rs, ro = rho(evaluation_y, p_source), rho(evaluation_y, p_oracle)
                    record.update(rho_source=rs, rho_oracle=ro, oracle_gap=ro - rs,
                                  weight_cosine_source_oracle=weight_cosine(source_model.raw_coef, oracle_model.raw_coef),
                                  status="ok" if np.isfinite(rs) and np.isfinite(ro) else "undefined_correlation")
                    result = group.iloc[ev][list(dict.fromkeys(KEY + IDENTITY))].copy()
                    result["seed"], result["score"] = seed, evaluation_y
                    result["prediction_source"], result["prediction_oracle"] = p_source, p_oracle
                    predictions.append(result)
                metrics.append(record)
    per_seed = pd.DataFrame(metrics)
    per_assay, summary = matched_summary(per_seed, args.seeds)
    run.table("per_seed.csv", per_seed)
    run.table("per_assay.csv", per_assay)
    run.table("summary.csv", summary)
    if predictions:
        run.table("predictions.csv", pd.concat(predictions, ignore_index=True))
    if selections:
        run.table("source_selection.csv", pd.concat(selections, ignore_index=True))
    run.finish(n_eligible_assays=int(eligible.sum()), split="random rows within each target assay; same evaluation rows for both probes",
               source_label_protocol="no-target-label fitting; outer fold excluded",
               oracle_label_protocol="assay-local target support; not a theoretical ceiling or cross-family result",
               alpha_rule="same alpha for source and oracle, fixed or selected only on source folds",
               cosine_role="source-vs-oracle coefficients in the same original feature coordinates; exploratory descriptive geometry",
               rank_scope="percentile ranks computed separately in each fitting assay/support set")


def run_cosine(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    frame["score"] = scores(frame)
    run = Run(args, "descriptive fold-local label-fitted probes; not no-target-label evaluation", [args.metadata, args.embeddings])
    records, weights, rows = [], {}, []
    for task, task_frame in frame.groupby("task", sort=True):
        for fold in sorted(frame.fold.unique()):
            group = task_frame[task_frame.fold.eq(fold)]
            weight = np.full(x.shape[1], np.nan)
            intercept = np.nan
            status = "insufficient_rows"
            if len(group) >= 3:
                y = np.empty(len(group))
                for idx in group.groupby("assay_id").indices.values():
                    values = group.iloc[idx].score.to_numpy(dtype=float)
                    y[idx] = (norm.ppf(np.clip((rankdata(values, method="average") - 0.5) / len(values), 1e-6, 1 - 1e-6))
                              if args.label_transform == "rank-gaussian" else ranks(values))
                model = fit_probe(x[group.index.to_numpy()], y, args.alpha)
                weight, intercept = model.raw_coef, model.raw_intercept
                status = "ok" if np.isfinite(weight_cosine(weight, weight)) else "zero_or_invalid_weight"
            weights[task, fold] = weight
            records.append({"task": task, "fold": fold, "n_variants": len(group), "n_assays": group.assay_id.nunique(),
                            "alpha": args.alpha, "label_transform": args.label_transform, "raw_intercept": intercept,
                            "weight_norm": float(np.linalg.norm(weight)), "status": status})
        for a, b in combinations(sorted(frame.fold.unique()), 2):
            value = weight_cosine(weights[task, a], weights[task, b])
            rows.append({"task": task, "fold_a": a, "fold_b": b, "weight_cosine": value,
                         "status": "ok" if np.isfinite(value) else "undefined_weight"})
    pairwise = pd.DataFrame(rows, columns=["task", "fold_a", "fold_b", "weight_cosine", "status"])
    metadata = pd.DataFrame(records)
    summary = []
    for task in sorted(frame.task.unique()):
        values = pairwise.loc[pairwise.task.eq(task), "weight_cosine"]
        summary.append({"task": task, "n_pairs": len(values), "n_valid_pairs": int(values.notna().sum()),
                        "mean_weight_cosine": values.mean(), "median_weight_cosine": values.median()})
    run.table("pairwise_cosine.csv", pairwise)
    run.table("probe_metadata.csv", metadata)
    run.table("summary.csv", pd.DataFrame(summary))
    np.savez_compressed(run.path / "probe_weights.npz", weights=np.stack([weights[t, f] for t, f in weights]),
                        task=np.asarray([t for t, f in weights], dtype=str), fold=np.asarray([f for t, f in weights], dtype=int))
    run.finish(cosine_role="pairs of same-task probes each fitted only within a different fold",
               coefficient_space="original input feature coordinates: scaler-standardized coefficient divided by fitted feature scale",
               uncertainty="none; fold pairs share fitted probes and are not independent replicates",
               label_transform=args.label_transform)
