"""Saved-prediction controls, conditional uncertainty, and descriptive errors."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import IDENTITY, KEY, Run, keys, load_metadata, read_csv, require_columns, scores, single_substitutions, stable_seed
from .metrics import paired_cluster_interval, permutation_pvalue, ranks, rho


# User annotations must never be overwritten by generated metrics/provenance.
RESERVED_GROUP_COLUMNS = frozenset(KEY + IDENTITY + [
    "score", "model", "n_variants", "rho", "status", "analysis_scope", "label_protocol",
    "random_mean", "random_p05", "random_p95", "permutation_p_greater", "permutation_p_holm",
    "rho_source", "rho_oracle", "oracle_gap", "recovery", "group_column", "group_value",
    "n_assays", "n_valid", "mean_rho", "ci_low", "ci_high", "n_clusters", "delta",
    "baseline", "n_valid_pairs", "fraction_below_baseline", "n_valid_recovery", "mean_recovery",
])


def holm(pvalues):
    """Retain undefined hypotheses in the declared family (as p=1 for adjustment)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(np.where(np.isfinite(p), p, 1), kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(np.where(np.isfinite(p[order]), p[order], 1) * (len(p) - np.arange(len(p)))))
    adjusted[~np.isfinite(p)] = np.nan
    return adjusted


def prediction_frame(args):
    """Align one prediction realization to metadata; never pool repeated seeds."""
    group_columns = set(getattr(args, "group_by", []))
    collision = group_columns & (RESERVED_GROUP_COLUMNS | set(args.columns) |
                                 {getattr(args, "baseline_column", None), getattr(args, "oracle_column", None)})
    if collision:
        raise ValueError(f"Reserved subgroup annotation columns: {sorted(collision)}")
    metadata, prediction = load_metadata(args.metadata), read_csv(args.predictions)
    require_columns(prediction, KEY)
    if args.prediction_seed is not None:
        require_columns(prediction, ["seed"])
        prediction = prediction[prediction.seed.eq(str(args.prediction_seed))].copy()
        if prediction.empty:
            raise ValueError("Requested prediction seed is absent")
    if prediction.duplicated(KEY).any():
        raise ValueError("Prediction keys repeat; select one seed with --prediction-seed and one source-task/model run")
    if args.label_protocol == "no-target-label" and "label_protocol" in prediction:
        if not prediction.label_protocol.str.startswith("no-target-label").all():
            raise ValueError("Cannot relabel diagnostic/adaptation predictions as no-target-label")
    km, kp = keys(metadata), keys(prediction)
    common = km.intersection(kp, sort=False)
    if not args.allow_intersection and (len(common) != len(km) or len(common) != len(kp)):
        raise ValueError("Populations differ; align first or explicitly use --allow-intersection")
    if not len(common):
        raise ValueError("No common prediction/metadata keys")
    matched = metadata.iloc[km.get_indexer(common)].copy().reset_index(drop=True)
    supplied = prediction.iloc[kp.get_indexer(common)].reset_index(drop=True)
    for col in IDENTITY[1:]:
        if col in supplied and not matched[col].astype(str).eq(supplied[col].astype(str)).all():
            raise ValueError(f"Prediction identity disagrees with metadata: {col}")
    matched["score"] = scores(matched)
    columns = list(dict.fromkeys(args.columns + [getattr(args, c, None) for c in ("baseline_column", "oracle_column")]))
    for col in filter(None, columns):
        if col in KEY + IDENTITY + ["score", "label_protocol", "analysis_scope"]:
            raise ValueError(f"Reserved prediction column: {col}")
        matched[col] = scores(supplied if col in supplied else matched, col)
    for col in getattr(args, "group_by", []):
        require_columns(matched, [col])
        if matched.groupby("assay_id")[col].nunique().gt(1).any():
            raise ValueError(f"Subgroup {col} must be constant within each assay")
    excluded = [{"input": name, "assay_id": a, "mutant": m} for name, index in (("metadata", km), ("predictions", kp)) for a, m in index.difference(common)]
    return matched, pd.DataFrame(excluded, columns=["input", *KEY])


def permutation_correlations(y, predictions, repeats, seed):
    """Tie-preserving label permutation; same draws for all frozen predictors."""
    yr = rankdata(y).astype(float)
    yr -= yr.mean()
    pr = np.column_stack([rankdata(predictions[:, j]) for j in range(predictions.shape[1])]).astype(float)
    pr -= pr.mean(axis=0)
    denom = np.linalg.norm(yr) * np.linalg.norm(pr, axis=0)
    result = np.full((repeats, predictions.shape[1]), np.nan)
    valid = (denom > 0) & (len(y) >= 3)
    rng = np.random.default_rng(seed)
    for i in range(repeats):
        result[i, valid] = (rng.permutation(yr) @ pr[:, valid]) / denom[valid]
    return result


def run_evaluate(args):
    frame, exclusions = prediction_frame(args)
    columns = list(dict.fromkeys(args.columns + [c for c in (args.baseline_column, args.oracle_column) if c]))
    run = Run(args, args.label_protocol, [args.metadata, args.predictions])
    run.table("excluded_keys.csv", exclusions)
    run.table("evaluation_keys.csv", frame[list(dict.fromkeys(KEY + IDENTITY))])
    records, nulls, recovery = [], {}, []
    for assay, group in frame.groupby("assay_id", sort=True):
        y, p = group.score.to_numpy(float), group[columns].to_numpy(float)
        null = permutation_correlations(y, p, args.permutations, stable_seed(args.seed, assay, "evaluation-null"))
        identity = {c: group[c].iloc[0] for c in IDENTITY + args.group_by}
        values = {}
        for j, col in enumerate(columns):
            value = rho(y, p[:, j])
            values[col] = value
            valid = np.isfinite(value) and np.isfinite(null[:, j]).all()
            nulls[assay, col] = null[:, j]
            records.append({**identity, "model": col, "n_variants": len(group), "rho": value,
                            "random_mean": null[:, j].mean() if valid else np.nan,
                            "random_p05": np.quantile(null[:, j], .05) if valid else np.nan,
                            "random_p95": np.quantile(null[:, j], .95) if valid else np.nan,
                            "permutation_p_greater": permutation_pvalue(null[:, j], value) if valid else np.nan,
                            "status": "ok" if valid else "undefined_correlation"})
        if args.oracle_column:
            source, oracle = values[args.columns[0]], values[args.oracle_column]
            reference = null[:, columns.index(args.columns[0])].mean()
            valid = np.isfinite([source, oracle, reference]).all() and oracle >= args.oracle_min_rho and oracle - reference > 1e-12
            recovery.append({**identity, "rho_source": source, "rho_oracle": oracle, "random_mean": reference,
                             "oracle_gap": oracle - source, "recovery": (source - reference) / (oracle - reference) if valid else np.nan,
                             "status": "ok" if valid else "undefined_or_small_oracle_denominator"})
    per_assay = pd.DataFrame(records)
    summaries, contrasts, subgroup_rows = [], [], []
    for (task, model), group in per_assay.groupby(["task", "model"], sort=True):
        valid = group[group.status.eq("ok")]
        point = valid.rho.mean()
        null = np.mean([nulls[a, model] for a in valid.assay_id], axis=0) if len(valid) else np.full(args.permutations, np.nan)
        interval = paired_cluster_interval(valid.assign(delta=valid.rho), args.bootstrap, stable_seed(args.seed, task, model, "interval"))
        summaries.append({"task": task, "model": model, "n_assays": len(group), "n_valid": len(valid), "mean_rho": point,
                          **{k: v for k, v in interval.items() if k != "delta"},
                          "random_mean": null.mean(), "random_p05": np.quantile(null, .05), "random_p95": np.quantile(null, .95),
                          "permutation_p_greater": permutation_pvalue(null, point) if len(valid) else np.nan})
        for column in args.group_by:
            for value, subgroup in group.groupby(column, sort=True):
                finite = subgroup[subgroup.status.eq("ok")]
                ci = paired_cluster_interval(finite.assign(delta=finite.rho), args.bootstrap, stable_seed(args.seed, task, model, column, value))
                subgroup_rows.append({"task": task, "model": model, "group_column": column, "group_value": value,
                                      "n_assays": len(subgroup), "n_valid": len(finite), "mean_rho": finite.rho.mean(),
                                      **{k: v for k, v in ci.items() if k != "delta"}})
        if args.baseline_column and model != args.baseline_column:
            baseline = per_assay[per_assay.model.eq(args.baseline_column)][["assay_id", "rho"]].rename(columns={"rho": "rho_baseline"})
            paired = group.merge(baseline, on="assay_id", validate="one_to_one")
            paired = paired[np.isfinite(paired.rho) & np.isfinite(paired.rho_baseline)].copy()
            paired["delta"] = paired.rho - paired.rho_baseline
            ci = paired_cluster_interval(paired, args.bootstrap, stable_seed(args.seed, task, model, "baseline"))
            contrasts.append({"task": task, "model": model, "baseline": args.baseline_column, "n_assays": len(group),
                              "n_valid_pairs": len(paired), "fraction_below_baseline": (paired.delta < 0).mean(), **ci})
    summary = pd.DataFrame(summaries)
    summary["permutation_p_holm"] = holm(summary.permutation_p_greater)
    run.table("per_assay.csv", per_assay)
    run.table("summary.csv", summary)
    if contrasts:
        run.table("baseline_comparisons.csv", pd.DataFrame(contrasts))
    if subgroup_rows:
        run.table("subgroups.csv", pd.DataFrame(subgroup_rows))
    if recovery:
        recovery = pd.DataFrame(recovery)
        run.table("recovery.csv", recovery)
        run.table("recovery_summary.csv", recovery.groupby("task", as_index=False).agg(
            n_assays=("assay_id", "size"), n_valid_recovery=("recovery", "count"), mean_recovery=("recovery", "mean")))
    run.finish(uncertainty="95% conditional supercluster intervals; fixed predictions, no model refits or multiplicity correction for intervals",
               null="within-assay evaluation-label exchangeability, preserving ties; not a source-label-shuffle refit control",
               multiplicity="Holm over all task/model ranking-signal tests in summary.csv; not across separate runs or baseline contrasts",
               recovery="mean of per-assay (source - random mean)/(oracle - random mean), excluding weak/invalid denominators; not clipped",
               declared_protocol_not_independently_certified=True)


def pair_accuracy(y, prediction, max_pairs, seed):
    n = len(y)
    if n * (n - 1) // 2 <= max_pairs:
        a, b = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(seed)
        a = rng.integers(n, size=max_pairs)
        b = rng.integers(n - 1, size=max_pairs)
        b += b >= a  # Uniform distinct ordered pairs, sampled with replacement.
    dy, dp = y[a] - y[b], prediction[a] - prediction[b]
    keep = dy != 0
    value = np.mean((np.sign(dy[keep]) == np.sign(dp[keep])) + .5 * (dp[keep] == 0)) if keep.any() else np.nan
    return float(value), int(keep.sum())


def run_errors(args):
    frame, exclusions = prediction_frame(args)
    run = Run(args, "descriptive evaluation-label error audit; input protocol: " + args.label_protocol, [args.metadata, args.predictions])
    run.table("excluded_keys.csv", exclusions)
    rows, curves, strata = [], [], []
    for assay, group in frame.groupby("assay_id", sort=True):
        identity = {c: group[c].iloc[0] for c in IDENTITY}
        y = group.score.to_numpy(float)
        yr = ranks(y)
        single = single_substitutions(group)
        position = pd.Series("unsupported", index=group.index)
        aa = pd.Series("unsupported", index=group.index)
        if len(single):
            position.loc[single.index] = (np.minimum((ranks(single.position.to_numpy()) * 4).astype(int), 3) + 1).astype(str)
            aa.loc[single.index] = single.wt_aa.map({a: c for c, letters in (("nonpolar", "AVLIMFWYPGC"), ("polar", "STNQ"), ("charged", "KRHDE")) for a in letters})
        for col in args.columns:
            p = group[col].to_numpy(float)
            pr = ranks(p)
            acc, n_pairs = pair_accuracy(y, p, args.max_pairs, stable_seed(args.seed, assay, "pairs"))
            record = {**identity, "model": col, "n_variants": len(group), "rho": rho(y, p),
                      "pair_accuracy": acc, "n_informative_pairs": n_pairs, "rank_rmse": float(np.sqrt(np.mean((yr - pr) ** 2)))}
            record["ranking_status"] = "ok" if np.isfinite(record["rho"]) else "undefined_constant_or_too_small"
            for side, positive, directed in (("top", yr >= 1 - args.tail_fraction, p), ("bottom", yr <= args.tail_fraction, -p)):
                defined = 0 < positive.sum() < len(positive)
                record[side + "_n_positive"] = int(positive.sum())
                record[side + "_ap"] = average_precision_score(positive, directed) if defined else np.nan
                record[side + "_auroc"] = roc_auc_score(positive, directed) if defined else np.nan
            # Original predictions, NOT re-ranked predictions or in-sample calibrated residuals.
            if args.percentile_predictions:
                slope = float(np.sum((p - p.mean()) * (yr - yr.mean())) / np.sum((p - p.mean()) ** 2)) if np.ptp(p) else np.nan
                record.update(prediction_rank_bias=float(np.mean(yr - p)), calibration_slope=slope,
                              calibration_intercept=float(yr.mean() - slope * p.mean()))
            rows.append(record)
            bins = np.minimum((pr * args.bins).astype(int), args.bins - 1)
            for b in np.unique(bins):
                chosen = bins == b
                curves.append({**identity, "model": col, "prediction_bin": int(b + 1), "n_variants": int(chosen.sum()),
                               "mean_prediction": p[chosen].mean(), "mean_prediction_rank": pr[chosen].mean(), "mean_effect_rank": yr[chosen].mean()})
            categories = {"effect_rank_quartile": (np.minimum((yr * 4).astype(int), 3) + 1).astype(str),
                          "observed_position_quartile": position.to_numpy(), "wt_aa_category": aa.to_numpy()}
            for name, values in categories.items():
                for value in sorted(set(values)):
                    chosen = values == value
                    strata.append({**identity, "model": col, "stratum": name, "value": value, "n_variants": int(chosen.sum()),
                                   "mean_rank_error_pred_minus_effect": float(np.mean(pr[chosen] - yr[chosen])),
                                   "rank_rmse": float(np.sqrt(np.mean((pr[chosen] - yr[chosen]) ** 2))), "rho": rho(y[chosen], p[chosen])})
    run.table("per_assay.csv", pd.DataFrame(rows))
    run.table("calibration_bins.csv", pd.DataFrame(curves))
    run.table("error_strata.csv", pd.DataFrame(strata))
    run.finish(rank_scope="each evaluated assay, including ties; no evaluation-fitted predictions are exported",
               prediction_rank_bias="effect percentile rank minus original prediction; only with explicit --percentile-predictions",
               calibration="evaluation-only slope/intercept descriptive fit; not out-of-sample calibration or probability reliability",
               strata="within-assay effect-rank quartile, observed-position quartile, WT residue class; nonsingle mutations remain unsupported")
