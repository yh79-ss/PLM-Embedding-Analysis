"""Fixed-prior residual controls with strictly source-only nested selection.

This portable estimator ranks both effects and the fixed prior on the supplied
rows of each fitting assay. It does not reproduce the historical study's
pre-cap full-assay effect-rank universe. Target prior ranks use predictors only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IDENTITY, KEY, Run, load_embeddings, load_metadata, scores, stable_seed
from .metrics import assay_metrics, macro_summary, paired_cluster_interval, ranks, rho
from .probes import fit_probe


PREDICTIONS = ["prior_prediction", "fixed_prediction", "unit_prediction", "prediction"]
TIE_TOLERANCE = 1e-12


def assay_ranks(frame, column):
    """Rank only the rows explicitly passed to this fitting/prediction step."""
    out = np.empty(len(frame), dtype=float)
    for indices in frame.groupby("assay_id", sort=True).indices.values():
        out[indices] = ranks(frame.iloc[indices][column].to_numpy(dtype=float))
    return out


def residual_response(frame, baseline_column):
    return assay_ranks(frame, "score") - assay_ranks(frame, baseline_column)


def combine_prior(prior, residual, multiplier):
    """The prior coefficient stays exactly one; no target-fitted calibration."""
    prior = np.asarray(prior, dtype=float)
    if multiplier == 0:
        return prior.copy()
    prediction = prior + float(multiplier) * np.asarray(residual, dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite prior-plus-residual prediction")
    return prediction


def choose_residual_setting(candidate_scores):
    """Tolerance is relative to the true maximum, never a chained near tie.

    Input has one row per candidate: alpha (NaN for lambda zero), multiplier,
    and mean_rho. Smaller multiplier, then larger positive-lambda alpha, wins.
    """
    if candidate_scores.empty or not np.isfinite(candidate_scores.mean_rho).all():
        raise ValueError("Undefined residual-selection score; selection populations cannot silently change")
    maximum = float(candidate_scores.mean_rho.max())
    tied = candidate_scores[candidate_scores.mean_rho >= maximum - TIE_TOLERANCE]
    chosen = tied.sort_values(["multiplier", "alpha"], ascending=[True, False], na_position="last").iloc[0]
    return {"alpha": float(chosen.alpha) if chosen.multiplier > 0 else np.nan,
            "multiplier": float(chosen.multiplier), "mean_rho": float(chosen.mean_rho)}


def validate_grids(alphas, multipliers):
    alphas, multipliers = list(map(float, alphas)), list(map(float, multipliers))
    if not alphas or len(set(alphas)) != len(alphas) or not all(np.isfinite(a) and a > 0 for a in alphas):
        raise ValueError("Residual alphas must be unique, finite and strictly positive")
    if (not multipliers or len(set(multipliers)) != len(multipliers)
            or not all(np.isfinite(v) and 0 <= v <= 1 for v in multipliers)
            or not {0., 1.}.issubset(multipliers)):
        raise ValueError("Residual lambdas must be unique, finite, in [0, 1], and include 0 and 1")
    return sorted(alphas), sorted(multipliers)


def select_residual(frame, x, alphas, multipliers, baseline_column="baseline"):
    """The caller must pass outer-source rows only, including all preprocessing.

    Every declared source-validation assay contributes once to every candidate.
    Undefined Spearman for any candidate/assay aborts, instead of selecting on a
    candidate-dependent population. Lambda zero occurs once and has no alpha.
    """
    alphas, multipliers = validate_grids(alphas, multipliers)
    folds = sorted(frame.fold.unique())
    if len(folds) < 2:
        raise ValueError("Residual selection needs at least two source folds (three folds overall per task)")
    rows = []
    for fold in folds:
        train = np.flatnonzero(frame.fold.ne(fold))
        valid = np.flatnonzero(frame.fold.eq(fold))
        training, held = frame.iloc[train], frame.iloc[valid]
        response = residual_response(training, baseline_column)
        prior = assay_ranks(held, baseline_column)
        candidates = [(np.nan, 0., prior)]
        for alpha in alphas:
            residual = fit_probe(x[train], response, alpha).predict(x[valid])
            candidates.extend((alpha, value, combine_prior(prior, residual, value))
                              for value in multipliers if value > 0)
        for alpha, multiplier, prediction in candidates:
            for assay, indices in held.groupby("assay_id", sort=True).indices.items():
                value = rho(held.iloc[indices].score.to_numpy(dtype=float), prediction[indices])
                rows.append({"alpha": alpha, "multiplier": multiplier, "inner_fold": int(fold),
                             "assay_id": assay, "rho": value})
    ledger = pd.DataFrame(rows)
    if not np.isfinite(ledger.rho).all():
        raise ValueError("A residual source-validation candidate/assay has undefined Spearman; "
                         "no silent selection-population changes allowed")
    # dropna=False preserves the scientifically alpha-undefined prior candidate.
    candidate_scores = ledger.groupby(["alpha", "multiplier"], dropna=False, as_index=False).agg(
        mean_rho=("rho", "mean"), n_assays=("assay_id", "size"))
    selected = choose_residual_setting(candidate_scores)
    unit = choose_residual_setting(candidate_scores[candidate_scores.multiplier.eq(1)])
    ledger["selected_joint"] = (ledger.multiplier.eq(selected["multiplier"])
                                & (ledger.alpha.eq(selected["alpha"]) if selected["multiplier"] > 0
                                   else ledger.alpha.isna()))
    ledger["selected_unit"] = ledger.multiplier.eq(1) & ledger.alpha.eq(unit["alpha"])
    return selected, unit, ledger, candidate_scores


def residual_contrasts(metrics, n_bootstrap, seed):
    """Paired assay-macro differences, conditionally resampling target clusters."""
    index = list(dict.fromkeys(IDENTITY)) + ["n_variants"]
    wide = metrics.pivot(index=index, columns="model", values="rho").reset_index()
    arms = [(arm, "prior_prediction") for arm in PREDICTIONS if arm != "prior_prediction"]
    arms += [("unit_prediction", "fixed_prediction"), ("prediction", "unit_prediction"),
             ("prediction", "fixed_prediction")]
    per_assay, summaries = [], []
    for a, b in arms:
        paired = wide[index].copy()
        paired["model_a"], paired["model_b"] = a, b
        paired["rho_a"], paired["rho_b"] = wide[a], wide[b]
        paired["paired_valid"] = np.isfinite(wide[a]) & np.isfinite(wide[b])
        paired["delta"] = (wide[a] - wide[b]).where(paired.paired_valid)
        per_assay.append(paired)
        for task, group in paired.groupby("task", sort=True):
            valid = group[group.paired_valid]
            interval = paired_cluster_interval(valid, n_bootstrap, stable_seed(seed, task, a, b, "residual-ci"))
            summaries.append({"task": task, "model_a": a, "model_b": b, "n_assays": len(group),
                              "n_valid": len(valid), "n_excluded": len(group) - len(valid),
                              "fraction_positive": float(valid.delta.gt(0).mean()) if len(valid) else np.nan,
                              **interval,
                              "interval_scope": "pointwise conditional target-supercluster; fixed predictions and selection"})
    return pd.concat(per_assay, ignore_index=True), pd.DataFrame(summaries)


def run_residual(args):
    if not getattr(args, "confirm_no_target_labels", False):
        raise ValueError("Residual analysis requires --confirm-no-target-labels for fixed-prior provenance")
    if not getattr(args, "baseline_provenance", "").strip():
        raise ValueError("Residual analysis requires a nonempty --baseline-provenance declaration")
    alphas, multipliers = validate_grids(args.alphas, args.lambdas)
    if not np.isfinite(args.fixed_alpha) or args.fixed_alpha <= 0:
        raise ValueError("Fixed alpha must be finite and positive")
    if args.bootstrap < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    if args.baseline_column in set(KEY + IDENTITY + ["score", "label_protocol", "analysis_scope"]):
        raise ValueError("The baseline must be a prespecified predictor, not an identity or measured-effect column")
    frame = load_metadata(args.metadata)
    if "label_protocol" in frame:
        declared = frame.label_protocol.astype(str).str.lower()
        if not declared.str.startswith("no-target-label").all():
            raise ValueError("Input label_protocol cannot be relabelled as a no-target-label fixed prior")
    frame["score"] = scores(frame)
    frame[args.baseline_column] = scores(frame, args.baseline_column)
    x = load_embeddings(args.embeddings, frame)
    run = Run(args, "no-target-label protein-disjoint; fixed prior plus source-only nested residual selection",
              [args.metadata, args.embeddings])
    predictions, ledgers, score_tables, selections = [], [], [], []
    for task in sorted(frame.task.unique()):
        for fold in sorted(frame.loc[frame.task.eq(task), "fold"].unique()):
            source_idx = np.flatnonzero(frame.task.eq(task) & frame.fold.ne(fold))
            target_idx = np.flatnonzero(frame.task.eq(task) & frame.fold.eq(fold))
            source, target = frame.iloc[source_idx], frame.iloc[target_idx]
            selected, unit, ledger, candidate_scores = select_residual(
                source, x[source_idx], alphas, multipliers, args.baseline_column)
            cell = {"task": task, "outer_fold": int(fold)}
            ledgers.append(ledger.assign(**cell))
            score_tables.append(candidate_scores.assign(**cell))
            row = {**cell, "selected_alpha": selected["alpha"], "selected_lambda": selected["multiplier"],
                   "selected_source_rho": selected["mean_rho"], "unit_alpha": unit["alpha"],
                   "unit_source_rho": unit["mean_rho"], "fixed_alpha": args.fixed_alpha,
                   "joint_alpha_defined": selected["multiplier"] > 0,
                   "joint_alpha_at_max": selected["multiplier"] > 0 and selected["alpha"] == max(alphas),
                   "unit_alpha_at_max": unit["alpha"] == max(alphas),
                   "lambda_at_zero": selected["multiplier"] == 0,
                   "lambda_at_one": selected["multiplier"] == 1,
                   "n_source_assays": source.assay_id.nunique(), "n_source_folds": source.fold.nunique()}
            selections.append(row)
            prior = assay_ranks(target, args.baseline_column)
            response = residual_response(source, args.baseline_column)
            # Reuse an identical final fit when multiple controls select its alpha.
            required_alphas = {float(args.fixed_alpha), unit["alpha"]}
            if selected["multiplier"] > 0:
                required_alphas.add(selected["alpha"])
            residuals = {alpha: fit_probe(x[source_idx], response, alpha).predict(x[target_idx])
                         for alpha in sorted(required_alphas)}
            result = target[list(dict.fromkeys(KEY + IDENTITY)) + ["score"]].copy()
            result["prior_prediction"] = prior
            result["fixed_prediction"] = combine_prior(prior, residuals[float(args.fixed_alpha)], 1.)
            result["unit_prediction"] = combine_prior(prior, residuals[unit["alpha"]], 1.)
            result["prediction"] = combine_prior(prior, residuals.get(selected["alpha"]), selected["multiplier"])
            result["selected_alpha"], result["selected_lambda"] = selected["alpha"], selected["multiplier"]
            result["unit_alpha"], result["fixed_alpha"] = unit["alpha"], args.fixed_alpha
            predictions.append(result)
            print(f"residual: {task}, fold {fold}, alpha {selected['alpha']:g}, lambda {selected['multiplier']:g}", flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    metrics = assay_metrics(prediction, PREDICTIONS)
    contrast_assays, contrasts = residual_contrasts(metrics, args.bootstrap, args.seed)
    for name, table in (("predictions.csv", prediction), ("source_selection.csv", pd.concat(ledgers, ignore_index=True)),
                        ("source_candidate_scores.csv", pd.concat(score_tables, ignore_index=True)),
                        ("selected_hyperparameters.csv", pd.DataFrame(selections)), ("per_assay.csv", metrics),
                        ("summary.csv", macro_summary(metrics)), ("contrast_per_assay.csv", contrast_assays),
                        ("contrasts.csv", contrasts)):
        run.table(name, table)
    run.finish(n_variants=len(prediction), n_assays=prediction.assay_id.nunique(),
               formula="rank(prior) + lambda * Ridge_alpha(X -> rank(score) - rank(prior)); prior coefficient = 1",
               rank_scope="average-tie percentile ranks on supplied rows within each fitting assay; target prior ranks use predictors only",
               historical_equivalence="not an exact historical replay: full-pre-cap effect ranks are not accepted or reconstructed",
               selection="same-task outer-source-fold OOF unweighted assay-macro Spearman; all candidate/assay scores must be finite",
               tie_rule="within 1e-12 of true maximum: smaller lambda, then larger alpha; lambda zero has undefined alpha",
               target_label_use="final scoring only; no target-label fitting, selection, scaling, or calibration",
               prior_provenance="user-declared fixed predictor independent of all supplied effect labels; not independently certified; this utility does not construct or validate nested source-fitted priors",
               ridge_weighting="equal source rows; assay-macro evaluation", solver="svd",
               uncertainty="pointwise 95% percentile target-supercluster bootstrap conditional on fixed predictions, splits and selection; not multiplicity-adjusted")
