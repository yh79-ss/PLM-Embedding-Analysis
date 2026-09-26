"""Source-only nested Ridge and explicitly separate target-support diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .data import IDENTITY, KEY, Run, load_embeddings, load_metadata, scores, single_substitutions, stable_seed
from .metrics import assay_metrics, macro_summary, ranks, rho


@dataclass
class FittedProbe:
    scaler: StandardScaler
    model: Ridge

    @property
    def raw_coef(self):
        return np.asarray(self.model.coef_, dtype=float) / self.scaler.scale_

    @property
    def raw_intercept(self):
        return float(self.model.intercept_ - self.scaler.mean_ @ self.raw_coef)

    def predict(self, x):
        prediction = self.model.predict(self.scaler.transform(x))
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite Ridge prediction")
        return prediction


def fit_probe(x_train, y_train, alpha):
    # Exact SVD avoids treating a finite iterative-solver result as convergence.
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=float(alpha), solver="svd").fit(scaler.transform(x_train), y_train)
    return FittedProbe(scaler, model)


def fit_predict(x_train, y_train, x_test, alpha):
    return fit_probe(x_train, y_train, alpha).predict(x_test)


def rank_assays(frame):
    out = np.empty(len(frame), dtype=float)
    for _, idx in frame.groupby("assay_id", sort=True).indices.items():
        out[idx] = ranks(frame.iloc[idx].score.to_numpy(dtype=float))
    return out


def choose_alpha(values, alphas, prefer_larger=False):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Undefined alpha-selection score; inspect constant labels or insufficient assay coverage")
    ties = np.flatnonzero(values >= values.max() - 1e-12)
    return float(max(alphas[i] for i in ties) if prefer_larger else min(alphas[i] for i in ties))


def select_source_alpha(frame, x, alphas):
    """The caller supplies only outer-source rows, including for preprocessing."""
    folds = sorted(frame.fold.unique())
    if len(folds) < 2:
        raise ValueError("Nested source selection needs at least two source folds (three folds overall)")
    rows = []
    for alpha in alphas:
        for fold in folds:
            train, valid = np.flatnonzero(frame.fold.ne(fold)), np.flatnonzero(frame.fold.eq(fold))
            prediction = fit_predict(x[train], rank_assays(frame.iloc[train]), x[valid], alpha)
            held = frame.iloc[valid].copy()
            held["prediction"] = prediction
            for assay, group in held.groupby("assay_id", sort=True):
                rows.append({"alpha": alpha, "inner_fold": int(fold), "assay_id": assay,
                             "rho": rho(group.score.to_numpy(dtype=float), group.prediction.to_numpy())})
    table = pd.DataFrame(rows)
    if not np.isfinite(table.rho).all():
        raise ValueError("A source validation assay has undefined Spearman; no silent selection-population changes allowed")
    means = [float(table.loc[table.alpha.eq(a), "rho"].mean()) for a in alphas]
    selected = choose_alpha(means, alphas)
    table["selected"] = table.alpha.eq(selected)
    return selected, table


def run_probe(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    inputs = [args.metadata, args.embeddings]
    if args.aux_embeddings:
        auxiliary = load_embeddings(args.aux_embeddings, frame)
        x = np.column_stack([x, auxiliary])
        inputs.append(args.aux_embeddings)
    frame["score"] = scores(frame)
    tasks = sorted(frame.task.unique())
    if bool(args.source_task) != bool(args.target_task):
        raise ValueError("Use --source-task and --target-task together")
    pairs = [(args.source_task, args.target_task)] if args.source_task else [(t, t) for t in tasks]
    if any(s not in tasks or t not in tasks for s, t in pairs):
        raise ValueError("Requested task is absent from metadata")
    protocol = "no-target-label protein-disjoint; source-only nested alpha selection"
    if args.shuffle_source_labels:
        protocol += "; shuffled-source-label control"
    run = Run(args, protocol, inputs)
    predictions, selections = [], []
    for source_task, target_task in pairs:
        for fold in sorted(frame.loc[frame.task.eq(target_task), "fold"].unique()):
            source_idx = np.flatnonzero(frame.task.eq(source_task) & frame.fold.ne(fold))
            target_idx = np.flatnonzero(frame.task.eq(target_task) & frame.fold.eq(fold))
            source = frame.iloc[source_idx].copy().reset_index(drop=True)
            if args.shuffle_source_labels:
                for assay, idx in source.groupby("assay_id").indices.items():
                    rng = np.random.default_rng(stable_seed(args.seed, source_task, fold, assay, "shuffle"))
                    source.loc[idx, "score"] = rng.permutation(source.loc[idx, "score"].to_numpy())
            selected, selection = select_source_alpha(source, x[source_idx], args.alphas)
            selection["outer_fold"], selection["source_task"], selection["target_task"] = fold, source_task, target_task
            selections.append(selection)
            result = frame.iloc[target_idx][list(dict.fromkeys(KEY + IDENTITY)) + ["score"]].copy()
            result["prediction"] = fit_predict(x[source_idx], rank_assays(source), x[target_idx], selected)
            result["fixed_prediction"] = fit_predict(x[source_idx], rank_assays(source), x[target_idx], args.fixed_alpha)
            result["selected_alpha"], result["source_task"] = selected, source_task
            if "baseline" in frame:
                result["baseline"] = scores(frame.iloc[target_idx], "baseline")
            predictions.append(result)
            print(f"probe: {source_task} -> {target_task}, fold {fold}, alpha {selected:g}", flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    columns = ["prediction", "fixed_prediction"] + (["baseline"] if "baseline" in prediction else [])
    metrics = assay_metrics(prediction, columns)
    run.table("predictions.csv", prediction)
    run.table("source_selection.csv", pd.concat(selections, ignore_index=True))
    run.table("per_assay.csv", metrics)
    source_scores = metrics.pivot(index=["assay_id", "task", "protein_id", "super_cluster", "fold", "n_variants"],
                                  columns="model", values="rho").reset_index().rename(columns={
        "prediction": "rho_source", "fixed_prediction": "rho_source_fixed", "baseline": "rho_baseline"})
    run.table("source_scores.csv", source_scores)
    run.table("summary.csv", macro_summary(metrics))
    run.finish(n_variants=len(prediction), n_assays=prediction.assay_id.nunique(),
               rank_scope="within each fitting assay; inner fits recompute ranks using inner-training labels only",
               ridge_weighting="equal rows; evaluation is equal assays", solver="svd",
               feature_blocks="key-aligned base plus auxiliary concatenation, each coordinate source-standardized; no block-scale tuning" if args.aux_embeddings else "base only")


def paired_support_split(frame, seed, budget):
    """Corrected matched-position design: replacements use actual evaluation sites."""
    rng = np.random.default_rng(seed)
    pos, types = frame.position.to_numpy(), frame.substitution.to_numpy()
    sites = np.unique(pos)
    chosen = rng.choice(sites, size=int(np.ceil(len(sites) / 3)), replace=False)
    evaluation = []
    for site in sorted(chosen):
        idx = np.flatnonzero(pos == site)
        if len(idx) >= 2:
            evaluation.extend(rng.choice(idx, size=max(1, len(idx) // 2), replace=False))
    evaluation = np.sort(evaluation).astype(int)
    pool = np.flatnonzero(~np.isin(pos, chosen))
    disjoint = np.sort(rng.choice(pool, size=min(budget, len(pool)), replace=False))
    overlap_pool = np.setdiff1d(np.flatnonzero(np.isin(pos, pos[evaluation])), evaluation)
    overlap = disjoint.copy()
    replacements = 0
    for category in sorted(np.unique(types[disjoint])):
        slots = np.flatnonzero(types[disjoint] == category)
        candidates = overlap_pool[types[overlap_pool] == category]
        n = min(len(slots), len(candidates))
        if n:
            overlap[rng.choice(slots, n, replace=False)] = rng.choice(candidates, n, replace=False)
            replacements += n
    overlap.sort()
    if (np.intersect1d(evaluation, overlap).size or np.intersect1d(evaluation, disjoint).size
            or np.intersect1d(pos[evaluation], pos[disjoint]).size
            or len(np.unique(overlap)) != len(overlap)
            or not np.array_equal(np.sort(types[overlap]), np.sort(types[disjoint]))
            or not np.isin(pos[np.setdiff1d(overlap, disjoint)], pos[evaluation]).all()):
        raise ValueError("Matched support split invariant failed")
    stats = {"n_support": len(disjoint), "n_evaluation": len(evaluation),
             "n_evaluation_positions": len(np.unique(pos[evaluation])), "n_replacements": replacements,
             "evaluation_position_overlap": float(np.isin(pos[evaluation], pos[overlap]).mean()) if len(evaluation) else 0.0}
    stats["eligible"] = bool(len(disjoint) == budget and len(evaluation) >= 30
                             and stats["n_evaluation_positions"] >= 4 and replacements >= 10
                             and len(np.unique(pos[disjoint])) >= 3 and len(np.unique(pos[overlap])) >= 3)
    return evaluation, {"disjoint": disjoint, "overlap": overlap}, stats


def select_support_alpha(x, y, positions, alphas):
    oof = np.empty((len(y), len(alphas)))
    for train, valid in GroupKFold(n_splits=3).split(x, groups=positions):
        for i, alpha in enumerate(alphas):
            # Support diagnostic preserves its historical rank/n convention.
            from scipy.stats import rankdata
            yr = rankdata(y[train], method="average") / len(train)
            oof[valid, i] = fit_predict(x[train], yr, x[valid], alpha)
    values = [rho(y, oof[:, i]) for i in range(len(alphas))]
    return choose_alpha(values, alphas, prefer_larger=True), values


def run_support(args):
    from scipy.stats import rankdata
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    single = single_substitutions(frame)
    if len(single) != len(frame):
        raise ValueError("support requires only canonical nonsynonymous single substitutions")
    run = Run(args, "target-support diagnostic; support labels permitted; evaluation labels scoring only", [args.metadata, args.embeddings])
    plans, inventory, split_rows = {}, [], []
    for assay, group in single.groupby("assay_id", sort=True):
        for seed in args.seeds:
            ev, arms, stats = paired_support_split(group, stable_seed(seed, assay, "support"), args.support_size)
            plans[assay, seed] = ev, arms
            inventory.append({"assay_id": assay, "seed": seed, **stats})
            for role, idx in {"evaluation": ev, **arms}.items():
                rows = group.iloc[idx][KEY].copy()
                rows["seed"], rows["role"] = seed, role
                split_rows.append(rows)
    inventory = pd.DataFrame(inventory)
    run.table("eligibility.csv", inventory)
    run.table("support_splits.csv", pd.concat(split_rows, ignore_index=True))
    eligible = inventory.groupby("assay_id").eligible.all()
    predictions, selection_rows = [], []
    # All eligibility and row assignment decisions above are label-independent.
    for assay, group in single.groupby("assay_id", sort=True):
        if not eligible.loc[assay]:
            continue
        y, features = scores(group), x[group.index.to_numpy()]
        for seed in args.seeds:
            ev, arms = plans[assay, seed]
            result = group.iloc[ev][list(dict.fromkeys(KEY + IDENTITY))].copy()
            result["score"], result["seed"] = y[ev], seed
            for arm, support in arms.items():
                alpha, values = select_support_alpha(features[support], y[support], group.position.to_numpy()[support], args.alphas)
                for a, value in zip(args.alphas, values):
                    selection_rows.append({"assay_id": assay, "seed": seed, "arm": arm, "alpha": a, "rho": value, "selected": a == alpha})
                yr = rankdata(y[support], method="average") / len(support)
                for readout, a in (("fixed", args.fixed_alpha), ("selected", alpha)):
                    result[f"{arm}_{readout}"] = fit_predict(features[support], yr, features[ev], a)
            predictions.append(result)
    run.table("support_selection.csv", pd.DataFrame(selection_rows, columns=["assay_id", "seed", "arm", "alpha", "rho", "selected"]))
    if predictions:
        prediction = pd.concat(predictions, ignore_index=True)
        columns = [f"{arm}_{readout}" for arm in ("disjoint", "overlap") for readout in ("fixed", "selected")]
        metrics = []
        for seed, group in prediction.groupby("seed"):
            m = assay_metrics(group, columns)
            m["seed"] = seed
            metrics.append(m)
        per_seed = pd.concat(metrics, ignore_index=True)
        # Strict seed completeness: an undefined seed makes that assay/model undefined.
        per_assay = per_seed.groupby(["assay_id", "task", "model", "super_cluster"], as_index=False).agg(
            rho=("rho", lambda s: s.mean() if s.notna().all() else np.nan), n_seeds=("seed", "size"))
        run.table("predictions.csv", prediction)
        run.table("per_seed.csv", per_seed)
        run.table("per_assay.csv", per_assay)
        run.table("summary.csv", macro_summary(per_assay))
    run.finish(n_eligible_assays=int(eligible.sum()), n_excluded_assays=int((~eligible).sum()),
               eligibility="all declared seeds pass; full requested support budget; metadata-only gates",
               no_eligible_assays=not bool(eligible.any()))
