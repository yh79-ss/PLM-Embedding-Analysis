"""Label-free embedding/library diagnostics and descriptive context agreement."""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from .data import KEY, Run, load_embeddings, load_metadata, require_columns, scores, single_substitutions, stable_seed
from .metrics import js_divergence, mmd_squared, moment_distances, rho


def sample_indices(idx, cap, seed):
    return np.sort(np.random.default_rng(seed).choice(idx, size=min(len(idx), cap), replace=False))


def run_shift(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    run = Run(args, "descriptive embedding geometry; no effect-label use", [args.metadata, args.embeddings])
    if args.projection_dim:
        rng = np.random.default_rng(args.seed)
        projection = rng.normal(0, 1 / np.sqrt(args.projection_dim), size=(x.shape[1], args.projection_dim))
        x = x @ projection
    rows, reference = [], []
    for task, task_frame in frame.groupby("task", sort=True):
        for fold in sorted(task_frame.fold.unique()):
            source = task_frame[task_frame.fold.ne(fold)]
            held = task_frame[task_frame.fold.eq(fold)]
            for side, collection in (("source_reference", source), ("held_out", held)):
                for assay, group in collection.groupby("assay_id", sort=True):
                    pool = source[source.super_cluster.ne(group.super_cluster.iloc[0])]
                    record = {"task": task, "outer_fold": fold, "assay_id": assay,
                              "super_cluster": group.super_cluster.iloc[0], "side": side,
                              "n_pool_available": len(pool), "n_assay_available": len(group)}
                    if len(pool) < 2 or len(group) < 2:
                        record.update(mmd_squared=np.nan, centroid_distance=np.nan, covariance_distance=np.nan,
                                      n_pool_sample=0, n_assay_sample=0, status="insufficient_rows")
                    else:
                        a = sample_indices(pool.index.to_numpy(), args.max_samples, stable_seed(args.seed, task, fold, assay, "pool"))
                        b = sample_indices(group.index.to_numpy(), args.max_samples, stable_seed(args.seed, task, fold, assay, "assay"))
                        centroid, covariance = moment_distances(x[a], x[b])
                        record.update(mmd_squared=mmd_squared(x[a], x[b]), centroid_distance=centroid,
                                      covariance_distance=covariance, n_pool_sample=len(a), n_assay_sample=len(b), status="ok")
                    (reference if side == "source_reference" else rows).append(record)
    columns = ["task", "outer_fold", "assay_id", "super_cluster", "side", "n_pool_available", "n_assay_available",
               "mmd_squared", "centroid_distance", "covariance_distance", "n_pool_sample", "n_assay_sample", "status"]
    held, reference = pd.DataFrame(rows, columns=columns), pd.DataFrame(reference, columns=columns)
    held["reference_percentile"] = np.nan
    held["n_reference_assays"] = 0
    for idx, row in held.iterrows():
        values = reference.loc[reference.task.eq(row.task) & reference.outer_fold.eq(row.outer_fold), "mmd_squared"].dropna().to_numpy()
        held.loc[idx, "n_reference_assays"] = len(values)
        if len(values) and np.isfinite(row.mmd_squared):
            held.loc[idx, "reference_percentile"] = 100 * (np.mean(values < row.mmd_squared) + 0.5 * np.mean(values == row.mmd_squared))
    run.table("held_out.csv", held)
    run.table("source_reference.csv", reference)
    run.table("summary.csv", held.groupby("task", as_index=False).agg(
        n_assays=("assay_id", "size"), n_valid_mmd=("mmd_squared", "count"),
        n_calibrated=("reference_percentile", "count"), median_mmd_squared=("mmd_squared", "median"),
        median_reference_percentile=("reference_percentile", "median"),
        median_centroid_distance=("centroid_distance", "median"), median_covariance_distance=("covariance_distance", "median")))
    run.finish(reference="each source assay vs source pool excluding its entire supercluster, same outer fold/task",
               kernel="biased MMD squared; mean of RBF kernels with squared-bandwidth multipliers 0.5,1,2,4",
               bandwidth="median positive squared pairwise distance of the two sampled sets, recomputed for every comparison",
               label_filtering="none; score values and label availability are not used", feature_scaling="none")


def composition_vectors(single):
    types = [a + ">" + b for a in "ACDEFGHIKLMNPQRSTVWY" for b in "ACDEFGHIKLMNPQRSTVWY" if a != b]
    vectors, coverage = {}, []
    for assay, group in single.groupby("assay_id", sort=True):
        length = group.protein_length.iloc[0]
        if (group.position > length).any():
            raise ValueError(f"Mutation outside protein_length in {assay}")
        counts = group.substitution.value_counts().reindex(types, fill_value=0).to_numpy(dtype=float)
        deciles = np.minimum(((group.position.to_numpy() - 1) * 10 / length).astype(int), 9)
        position_counts = np.bincount(deciles, minlength=10).astype(float)
        vectors[assay] = counts / counts.sum(), position_counts / position_counts.sum()
        coverage.append({"assay_id": assay, "task": group.task.iloc[0], "super_cluster": group.super_cluster.iloc[0],
                         "fold": int(group.fold.iloc[0]), "n_single": len(group), "n_positions": group.position.nunique(),
                         "site_coverage": group.position.nunique() / length,
                         "single_substitution_coverage": len(group) / (19 * length)})
    return vectors, pd.DataFrame(coverage)


def run_composition(args):
    frame = load_metadata(args.metadata)
    require_columns(frame, ["protein_length"])
    length = pd.to_numeric(frame.protein_length, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(length).all() or (length < 1).any() or (length != np.floor(length)).any():
        raise ValueError("protein_length must be a positive integer")
    frame["protein_length"] = length.astype(int)
    if frame.groupby("assay_id").protein_length.nunique().gt(1).any():
        raise ValueError("Inconsistent protein lengths within an assay")
    single = single_substitutions(frame)
    if single.empty:
        raise ValueError("No canonical nonsynonymous single substitutions")
    run = Run(args, "descriptive variant composition; no effect-label use", [args.metadata])
    vectors, coverage = composition_vectors(single)
    rows = []
    for _, row in coverage.iterrows():
        source = coverage[coverage.task.eq(row.task) & coverage.fold.ne(row.fold)]
        record = row.to_dict()
        record["n_source_assays"] = len(source)
        if source.empty:
            record.update(substitution_jsd=np.nan, position_decile_jsd=np.nan, unsupported_type_mass=np.nan)
        else:
            source_type = np.mean([vectors[a][0] for a in source.assay_id], axis=0)
            source_pos = np.mean([vectors[a][1] for a in source.assay_id], axis=0)
            target_type, target_pos = vectors[row.assay_id]
            record.update(substitution_jsd=js_divergence(source_type, target_type),
                          position_decile_jsd=js_divergence(source_pos, target_pos),
                          unsupported_type_mass=float(target_type[source_type == 0].sum()))
        rows.append(record)
    table = pd.DataFrame(rows)
    counts = frame.groupby(["assay_id", "task"], as_index=False).size().rename(columns={"size": "n_input"})
    counts = counts.merge(coverage[["assay_id", "n_single"]], on="assay_id", how="left")
    counts["n_single"] = counts.n_single.fillna(0).astype(int)
    counts["n_excluded_noncanonical_or_nonsingle"] = counts.n_input - counts.n_single
    run.table("coverage.csv", counts.merge(coverage.drop(columns=["task", "n_single"]), on="assay_id", how="left"))
    run.table("per_assay.csv", table)
    run.table("summary.csv", table.groupby("task", as_index=False).agg(
        n_assays=("assay_id", "size"), n_valid_distance=("substitution_jsd", "count"),
        mean_substitution_jsd=("substitution_jsd", "mean"), mean_position_decile_jsd=("position_decile_jsd", "mean"),
        mean_unsupported_type_mass=("unsupported_type_mass", "mean"), median_site_coverage=("site_coverage", "median")))
    run.finish(source_weighting="equal mass per source assay", divergence="base-2 Jensen-Shannon divergence, range [0,1]",
               population="only provided rows; no automatic cap or inferred raw library", n_excluded=len(frame) - len(single))


def normalized_sequence(value):
    return "".join(str(value).upper().split())


def run_context(args):
    frame = load_metadata(args.metadata)
    require_columns(frame, ["background_sequence", "mutated_sequence"])
    frame["background_sequence"] = frame.background_sequence.map(normalized_sequence)
    frame["mutated_sequence"] = frame.mutated_sequence.map(normalized_sequence)
    if frame.groupby("assay_id").background_sequence.nunique().gt(1).any():
        raise ValueError("Each assay must have one exact background sequence")
    single = single_substitutions(frame)
    for row in single.itertuples():
        background = row.background_sequence
        if row.position > len(background) or background[row.position - 1] != row.wt_aa:
            raise ValueError(f"Wild-type residue/position mismatch: {row.assay_id}, {row.mutant}")
        expected = background[:row.position - 1] + row.mut_aa + background[row.position:]
        if row.mutated_sequence != expected:
            raise ValueError(f"Full mutated sequence mismatch: {row.assay_id}, {row.mutant}")
    run = Run(args, "descriptive assay-context agreement; both assays' labels used; no predictor fit", [args.metadata])
    assay_info = frame.groupby("assay_id", sort=True).first()
    eligible, inventory = [], []
    for (a, info_a), (b, info_b) in combinations(assay_info.iterrows(), 2):
        same_background = info_a.background_sequence == info_b.background_sequence
        if not same_background and info_a.protein_id != info_b.protein_id:
            continue
        left = single[single.assay_id.eq(a)].set_index("mutant")
        right = single[single.assay_id.eq(b)].set_index("mutant")
        common = left.index.intersection(right.index).sort_values()
        n_positions = left.loc[common, "position"].nunique()
        reason = "eligible" if same_background and len(common) >= args.min_variants and n_positions >= args.min_positions else (
            "different_exact_background" if not same_background else "insufficient_shared_coverage")
        record = {"assay_a": a, "assay_b": b, "task_a": info_a.task, "task_b": info_b.task,
                  "n_shared_variants": len(common), "n_shared_positions": n_positions, "status": reason,
                  "background_sha256": hashlib.sha256(info_a.background_sequence.encode()).hexdigest()}
        inventory.append(record)
        if reason == "eligible":
            eligible.append((record, left.loc[common], right.loc[common]))
    columns = ["assay_a", "assay_b", "task_a", "task_b", "n_shared_variants", "n_shared_positions", "status", "background_sha256"]
    run.table("candidate_pairs.csv", pd.DataFrame(inventory, columns=columns))
    rows = []
    for record, left, right in eligible:
        y_a, y_b = scores(left), scores(right)
        pos = left.position.to_numpy()
        sites = np.unique(pos)
        groups = [np.flatnonzero(pos == site) for site in sites]
        rng = np.random.default_rng(stable_seed(args.seed, record["assay_a"], record["assay_b"]))
        draws = []
        for _ in range(args.bootstrap):
            indices = np.concatenate([groups[i] for i in rng.integers(len(sites), size=len(sites))])
            draws.append(rho(y_a[indices], y_b[indices]))
        finite = np.asarray(draws)[np.isfinite(draws)]
        # Do not silently condition an interval on nondegenerate resamples.
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) == args.bootstrap else (np.nan, np.nan)
        rows.append({**record, "rho": rho(y_a, y_b), "ci_low": low, "ci_high": high, "n_finite_bootstrap": len(finite)})
    run.table("pair_concordance.csv", pd.DataFrame(rows, columns=columns + ["rho", "ci_low", "ci_high", "n_finite_bootstrap"]))
    run.finish(n_eligible_pairs=len(eligible), n_excluded_nonsingle_rows=len(frame) - len(single),
               uncertainty="paired position-block percentile bootstrap within each assay pair, reranking every draw; no multiple-testing correction")
