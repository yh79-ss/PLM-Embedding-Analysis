"""Source-selected task-transfer matrix with distinct signal and baseline checks."""
import numpy as np
import pandas as pd

from .data import IDENTITY, KEY, Run, load_embeddings, load_metadata, scores, stable_seed
from .evaluation import holm, permutation_correlations
from .metrics import paired_cluster_interval, permutation_pvalue, rho
from .probes import fit_probe, rank_assays, select_source_alpha


def task_matrix(summary, value):
    """Encode arbitrary task labels so neither identity nor provenance collides."""
    tasks = sorted(summary.target_task.unique())
    mapping = pd.DataFrame({"matrix_column": [f"target_{i:04d}" for i in range(len(tasks))],
                            "target_task": tasks})
    encoded = dict(zip(mapping.target_task, mapping.matrix_column))
    matrix = summary.pivot(index="source_task", columns="target_task", values=value)
    return matrix.rename(columns=encoded).reset_index(), mapping


def run_cross_task(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    frame["score"] = scores(frame)
    baseline = scores(frame, args.baseline_column)
    tasks = sorted(frame.task.unique())
    if len(tasks) < 2:
        raise ValueError("cross-task requires at least two task labels")
    run = Run(args, "no-target-label task-transfer; source-only inner selection; entire target outer fold excluded", [args.metadata, args.embeddings])
    predictions, selections = [], []
    for source_task in tasks:
        for fold in sorted(frame.fold.unique()):
            source_idx = np.flatnonzero(frame.task.eq(source_task) & frame.fold.ne(fold))
            target_idx = np.flatnonzero(frame.fold.eq(fold))
            source = frame.iloc[source_idx].copy().reset_index(drop=True)
            if len(source) < 3:
                raise ValueError(f"Insufficient source rows: {source_task}, fold {fold}")
            alpha, selection = select_source_alpha(source, x[source_idx], args.alphas)
            selection["source_task"], selection["outer_fold"] = source_task, fold
            selections.append(selection)
            model = fit_probe(x[source_idx], rank_assays(source), alpha)
            result = frame.iloc[target_idx][list(dict.fromkeys(KEY + IDENTITY)) + ["score"]].copy()
            result["prediction"], result["baseline"] = model.predict(x[target_idx]), baseline[target_idx]
            result["source_task"], result["selected_alpha"] = source_task, alpha
            predictions.append(result)
            print(f"cross-task: source {source_task}, excluded fold {fold}, alpha {alpha:g}", flush=True)
    prediction = pd.concat(predictions, ignore_index=True)
    per_assay, nulls = [], {}
    for (source_task, assay), group in prediction.groupby(["source_task", "assay_id"], sort=True):
        value, reference = rho(group.score, group.prediction), rho(group.score, group.baseline)
        nulls[source_task, assay] = permutation_correlations(group.score.to_numpy(float), group[["prediction"]].to_numpy(float),
                                                            args.permutations, stable_seed(args.seed, source_task, assay, "task-null"))[:, 0]
        per_assay.append({**{c: group[c].iloc[0] for c in IDENTITY}, "source_task": source_task, "n_variants": len(group),
                          "rho_source": value, "rho_baseline": reference, "delta": value - reference,
                          "status": "ok" if np.isfinite(value) and np.isfinite(reference) else "undefined_pair"})
    per_assay = pd.DataFrame(per_assay)
    summaries = []
    for (source_task, target_task), group in per_assay.groupby(["source_task", "task"], sort=True):
        paired = group[group.status.eq("ok")]
        observed = paired.rho_source.mean()
        null = np.mean([nulls[source_task, a] for a in paired.assay_id], axis=0) if len(paired) else np.full(args.permutations, np.nan)
        interval = paired_cluster_interval(paired, args.bootstrap, stable_seed(args.seed, source_task, target_task, "delta"))
        summaries.append({"source_task": source_task, "target_task": target_task, "diagonal": source_task == target_task,
                          "n_assays": len(group), "n_valid_pairs": len(paired), "rho_source": observed,
                          "rho_baseline": paired.rho_baseline.mean(), **interval,
                          "permutation_p_greater": permutation_pvalue(null, observed) if len(paired) else np.nan})
    summary = pd.DataFrame(summaries)
    summary["ranking_p_holm"] = np.nan
    off_diagonal = ~summary.diagonal
    summary.loc[off_diagonal, "ranking_p_holm"] = holm(summary.loc[off_diagonal, "permutation_p_greater"])
    run.table("predictions.csv", prediction)
    run.table("source_selection.csv", pd.concat(selections, ignore_index=True))
    run.table("per_assay.csv", per_assay)
    run.table("summary.csv", summary)
    rho_matrix, mapping = task_matrix(summary, "rho_source")
    increment_matrix, _ = task_matrix(summary, "delta")
    run.table("rho_matrix.csv", rho_matrix)
    run.table("increment_matrix.csv", increment_matrix)
    run.table("matrix_columns.csv", mapping)
    run.finish(selection="separate inner source-fold selection for each source-task/outer-fold fit; same assay-rank convention as probe",
               null="within-assay permutation of evaluation effects with fixed predictions; no source-label refitting",
               multiplicity="Holm only for off-diagonal ranking-signal tests in this one representation/run",
               uncertainty="95% conditional supercluster intervals for prior increments; NOT simultaneous/multiplicity-adjusted",
               matrix_columns="target_XXXX columns are mapped to original task labels in matrix_columns.csv",
               limitation="ranking significance is not incremental value; use calibrate-transfer for a declared joint family; no causal task-mapping claim")
