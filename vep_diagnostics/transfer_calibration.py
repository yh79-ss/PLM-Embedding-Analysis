"""Declared-family, fixed-prediction calibration of cross-task prior increments.

This is conditional diagnostic inference, not a model-refit interval or an exact
finite-sample simultaneous guarantee. All supplied run/direction hypotheses,
including undefined ones, count in the declared family.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import IDENTITY, KEY, Run, keys, read_csv, require_columns, scores, sha256, stable_seed, validate_metadata
from .evaluation import holm, permutation_correlations
from .metrics import permutation_pvalue, rho


def cluster_bounds(frame, repeats, seed, tail):
    """Assay-macro cluster percentile bounds with multiplicity and unequal sizes."""
    finite = frame[np.isfinite(frame.delta)].copy()
    clusters = finite.groupby("super_cluster").delta.agg(["sum", "count"])
    result = {"delta": float(finite.delta.mean()), "n_clusters": len(clusters),
              "ci_low": np.nan, "ci_high": np.nan, "increment_lower_bonferroni": np.nan}
    if len(clusters) < 2:
        return result
    totals, counts = clusters["sum"].to_numpy(), clusters["count"].to_numpy()
    rng = np.random.default_rng(seed)
    draw = np.empty(repeats)
    for i in range(repeats):
        chosen = rng.integers(len(clusters), size=len(clusters))
        draw[i] = totals[chosen].sum() / counts[chosen].sum()
    low, high, adjusted_low = np.quantile(draw, [.025, .975, tail])
    return {**result, "ci_low": float(low), "ci_high": float(high),
            "increment_lower_bonferroni": float(adjusted_low)}


def key_hash(frame):
    records = sorted(map(tuple, frame[KEY].astype(str).to_numpy()))
    return hashlib.sha256(json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _numeric_column(frame, column):
    # Empty output cells represent undefined metrics, not identifiers.
    return pd.to_numeric(frame[column].replace("", np.nan), errors="raise").to_numpy(float)


def _load_run(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("arguments", {}).get("command") != "cross-task":
        raise ValueError(f"Expected a completed cross-task run: {directory}")
    protocol = manifest.get("label_protocol", "")
    if not protocol.startswith("no-target-label"):
        raise ValueError(f"Cross-task calibration requires no-target-label predictions: {directory}")
    files = [directory / name for name in ("predictions.csv", "per_assay.csv", "summary.csv")]
    for path in files:
        if not path.is_file() or manifest.get("outputs", {}).get(path.name) != sha256(path):
            raise ValueError(f"Missing or changed cross-task output: {path}")
    prediction, saved_assays, saved_summary = map(read_csv, files)
    require_columns(prediction, KEY + IDENTITY[1:] + ["source_task", "score", "prediction", "baseline", "analysis_scope", "label_protocol"])
    require_columns(saved_assays, IDENTITY + ["source_task", "n_variants", "status"])
    require_columns(saved_summary, ["source_task", "target_task"])
    expected_scope = str(manifest["arguments"].get("scope", ""))
    for table in (prediction, saved_assays, saved_summary):
        require_columns(table, ["analysis_scope", "label_protocol"])
        if not table.analysis_scope.eq(expected_scope).all() or not table.label_protocol.eq(protocol).all():
            raise ValueError(f"Table scope/protocol differs from its manifest: {directory}")
    if prediction.duplicated(["source_task", *KEY]).any():
        raise ValueError("Repeated prediction keys within a source task")
    for col in ("score", "prediction", "baseline"):
        prediction[col] = scores(prediction, col)
    tasks = sorted(prediction.task.unique())
    if len(tasks) < 2 or set(prediction.source_task) != set(tasks):
        raise ValueError("A complete task matrix must include every source and target task")
    population = None
    for _, group in prediction.groupby("source_task", sort=True):
        checked = validate_metadata(group.drop(columns="source_task"))
        checked = checked.sort_values(KEY).reset_index(drop=True)
        if population is None:
            population = checked
        else:
            columns = list(dict.fromkeys(KEY + IDENTITY)) + ["score", "baseline"]
            if not population[columns].equals(checked[columns]):
                raise ValueError("Source directions do not share identical evaluation populations/labels/prior")
    rows = []
    for (source, assay), group in prediction.groupby(["source_task", "assay_id"], sort=True):
        candidate, baseline = rho(group.score, group.prediction), rho(group.score, group.baseline)
        rows.append({**{c: group[c].iloc[0] for c in IDENTITY}, "source_task": source,
                     "n_variants": len(group), "rho_source": candidate, "rho_baseline": baseline,
                     "delta": candidate - baseline,
                     "status": "ok" if np.isfinite([candidate, baseline]).all() else "undefined_pair"})
    assays = pd.DataFrame(rows)
    index = ["source_task", "assay_id"]
    if saved_assays.duplicated(index).any() or set(map(tuple, saved_assays[index].to_numpy())) != set(map(tuple, assays[index].to_numpy())):
        raise ValueError("Saved per-assay population differs from predictions")
    saved = saved_assays.set_index(index).loc[pd.MultiIndex.from_frame(assays[index])].reset_index()
    for col in IDENTITY + ["n_variants", "status"]:
        if not assays[col].astype(str).eq(saved[col].astype(str)).all():
            raise ValueError(f"Saved per-assay identity/count/status mismatch: {col}")
    for col in ("rho_source", "rho_baseline", "delta"):
        if col not in saved_assays:
            raise ValueError(f"Saved per-assay table is missing {col}")
        if not np.allclose(assays[col], _numeric_column(saved, col), rtol=1e-12, atol=1e-12, equal_nan=True):
            raise ValueError(f"Saved per-assay metric does not match predictions: {col}")
    expected = {(a, b) for a in tasks for b in tasks}
    if saved_summary.duplicated(["source_task", "target_task"]).any() or set(map(tuple, saved_summary[["source_task", "target_task"]].to_numpy())) != expected:
        raise ValueError("Saved summary is not the complete task matrix")
    return {"path": directory, "manifest": manifest, "prediction": prediction, "assays": assays,
            "population": population, "tasks": tasks, "inputs": [manifest_path, *files]}


def _check_populations(runs, allow_different):
    reference = runs[0]["population"]
    reference_keys = keys(reference)
    differences = []
    for number, run in enumerate(runs):
        population = run["population"]
        current_keys = keys(population)
        common = reference_keys.intersection(current_keys, sort=False)
        different = len(common) != len(reference_keys) or len(common) != len(current_keys)
        scopes_differ = run["manifest"]["arguments"].get("scope") != runs[0]["manifest"]["arguments"].get("scope")
        if run["tasks"] != runs[0]["tasks"]:
            raise ValueError("Every run must declare the same task labels and complete directions")
        if (different or scopes_differ) and not allow_different:
            raise ValueError("Run populations/scopes differ; use --allow-different-populations to retain separate explicit populations")
        # A key may appear only in later runs. Checking against run zero alone
        # would miss contradictions between those later, explicitly different
        # populations. Verify every pairwise overlap before any aggregation.
        for previous in runs[:number]:
            other = previous["population"]
            other_keys = keys(other)
            overlap = other_keys.intersection(current_keys, sort=False)
            left = other.iloc[other_keys.get_indexer(overlap)].reset_index(drop=True)
            right = population.iloc[current_keys.get_indexer(overlap)].reset_index(drop=True)
            for col in IDENTITY[1:]:
                if not left[col].astype(str).eq(right[col].astype(str)).all():
                    raise ValueError(f"Shared evaluation key has inconsistent {col}")
            for col in ("score", "baseline"):
                if not np.array_equal(left[col].to_numpy(), right[col].to_numpy()):
                    raise ValueError(f"Shared evaluation keys have different {col}; a joint family requires the same labels and fixed prior")
        for side, index in (("reference_only", reference_keys.difference(current_keys)),
                            ("run_only", current_keys.difference(reference_keys))):
            differences.extend({"run_id": f"run_{number:04d}", "difference": side, "assay_id": a, "mutant": m}
                               for a, m in index)
    return pd.DataFrame(differences, columns=["run_id", "difference", *KEY])


def run_calibrate_transfer(args):
    if not args.family_id.strip() or not np.isfinite(args.rho_threshold) or not -1 <= args.rho_threshold <= 1:
        raise ValueError("Declare a nonempty family ID and a finite prespecified rho threshold in [-1, 1]")
    if not 0 < args.family_alpha < 1 or min(args.bootstrap, args.permutations) < 1:
        raise ValueError("Family alpha must be in (0, 1); bootstrap/permutation counts must be positive")
    directories = [Path(p).resolve() for p in args.runs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Supply distinct, completed run directories")
    runs = [_load_run(path) for path in directories]
    differences = _check_populations(runs, args.allow_different_populations)
    family_size = sum(len(run["tasks"]) * (len(run["tasks"]) - 1) for run in runs)
    tail = args.family_alpha / family_size
    if args.bootstrap * tail < 1:
        raise ValueError("Too few bootstrap replicates to resolve the Bonferroni tail; increase --bootstrap")
    run = Run(args, "no-target-label task-transfer; conditional evaluation-only joint-family calibration",
              [p for item in runs for p in item["inputs"]])
    result, assay_tables, provenance = [], [], []
    for number, item in enumerate(runs):
        run_id = f"run_{number:04d}"
        representation = item["manifest"]["arguments"].get("representation", "unspecified")
        assay_tables.append(item["assays"].assign(run_id=run_id, representation=representation))
        provenance.append({"run_id": run_id, "representation": representation, "run_path": str(item["path"]),
                           "input_scope": item["manifest"]["arguments"].get("scope", ""),
                           "input_protocol": item["manifest"]["label_protocol"],
                           "input_version": item["manifest"].get("version", "unknown"),
                           "manifest_sha256": sha256(item["inputs"][0]),
                           "evaluation_key_sha256": key_hash(item["population"]),
                           "n_evaluation_rows": len(item["population"])})
        for (source, target), group in item["assays"].groupby(["source_task", "task"], sort=True):
            if source == target:
                continue
            paired = group[group.status.eq("ok")]
            null = np.zeros(args.permutations)
            for assay in paired.assay_id:
                predictions = item["prediction"]
                rows = predictions[predictions.source_task.eq(source) & predictions.assay_id.eq(assay)].sort_values(KEY)
                # Common null draws across representations/directions on matched
                # populations; Holm remains valid without independence.
                draw = permutation_correlations(rows.score.to_numpy(float), rows[["prediction"]].to_numpy(float),
                                                args.permutations, stable_seed(args.seed, assay, "joint-transfer-null"))[:, 0]
                null += draw
            observed = float(paired.rho_source.mean())
            null = null / len(paired) if len(paired) else np.full(args.permutations, np.nan)
            interval = cluster_bounds(paired, args.bootstrap, stable_seed(args.seed, run_id, source, target, "increment"), tail)
            result.append({"run_id": run_id, "representation": representation, "source_task": source,
                           "target_task": target, "family_id": args.family_id, "family_size": family_size,
                           "family_alpha": args.family_alpha, "rho_threshold": args.rho_threshold,
                           "n_assays": len(group), "n_valid_pairs": len(paired), "rho_source": observed,
                           "rho_baseline": float(paired.rho_baseline.mean()), **interval,
                           "ranking_p": permutation_pvalue(null, observed), "bonferroni_tail": tail,
                           "bootstrap_expected_tail_draws": args.bootstrap * tail})
    calibrated = pd.DataFrame(result)
    calibrated["ranking_p_holm"] = holm(calibrated.ranking_p)
    calibrated["ranking_signal"] = calibrated.ranking_p_holm.le(args.family_alpha)
    calibrated["rho_condition"] = calibrated.rho_source.ge(args.rho_threshold)
    calibrated["nonnegative_point_increment"] = calibrated.delta.ge(0)
    calibrated["nonnegative_adjusted_lower"] = calibrated.increment_lower_bonferroni.ge(0)
    calibrated["reliable_increment"] = calibrated[["ranking_signal", "rho_condition", "nonnegative_point_increment", "nonnegative_adjusted_lower"]].all(axis=1)
    calibrated["status"] = np.where(np.isfinite(calibrated.increment_lower_bonferroni) & np.isfinite(calibrated.ranking_p),
                                    "defined", "undefined_or_insufficient_clusters")
    run.table("calibration.csv", calibrated)
    run.table("per_assay.csv", pd.concat(assay_tables, ignore_index=True))
    run.table("input_runs.csv", pd.DataFrame(provenance))
    run.table("population_differences.csv", differences)
    run.finish(family_id=args.family_id, family_size=family_size,
               family_definition="All off-diagonal directions in every supplied complete run, including undefined hypotheses; no data-dependent omission",
               threshold_prespecification="User declaration; the software cannot certify that family/threshold were chosen before inspecting outcomes",
               ranking="Recomputed tie-inclusive within-assay permutation p-values, Holm adjusted jointly over the declared family",
               increment="One-sided Bonferroni nominal percentile-bootstrap lower bound at family_alpha/family_size; ordinary two-sided 95% intervals kept separate",
               reliable_increment="rho >= prespecified threshold AND point delta >= 0 AND adjusted lower >= 0 AND Holm ranking p <= family_alpha",
               uncertainty="Conditional on saved fitted predictions, realized splits and finite paired assay population; supercluster resampling with multiplicity; no refits; approximate coverage, not an exact finite-sample FWER guarantee",
               population_handling="Separate per-run populations retained; never silently intersected or pooled",
               bootstrap_expected_tail_draws=args.bootstrap * tail,
               low_bootstrap_tail_resolution=bool(args.bootstrap * tail < 10),
               declared_protocol_not_independently_certified=True)
