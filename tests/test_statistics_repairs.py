"""Adversarial acceptance checks for C1--C4 and joint transfer calibration."""
from collections import namedtuple
from itertools import permutations
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from vep_diagnostics import metrics
from vep_diagnostics.cross_task import task_matrix
from vep_diagnostics.data import IDENTITY, Run, sha256, stable_seed
from vep_diagnostics.evaluation import RESERVED_GROUP_COLUMNS, holm, permutation_correlations, prediction_frame, run_evaluate
from vep_diagnostics.metrics import permutation_pvalue, rho
from vep_diagnostics.transfer_calibration import cluster_bounds, run_calibrate_transfer


@pytest.mark.parametrize("y,p", [([0, 1, 2], [0, 1, 2]), ([0, 1, 2], [2, 1, 0]),
                                  ([0, 0, 2], [0, 1, 1]), ([0, 1, 2, 3], [0, 1, 3, 2])])
def test_exact_integer_rank_permutation_space_includes_ties(y, p):
    yr, pr = 2 * rankdata(y), 2 * rankdata(p)
    # Centered twice-ranks are integers for these fixtures, allowing an exact
    # tail-count reference independent of floating correlation normalization.
    yr, pr = yr - yr.mean(), pr - pr.mean()
    numerators = np.array([np.asarray(order) @ pr for order in permutations(yr)])
    expected_count = np.count_nonzero(numerators >= yr @ pr)
    null = numerators / (np.linalg.norm(yr) * np.linalg.norm(pr))
    assert permutation_pvalue(null, rho(y, p)) == (1 + expected_count) / (len(null) + 1)


def test_tail_near_zero_negative_nan_and_real_differences():
    eps = np.finfo(float).eps
    assert permutation_pvalue(np.array([1 - eps, 1 - 1e-10, -.5]), 1) == .5
    assert permutation_pvalue(np.array([-.5 - eps, -.5 - 1e-10, -.25]), -.5) == .75
    assert permutation_pvalue(np.array([-eps, -1e-10, .5]), 0) == .75
    assert np.isnan(permutation_pvalue([np.nan], 1))
    assert np.isnan(permutation_pvalue([], 1))
    with pytest.raises(ValueError, match="one-dimensional"):
        permutation_pvalue(np.ones((2, 2)), 1)


def test_scipy_19_namedtuple_result_compatibility(monkeypatch):
    legacy = namedtuple("SpearmanrResult", "correlation pvalue")
    monkeypatch.setattr(metrics, "spearmanr", lambda *_: legacy(.75, .01))
    assert rho([0, 1, 2], [0, 1, 2]) == .75


def test_evaluation_assay_macro_and_holm_use_tie_inclusive_tails(tmp_path):
    data = pd.DataFrame([{"assay_id": "a", "mutant": f"A{i + 1}C", "protein_id": "p", "task": "T",
                          "super_cluster": "c", "fold": 0, "score": i, "prediction": i} for i in range(3)])
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    args = SimpleNamespace(metadata=path, predictions=path, out=tmp_path / "output", scope="synthetic",
                           label_protocol="no-target-label", prediction_seed=None, allow_intersection=False,
                           columns=["prediction"], group_by=[], baseline_column=None, oracle_column=None,
                           permutations=10000, bootstrap=10, seed=0)
    run_evaluate(args)
    null = permutation_correlations(data.score.to_numpy(), data[["prediction"]].to_numpy(), 10000,
                                    stable_seed(0, "a", "evaluation-null"))[:, 0]
    expected = permutation_pvalue(null, 1)
    assert .14 < expected < .19
    for name in ("per_assay.csv", "summary.csv"):
        output = pd.read_csv(args.out / name)
        assert output.permutation_p_greater.iloc[0] == pytest.approx(expected)
    assert output.permutation_p_holm.iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("column", sorted(RESERVED_GROUP_COLUMNS) + ["prediction"])
def test_generated_fields_cannot_be_group_annotations(column):
    args = SimpleNamespace(group_by=[column], columns=["prediction"])
    with pytest.raises(ValueError, match="Reserved subgroup"):
        prediction_frame(args)  # Rejection precedes file access, including API calls.


def test_all_task_labels_round_trip_through_safe_matrix_columns():
    labels = ["analysis_scope", "label_protocol", "source_task", "target_0000", "NA"]
    summary = pd.DataFrame([{"source_task": s, "target_task": t, "value": i * 10 + j}
                            for i, s in enumerate(labels) for j, t in enumerate(labels)])
    matrix, mapping = task_matrix(summary, "value")
    assert not set(matrix.columns) & {"analysis_scope", "label_protocol"}
    for row in summary.itertuples():
        column = mapping.loc[mapping.target_task.eq(row.target_task), "matrix_column"].item()
        assert matrix.loc[matrix.source_task.eq(row.source_task), column].item() == row.value


def make_saved_run(tmp_path, name, undefined=False, offset=0, scope="synthetic", below_prior=False):
    args = SimpleNamespace(command="cross-task", out=tmp_path / name, scope=scope, representation=name)
    run = Run(args, "no-target-label task-transfer; synthetic fixture", [])
    records = []
    for source in ("A", "B"):
        for task in ("A", "B"):
            for assay_number in range(3):
                for i in range(4 if below_prior else 3):
                    records.append({"source_task": source, "assay_id": f"{task}{assay_number + offset}",
                                    "mutant": f"A{i + 1}C", "protein_id": f"p{task}{assay_number + offset}",
                                    "super_cluster": f"c{task}{assay_number + offset}", "task": task,
                                    "fold": (assay_number + offset) % 3, "score": i,
                                    "prediction": (0 if undefined and source == "A" and task == "B" else
                                                   [0, 1, 3, 2][i] if below_prior else i),
                                    "baseline": i if below_prior else -i})
    prediction = pd.DataFrame(records)
    assays = []
    for (source, _), group in prediction.groupby(["source_task", "assay_id"], sort=True):
        value, base = rho(group.score, group.prediction), rho(group.score, group.baseline)
        assays.append({**{c: group[c].iloc[0] for c in IDENTITY}, "source_task": source,
                       "n_variants": len(group), "rho_source": value, "rho_baseline": base,
                       "delta": value - base, "status": "ok" if np.isfinite(value) else "undefined_pair"})
    summary = pd.DataFrame([{"source_task": s, "target_task": t} for s in ("A", "B") for t in ("A", "B")])
    run.table("predictions.csv", prediction)
    run.table("per_assay.csv", pd.DataFrame(assays))
    run.table("summary.csv", summary)
    run.finish()
    return args.out


def calibration_args(tmp_path, runs):
    return SimpleNamespace(runs=runs, family_id="prespecified synthetic two-interfaces family", rho_threshold=.1,
                           family_alpha=.05, bootstrap=400, permutations=2000, seed=0,
                           scope="synthetic joint family", out=tmp_path / "calibration", allow_different_populations=False)


def test_joint_family_preserves_undefined_members_and_separates_signal_increment(tmp_path):
    runs = [make_saved_run(tmp_path, "one"), make_saved_run(tmp_path, "two", undefined=True)]
    args = calibration_args(tmp_path, runs)
    run_calibrate_transfer(args)
    result = pd.read_csv(args.out / "calibration.csv")
    assert len(result) == 4
    assert result.family_size.eq(4).all()
    np.testing.assert_allclose(result.ranking_p_holm, holm(result.ranking_p), equal_nan=True)
    undefined = result[result.run_id.eq("run_0001") & result.source_task.eq("A")].iloc[0]
    assert not undefined.reliable_increment and not undefined.ranking_signal
    assert np.isnan(undefined.ranking_p) and np.isnan(undefined.increment_lower_bonferroni)
    valid = result[result.n_valid_pairs.eq(3)]
    assert valid.increment_lower_bonferroni.eq(2).all()
    assert valid.ranking_signal.all() and valid.reliable_increment.all()
    manifest = json.loads((args.out / "manifest.json").read_text())
    assert manifest["family_size"] == 4 and manifest["low_bootstrap_tail_resolution"]
    assert "no refits" in manifest["uncertainty"]


def test_cluster_bound_matches_literal_resampling_unequal_cluster_sizes():
    frame = pd.DataFrame({"super_cluster": ["a", "a", "b", "c"], "delta": [.1, -.3, .5, .8]})
    rng = np.random.default_rng(91)
    groups = [group.delta.to_numpy() for _, group in frame.groupby("super_cluster")]
    expected = [np.concatenate([groups[i] for i in rng.integers(3, size=3)]).mean() for _ in range(400)]
    actual = cluster_bounds(frame, 400, 91, .0125)
    np.testing.assert_allclose([actual["ci_low"], actual["ci_high"], actual["increment_lower_bonferroni"]],
                               np.quantile(expected, [.025, .975, .0125]))
    assert actual["delta"] == pytest.approx(frame.delta.mean())
    assert np.isnan(cluster_bounds(frame.assign(super_cluster="one"), 400, 91, .0125)["increment_lower_bonferroni"])


def test_ranking_signal_is_not_reliable_prior_increment(tmp_path):
    args = calibration_args(tmp_path, [make_saved_run(tmp_path, "inferior", below_prior=True)])
    run_calibrate_transfer(args)
    result = pd.read_csv(args.out / "calibration.csv")
    assert result.ranking_signal.all()
    assert not result.reliable_increment.any()
    assert not result.nonnegative_point_increment.any()
    assert not result.nonnegative_adjusted_lower.any()
    np.testing.assert_allclose(result.delta, -.2)


def test_joint_family_rejects_tampering_and_duplicate_runs(tmp_path):
    directory = make_saved_run(tmp_path, "one")
    with pytest.raises(ValueError, match="distinct"):
        run_calibrate_transfer(calibration_args(tmp_path, [directory, directory]))
    with (directory / "predictions.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="changed"):
        run_calibrate_transfer(calibration_args(tmp_path, [directory]))


def test_population_differences_require_explicit_declaration_and_remain_separate(tmp_path):
    directories = [make_saved_run(tmp_path, "one"), make_saved_run(tmp_path, "two", offset=1)]
    args = calibration_args(tmp_path, directories)
    with pytest.raises(ValueError, match="populations/scopes differ"):
        run_calibrate_transfer(args)
    args.allow_different_populations = True
    run_calibrate_transfer(args)
    assert len(pd.read_csv(args.out / "population_differences.csv")) == 12
    result = pd.read_csv(args.out / "calibration.csv")
    assert result.n_assays.eq(3).all()
    assert pd.read_csv(args.out / "input_runs.csv").evaluation_key_sha256.nunique() == 2


def test_adjusted_tail_resolution_guard(tmp_path):
    args = calibration_args(tmp_path, [make_saved_run(tmp_path, "one")])
    args.bootstrap = 10
    with pytest.raises(ValueError, match="Too few bootstrap"):
        run_calibrate_transfer(args)


@pytest.mark.parametrize("column", ["score", "baseline", "fold"])
def test_conflict_between_later_populations_is_not_hidden_by_disjoint_first_run(tmp_path, column):
    directories = [make_saved_run(tmp_path, "zero", offset=0), make_saved_run(tmp_path, "one", offset=3),
                   make_saved_run(tmp_path, "two", offset=3)]
    changed = directories[2] / "predictions.csv"
    prediction = pd.read_csv(changed)
    prediction[column] += 3  # Additive offsets preserve every Spearman metric.
    prediction.to_csv(changed, index=False)
    manifest_path = directories[2] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["predictions.csv"] = sha256(changed)
    if column == "fold":
        saved_path = directories[2] / "per_assay.csv"
        saved = pd.read_csv(saved_path)
        saved.fold += 3
        saved.to_csv(saved_path, index=False)
        manifest["outputs"]["per_assay.csv"] = sha256(saved_path)
    manifest_path.write_text(json.dumps(manifest))
    args = calibration_args(tmp_path, directories)
    args.allow_different_populations = True
    with pytest.raises(ValueError, match="Shared evaluation"):
        run_calibrate_transfer(args)
