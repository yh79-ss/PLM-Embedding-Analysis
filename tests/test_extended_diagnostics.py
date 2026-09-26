"""Synthetic checks for the slide/code-derived diagnostic inventory."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from vep_diagnostics.cli import main
from vep_diagnostics.data import load_embeddings, load_metadata
from vep_diagnostics.demo import make_demo
from vep_diagnostics.diagnose import cosine_references
from vep_diagnostics.evaluation import holm, pair_accuracy, permutation_correlations
from vep_diagnostics.metrics import moment_distances


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("extended")
    make_demo(root / "demo", 14)
    main(["probe", "--metadata", str(root / "demo/variants.csv"), "--embeddings", str(root / "demo/embeddings.npz"),
          "--representation", "synthetic 8D", "--scope", "synthetic", "--alphas", ".1", "1", "--out", str(root / "probe")])
    return root


def common(command, data, output, embedding=False):
    args = [command, "--metadata", str(data / "demo/variants.csv"), "--scope", "synthetic", "--out", str(output)]
    if embedding:
        args += ["--embeddings", str(data / "demo/embeddings.npz"), "--representation", "synthetic 8D"]
    return args


def evaluation_args(command, data, output):
    return common(command, data, output) + ["--predictions", str(data / "probe/predictions.csv"), "--label-protocol", "no-target-label"]


def test_permutation_matches_spearman_and_preserves_ties():
    y = np.array([0, 0, 1, 2, 3, 3.])
    p = np.column_stack([y, -y, np.ones(6)])
    actual = permutation_correlations(y, p, 5, 12)
    rng = np.random.default_rng(12)
    for row in actual:
        permuted = rng.permutation(y)
        assert row[0] == pytest.approx(spearmanr(permuted, y).statistic)
        assert row[1] == pytest.approx(-row[0])
        assert np.isnan(row[2])
    np.testing.assert_allclose(holm([.01, .04, .03, np.nan]), [.04, .09, .09, np.nan])


@pytest.mark.parametrize("dimension", [1, 4, 200])
def test_covariance_distance_matches_direct_matrix(dimension):
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(5, dimension)), rng.normal(size=(6, dimension))
    centroid, covariance = moment_distances(a, b)
    assert centroid == pytest.approx(np.linalg.norm(a.mean(0) - b.mean(0)) / np.sqrt(dimension))
    assert covariance == pytest.approx(np.linalg.norm(np.atleast_2d(np.cov(a, rowvar=False)) - np.atleast_2d(np.cov(b, rowvar=False)), "fro") / np.sqrt(dimension))
    assert moment_distances(a, a)[1] == pytest.approx(0, abs=1e-7)


def test_evaluate_exact_metrics_baseline_and_subgroups(data, tmp_path):
    args = evaluation_args("evaluate", data, tmp_path / "eval") + ["--columns", "prediction", "fixed_prediction",
            "--baseline-column", "baseline", "--group-by", "provenance", "--permutations", "20", "--bootstrap", "20"]
    main(args)
    per_assay = pd.read_csv(tmp_path / "eval/per_assay.csv")
    predictions = pd.read_csv(data / "probe/predictions.csv")
    for row in per_assay.itertuples():
        group = predictions[predictions.assay_id.eq(row.assay_id)]
        assert row.rho == pytest.approx(spearmanr(group.score, group[row.model]).statistic)
    summary = pd.read_csv(tmp_path / "eval/summary.csv")
    assert len(summary) == 6
    assert summary.n_valid.eq(5).all()
    assert (summary.permutation_p_holm >= summary.permutation_p_greater - 1e-14).all()
    contrasts = pd.read_csv(tmp_path / "eval/baseline_comparisons.csv")
    for row in contrasts.itertuples():
        base = summary[summary.task.eq(row.task) & summary.model.eq("baseline")].mean_rho.iloc[0]
        candidate = summary[summary.task.eq(row.task) & summary.model.eq(row.model)].mean_rho.iloc[0]
        assert row.delta == pytest.approx(candidate - base)
    assert len(pd.read_csv(tmp_path / "eval/subgroups.csv")) == 12


def test_oracle_recovery_seed_and_protocol_guards(data, tmp_path):
    frame = pd.read_csv(data / "probe/predictions.csv")
    frame["prediction_oracle"], frame["seed"] = frame.score, 0
    frame["label_protocol"] = "target-support diagnostic"
    pd.concat([frame, frame.assign(seed=1)]).to_csv(tmp_path / "predictions.csv", index=False)
    args = common("evaluate", data, tmp_path / "recovery") + ["--predictions", str(tmp_path / "predictions.csv"),
            "--label-protocol", "target-support", "--oracle-column", "prediction_oracle", "--permutations", "10", "--bootstrap", "10"]
    with pytest.raises(SystemExit):
        main(args)
    main(args + ["--prediction-seed", "0"])
    recovery = pd.read_csv(tmp_path / "recovery/recovery.csv")
    np.testing.assert_allclose(recovery.recovery, (recovery.rho_source - recovery.random_mean) / (1 - recovery.random_mean))
    args[args.index("--label-protocol") + 1] = "no-target-label"
    with pytest.raises(SystemExit):
        main(args + ["--prediction-seed", "0"])


def test_evaluate_rejects_population_and_protocol_mismatch(data, tmp_path):
    frame = pd.read_csv(data / "probe/predictions.csv").iloc[1:].copy()
    frame.to_csv(tmp_path / "subset.csv", index=False)
    args = evaluation_args("evaluate", data, tmp_path / "out") + ["--permutations", "3", "--bootstrap", "3"]
    args[args.index("--predictions") + 1] = str(tmp_path / "subset.csv")
    with pytest.raises(SystemExit):
        main(args)
    main(args + ["--allow-intersection"])
    assert len(pd.read_csv(tmp_path / "out/excluded_keys.csv")) == 1
    frame["label_protocol"] = "descriptive fold-local fit"
    frame.to_csv(tmp_path / "relabelled.csv", index=False)
    args[args.index("--predictions") + 1] = str(tmp_path / "relabelled.csv")
    with pytest.raises(SystemExit):
        main(args + ["--allow-intersection"])


def test_errors_bias_is_original_not_post_calibrated(data, tmp_path):
    main(evaluation_args("errors", data, tmp_path / "errors") + ["--percentile-predictions", "--max-pairs", "500"])
    table = pd.read_csv(tmp_path / "errors/per_assay.csv")
    prediction = pd.read_csv(data / "probe/predictions.csv")
    for row in table.itertuples():
        group = prediction[prediction.assay_id.eq(row.assay_id)]
        assert row.prediction_rank_bias == pytest.approx(.5 - group.prediction.mean())
    assert "residual_bias" not in table
    assert table.top_auroc.between(0, 1).all()
    assert set(pd.read_csv(tmp_path / "errors/error_strata.csv").stratum) == {"wt_aa_category", "effect_rank_quartile", "observed_position_quartile"}
    assert pair_accuracy(np.arange(5.), np.arange(5.), 100, 0)[0] == 1
    assert pair_accuracy(np.arange(5.), np.ones(5), 100, 0)[0] == .5
    assert pair_accuracy(np.arange(5.), -np.arange(5.), 100, 0)[0] == 0


def test_geometry_and_provenance_do_not_use_scores(data, tmp_path):
    main(common("geometry", data, tmp_path / "geo", True) + ["--max-per-assay", "10"])
    frame = pd.read_csv(data / "demo/variants.csv")
    frame["score"] = "not consulted"
    frame.to_csv(tmp_path / "changed.csv", index=False)
    args = common("geometry", data, tmp_path / "geo_changed", True) + ["--max-per-assay", "10"]
    args[args.index("--metadata") + 1] = str(tmp_path / "changed.csv")
    main(args)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "geo/coordinates.csv"), pd.read_csv(tmp_path / "geo_changed/coordinates.csv"))
    args = common("provenance", data, tmp_path / "provenance") + ["--group-by", "provenance", "platform"]
    args[args.index("--metadata") + 1] = str(tmp_path / "changed.csv")
    main(args)
    summary = pd.read_csv(tmp_path / "provenance/summary.csv")
    assert summary.loc[summary.group_column.eq("provenance"), "largest_assay_fraction"].eq(.6).all()
    assert summary.loc[summary.group_column.eq("platform"), "effective_group_count"].eq(1).all()


def test_associations_use_finite_matched_assays_not_pooled_tasks(data, tmp_path):
    main(common("shift", data, tmp_path / "shift", True) + ["--projection-dim", "0", "--max-samples", "20"])
    main(common("associate", data, tmp_path / "assoc") + ["--metrics", str(data / "probe/source_scores.csv"),
         "--covariates", str(tmp_path / "shift/held_out.csv"), "--x", "mmd_squared", "centroid_distance", "--y", "rho_source"])
    summary = pd.read_csv(tmp_path / "assoc/summary.csv")
    joined = pd.read_csv(tmp_path / "assoc/joined_assays.csv")
    for row in summary.itertuples():
        group = joined[joined.task.eq(row.task)]
        assert row.spearman == pytest.approx(spearmanr(group[row.x], group[row.y]).statistic)
    assert len(summary) == 4


def test_cosine_references_matched_transform_and_single_assay_guard():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(24, 3))
    y = x @ np.array([.2, .4, 1])
    group = pd.DataFrame({"assay_id": ["a"] * 12 + ["b"] * 12, "task": "t", "fold": 0})
    args = SimpleNamespace(reference_repeats=3, seed=9, alpha=1)
    rows = pd.DataFrame(cosine_references(group, x, y, args))
    assert len(rows) == 6 and rows.weight_cosine.between(-1, 1).all()
    np.testing.assert_allclose(rows.weight_cosine, pd.DataFrame(cosine_references(group, x, y, args)).weight_cosine)
    group["assay_id"] = "one"
    single = pd.DataFrame(cosine_references(group, x, y, args))
    assert single.loc[single.reference.eq("assay_bootstrap"), "weight_cosine"].isna().all()
    assert single.loc[single.reference.eq("shuffled_labels"), "weight_cosine"].notna().all()


def test_cross_task_predictions_never_fit_outer_labels(data, tmp_path):
    args = common("cross-task", data, tmp_path / "cross", True) + ["--alphas", ".1", "1", "--permutations", "5", "--bootstrap", "5"]
    main(args)
    frame = pd.read_csv(data / "demo/variants.csv")
    frame.loc[frame.fold.eq(0), "score"] *= -100
    frame.to_csv(tmp_path / "changed.csv", index=False)
    args[args.index("--metadata") + 1], args[args.index("--out") + 1] = str(tmp_path / "changed.csv"), str(tmp_path / "cross_changed")
    main(args)
    a, b = pd.read_csv(tmp_path / "cross/predictions.csv"), pd.read_csv(tmp_path / "cross_changed/predictions.csv")
    np.testing.assert_allclose(a.loc[a.fold.eq(0), ["prediction", "selected_alpha"]], b.loc[b.fold.eq(0), ["prediction", "selected_alpha"]], rtol=0, atol=0)
    summary = pd.read_csv(tmp_path / "cross/summary.csv")
    assert len(summary) == 4 and summary.n_valid_pairs.eq(5).all()
    assert summary.loc[summary.diagonal, "ranking_p_holm"].isna().all()


def test_auxiliary_features_are_key_aligned_and_manifested(data, tmp_path):
    args = common("probe", data, tmp_path / "aug", True) + ["--alphas", "1", "--aux-embeddings", str(data / "demo/embeddings.npz"), "--aux-representation", "duplicate block control"]
    main(args)
    manifest = json.loads((tmp_path / "aug/manifest.json").read_text())
    assert len(manifest["inputs"]) == 3 and "concatenation" in manifest["feature_blocks"]
    frame = load_metadata(data / "demo/variants.csv")
    x = load_embeddings(data / "demo/embeddings.npz", frame)
    np.savez_compressed(tmp_path / "bad.npz", X=x[:-1], assay_id=frame.assay_id.to_numpy(str)[:-1], mutant=frame.mutant.to_numpy(str)[:-1])
    args[args.index("--aux-embeddings") + 1] = str(tmp_path / "bad.npz")
    with pytest.raises(SystemExit):
        main(args)


def test_baseline_packaging_preserves_scores(data, tmp_path):
    main(common("baselines", data, tmp_path / "fixed") + ["--score-columns", "baseline"])
    fixed = pd.read_csv(tmp_path / "fixed/predictions.csv")
    original = pd.read_csv(data / "demo/variants.csv")
    np.testing.assert_allclose(fixed.baseline, original.baseline)
    assert "score" not in fixed and fixed.label_protocol.str.startswith("no-target-label").all()


def test_optional_blosum62(data, tmp_path):
    pytest.importorskip("Bio")
    main(common("baselines", data, tmp_path / "blosum") + ["--blosum62"])
    fixed = pd.read_csv(tmp_path / "blosum/predictions.csv")
    assert fixed.loc[fixed.mutant.eq("A1C"), "blosum62"].eq(0).all()
