"""Frozen-oracle, complete-seed and exact comparator alignment regressions."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vep_diagnostics.data import load_metadata, stable_seed
from vep_diagnostics.diagnose import oracle_split, run_diagnose
from vep_diagnostics.paired import (attach_comparator, checked_predictions,
                                    paired_seed_summary, run_paired_seeds)
from vep_diagnostics.probes import run_support


def fixture_data(tmp_path, position_support=False):
    rng = np.random.default_rng(901)
    rows, features = [], []
    variants = ([f"A{position}{aa}" for position in range(1, 43) for aa in "CDEFGH"]
                if position_support else [f"A{position}C" for position in range(1, 25)])
    for assay in range(3):
        x = rng.normal(size=(len(variants), 4)) * [1, 4, .2, 2]
        y = x @ np.array([.6, -.3, .5, .2]) + rng.normal(size=len(x)) * .5
        for i, mutant in enumerate(variants):
            rows.append(dict(assay_id=f"a{assay}", mutant=mutant, protein_id=f"p{assay}", task="test",
                             super_cluster=f"c{assay}", fold=assay, score=y[i], prior=x[i, 0]))
        features.extend(x)
    frame = pd.DataFrame(rows)
    metadata, embeddings = tmp_path / "metadata.csv", tmp_path / "embeddings.npz"
    frame.to_csv(metadata, index=False)
    np.savez(embeddings, X=np.asarray(features), assay_id=frame.assay_id.to_numpy(dtype=str),
             mutant=frame.mutant.to_numpy(dtype=str))
    return metadata, embeddings


def args_for(metadata, embeddings, out, **overrides):
    return SimpleNamespace(**dict(command="diagnose", metadata=str(metadata), embeddings=str(embeddings),
                                  out=str(out), scope="synthetic diagnostics", representation="synthetic4D",
                                  seed=0, seeds=[0, 1], fixed_alpha=1., oracle_alpha=1., support_fraction=.5,
                                  source_selection="nested", alphas=[.01, .1], bootstrap=50,
                                  allow_target_support=True, **overrides))


def test_oracle_frozen_across_source_grids_and_fixed_control(tmp_path):
    metadata, embeddings = fixture_data(tmp_path)
    first = args_for(metadata, embeddings, tmp_path / "first")
    second = args_for(metadata, embeddings, tmp_path / "second")
    second.alphas, second.fixed_alpha = [10000., 1000000.], 10.
    run_diagnose(first)
    run_diagnose(second)
    a, b = [pd.read_csv(tmp_path / name / "predictions.csv") for name in ("first", "second")]
    pd.testing.assert_frame_equal(a[["assay_id", "mutant", "seed", "prediction_oracle"]],
                                  b[["assay_id", "mutant", "seed", "prediction_oracle"]], check_exact=True)
    assert not np.allclose(a.prediction_source, b.prediction_source)
    metrics = pd.read_csv(tmp_path / "first/per_seed.csv")
    np.testing.assert_allclose(metrics.source_improvement, metrics.gap_reduction, atol=5e-16)
    assert metrics.oracle_alpha.eq(1).all()
    summary = pd.read_csv(tmp_path / "first/paired_summary.csv").set_index("contrast")
    assert set(summary.index) == {"source_improvement", "gap_reduction", "oracle_gap", "oracle_gap_fixed"}
    assert summary.loc["source_improvement", "delta"] == pytest.approx(summary.loc["gap_reduction", "delta"])
    assert summary.n_valid_paired_assays.eq(3).all()


def test_source_and_oracle_do_not_use_evaluation_labels(tmp_path):
    metadata, embeddings = fixture_data(tmp_path)
    first = args_for(metadata, embeddings, tmp_path / "original")
    first.seeds = [0]
    run_diagnose(first)
    frame = load_metadata(metadata)
    frame["score"] = frame.score.astype(float)
    group = frame[frame.fold.eq(0)]
    _, ev = oracle_split(len(group), .5, stable_seed(0, "a0", "random-row-oracle"))
    frame.loc[group.iloc[ev].index, "score"] *= -100
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False)
    second = args_for(changed, embeddings, tmp_path / "perturbed")
    second.seeds = [0]
    run_diagnose(second)
    a, b = [pd.read_csv(tmp_path / name / "predictions.csv") for name in ("original", "perturbed")]
    columns = ["prediction_source", "prediction_source_fixed", "prediction_oracle"]
    np.testing.assert_array_equal(a.loc[a.fold.eq(0), columns], b.loc[b.fold.eq(0), columns])


def metric_rows():
    rows = []
    for i, assay in enumerate("abcd"):
        for seed in (0, 1):
            rows.append(dict(assay_id=assay, protein_id=assay, task="T", super_cluster="A" if i < 3 else "B",
                             fold=0 if i < 3 else 1, seed=seed, left=(i + seed) / 6, right=-i / 7))
    return pd.DataFrame(rows)


def test_seed_averaging_precedes_cluster_resampling_with_multiplicity():
    metrics = metric_rows()
    per_assay, summary = paired_seed_summary(metrics, {"test": ("left", "right")}, [0, 1], 100, 77)
    expected = metrics.assign(delta=metrics.left - metrics.right).groupby("assay_id").delta.mean()
    np.testing.assert_allclose(per_assay.set_index("assay_id").delta, expected)
    rng = np.random.default_rng(stable_seed(77, "T", "paired-seeds"))
    clusters = [per_assay[per_assay.super_cluster.eq(cluster)].delta.to_numpy() for cluster in ("A", "B")]
    draws = [np.concatenate([clusters[index] for index in rng.integers(2, size=2)]).mean() for _ in range(100)]
    np.testing.assert_allclose(summary[["ci_low", "ci_high"]].iloc[0], np.quantile(draws, [.025, .975]))
    assert summary.delta.iloc[0] == pytest.approx(expected.mean())
    assert summary.n_valid_paired_assays.iloc[0] == 4


def test_incomplete_and_undefined_pairs_never_change_seed_population():
    frame = metric_rows()
    frame = frame[~(frame.assay_id.eq("b") & frame.seed.eq(1))].copy()
    frame.loc[frame.assay_id.eq("c") & frame.seed.eq(1), "right"] = np.nan
    per_assay, summary = paired_seed_summary(frame, {"test": ("left", "right")}, [0, 1], 20, 0)
    indexed = per_assay.set_index("assay_id")
    assert indexed.loc[["b", "c"], "delta"].isna().all()
    assert indexed.loc[["b", "c"], "left_mean"].isna().all()
    assert summary.n_valid_paired_assays.iloc[0] == 2
    with pytest.raises(ValueError, match="Duplicate"):
        paired_seed_summary(pd.concat([frame, frame.iloc[:1]]), {"test": ("left", "right")}, [0, 1], 20, 0)
    with pytest.raises(ValueError, match="unique seeds"):
        paired_seed_summary(frame, {"test": ("left", "right")}, [0, 0], 20, 0)
    with pytest.raises(ValueError, match="undeclared"):
        paired_seed_summary(frame, {"test": ("left", "right")}, [0], 20, 0)


def test_related_contrasts_can_require_common_three_model_population():
    frame = metric_rows().assign(oracle=.9)
    frame.loc[frame.assay_id.eq("a") & frame.seed.eq(1), "oracle"] = np.nan
    paired, _ = paired_seed_summary(frame, {"gain": ("left", "right"), "gap": ("oracle", "left")}, [0, 1], 20, 0,
                                    required_columns=["left", "right", "oracle"])
    assert paired.loc[paired.assay_id.eq("a"), "delta"].isna().all()
    assert paired.loc[~paired.assay_id.eq("a"), "status"].eq("ok").all()


def test_comparator_alignment_and_identity_protocol_guards(tmp_path):
    metadata, _ = fixture_data(tmp_path)
    frame = load_metadata(metadata)
    prediction = frame.copy().assign(prediction=np.arange(len(frame)), label_protocol="no-target-label source-only")
    path = tmp_path / "source.csv"
    prediction.sample(frac=1, random_state=3).to_csv(path, index=False)
    checked = checked_predictions(path, frame, ["prediction"], [0, 1], "no-target-label")
    evaluation = frame.iloc[[7, 1, 12]].copy().assign(seed=1)
    aligned = attach_comparator(evaluation, checked, "prediction")
    np.testing.assert_array_equal(aligned.prediction_source, [7, 1, 12])
    seeded = pd.concat([prediction.assign(seed=seed, prediction=prediction.prediction + 100 * seed) for seed in (0, 1)])
    seeded.sample(frac=1, random_state=4).to_csv(path, index=False)
    checked = checked_predictions(path, frame, ["prediction"], [0, 1], "no-target-label")
    np.testing.assert_array_equal(attach_comparator(evaluation, checked, "prediction").prediction_source, [107, 101, 112])
    with pytest.raises(ValueError, match="missing evaluation"):
        attach_comparator(evaluation, checked[checked.seed.eq(0)], "prediction")
    prediction.assign(label_protocol="target-support").to_csv(path, index=False)
    with pytest.raises(ValueError, match="Cannot relabel"):
        checked_predictions(path, frame, ["prediction"], [0, 1], "no-target-label")
    prediction.assign(protein_id="incorrect").to_csv(path, index=False)
    with pytest.raises(ValueError, match="protein_id disagrees"):
        checked_predictions(path, frame, ["prediction"], [0, 1], "no-target-label")


def test_generic_seed_command_records_missing_assay_seed(tmp_path):
    metadata, _ = fixture_data(tmp_path)
    frame = load_metadata(metadata)
    predictions = pd.concat([frame.assign(seed=seed, left=frame.score.astype(float), right=-frame.score.astype(float))
                             for seed in (0, 1)], ignore_index=True)
    predictions = predictions[~(predictions.assay_id.eq("a1") & predictions.seed.eq(1))]
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    args = SimpleNamespace(metadata=str(metadata), predictions=str(path), left="left", right="right", seeds=[0, 1],
                           label_protocol="target-support", allow_evaluation_subset=True, bootstrap=30, seed=0,
                           command="paired-seeds", out=str(tmp_path / "paired"), scope="synthetic")
    run_paired_seeds(args)
    per_assay = pd.read_csv(tmp_path / "paired/per_assay.csv")
    assert np.isnan(per_assay.loc[per_assay.assay_id.eq("a1"), "delta"].iloc[0])
    assert per_assay.loc[per_assay.assay_id.ne("a1"), "delta"].eq(2).all()
    args.out, args.allow_evaluation_subset = str(tmp_path / "rejected"), False
    with pytest.raises(ValueError, match="allow-evaluation-subset"):
        run_paired_seeds(args)


def test_support_controls_use_exact_evaluation_rows_and_seed_pairing(tmp_path):
    metadata, embeddings = fixture_data(tmp_path, position_support=True)
    frame = load_metadata(metadata)
    source = frame.copy().assign(prediction=np.arange(len(frame)), label_protocol="no-target-label")
    path = tmp_path / "source.csv"
    source.sample(frac=1, random_state=13).to_csv(path, index=False)
    args = args_for(metadata, embeddings, tmp_path / "support")
    args.command, args.support_size, args.alphas = "support", 48, [.1, 10]
    args.source_predictions, args.source_column, args.source_label_protocol = str(path), "prediction", "no-target-label"
    args.baseline_column = "prior"
    run_support(args)
    result = pd.read_csv(tmp_path / "support/predictions.csv")
    # An inner merge in pandas 1.5 can group nonconsecutive repeated keys across
    # seeds. Keep the evaluation order explicit instead of comparing against a
    # join whose order varies across supported pandas versions.
    keys = ["assay_id", "mutant"]
    ordered_keys = pd.MultiIndex.from_frame(result[keys])
    expected = source.set_index(keys)[["prediction", "prior"]].reindex(ordered_keys)
    pd.testing.assert_index_equal(expected.index, ordered_keys)
    assert result.duplicated(keys).any()  # The fixture exercises repeated seed keys.
    # Independent tuple-key reference: no pandas join or indexer assumptions.
    lookup = {(assay, mutant): (float(prediction), float(prior))
              for assay, mutant, prediction, prior in source[keys + ["prediction", "prior"]].itertuples(index=False, name=None)}
    reference = np.asarray([lookup[key] for key in result[keys].itertuples(index=False, name=None)])
    np.testing.assert_allclose(expected.to_numpy(dtype=float), reference)
    np.testing.assert_allclose(result.prediction_source, expected.prediction)
    np.testing.assert_allclose(result.prediction_prior, expected.prior.astype(float))
    paired = pd.read_csv(tmp_path / "support/paired_summary.csv")
    assert len(paired) == 12
    assert paired.n_valid_paired_assays.eq(3).all()
    assert paired.n_declared_seeds.eq(2).all()


def test_paired_cli_rejects_generated_count_column_collision(tmp_path):
    from vep_diagnostics.cli import main
    metadata, _ = fixture_data(tmp_path)
    frame = load_metadata(metadata)
    prediction = frame.assign(seed=0, n_evaluation=np.arange(len(frame)), other=np.arange(len(frame)))
    path = tmp_path / "collision.csv"
    prediction.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        main(["paired-seeds", "--metadata", str(metadata), "--predictions", str(path),
              "--left", "n_evaluation", "--right", "other", "--seeds", "0", "--label-protocol", "descriptive",
              "--scope", "synthetic", "--out", str(tmp_path / "rejected")])
    assert not (tmp_path / "rejected").exists()
