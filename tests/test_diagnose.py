"""Matched evaluation and coefficient-space regression tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from vep_diagnostics.cli import main
from vep_diagnostics.data import load_metadata, stable_seed
from vep_diagnostics.demo import make_demo
from vep_diagnostics.diagnose import matched_summary, oracle_split, weight_cosine
from vep_diagnostics.probes import fit_probe


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("diagnose-data") / "demo"
    make_demo(path, 31)
    return path


def arguments(command, data, out):
    args = [command, "--metadata", str(data / "variants.csv"), "--embeddings", str(data / "embeddings.npz"),
            "--representation", "synthetic 8D", "--scope", "synthetic diagnostic", "--out", str(out)]
    if command == "diagnose":
        args += ["--allow-target-support", "--seeds", "0", "--alphas", "0.1", "10"]
    return args


def test_original_coordinate_weights_reconstruct_predictions():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, 4)) * [0.01, 20, 3, 1] + [12, -5, 7, 0]
    x[:, 3] = 1  # Constant feature gets sklearn's unit scale.
    model = fit_probe(x[:60], x[:60] @ np.array([2, -1, 0.3, 1]), 1)
    np.testing.assert_allclose(model.predict(x[60:]), x[60:] @ model.raw_coef + model.raw_intercept, rtol=1e-12, atol=1e-12)
    assert weight_cosine(model.raw_coef, model.raw_coef) == pytest.approx(1)
    assert weight_cosine(model.raw_coef, -model.raw_coef) == pytest.approx(-1)
    assert np.isnan(weight_cosine(np.zeros(4), model.raw_coef))
    assert np.isnan(weight_cosine(np.ones(3), np.ones(4)))


@pytest.mark.parametrize("selection", ["fixed", "nested"])
def test_matched_predictions_ignore_evaluation_labels(data, tmp_path, selection):
    a, b = tmp_path / "original", tmp_path / "changed"
    args = arguments("diagnose", data, a) + ["--source-selection", selection]
    main(args)
    frame = load_metadata(data / "variants.csv")
    frame["score"] = frame.score.astype(float)
    for assay, group in frame[frame.fold.eq(0)].groupby("assay_id"):
        support, ev = oracle_split(len(group), 0.5, stable_seed(0, assay, "random-row-oracle"))
        assert len(set(support) & set(ev)) == 0
        assert len(support) + len(ev) == len(group)
        frame.loc[group.iloc[ev].index, "score"] *= -100
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False)
    args[args.index("--metadata") + 1] = str(changed)
    args[args.index("--out") + 1] = str(b)
    main(args)
    pa, pb = pd.read_csv(a / "predictions.csv"), pd.read_csv(b / "predictions.csv")
    np.testing.assert_allclose(pa.loc[pa.fold.eq(0), ["prediction_source", "prediction_oracle"]],
                               pb.loc[pb.fold.eq(0), ["prediction_source", "prediction_oracle"]], atol=0, rtol=0)
    ma, mb = pd.read_csv(a / "per_seed.csv"), pd.read_csv(b / "per_seed.csv")
    np.testing.assert_allclose(ma.loc[ma.fold.eq(0), ["alpha", "weight_cosine_source_oracle"]],
                               mb.loc[mb.fold.eq(0), ["alpha", "weight_cosine_source_oracle"]], atol=0, rtol=0)
    for (assay, seed), group in pa.groupby(["assay_id", "seed"]):
        metric = ma[ma.assay_id.eq(assay) & ma.seed.eq(seed)].iloc[0]
        assert metric.rho_source == pytest.approx(spearmanr(group.score, group.prediction_source)[0])
        assert metric.rho_oracle == pytest.approx(spearmanr(group.score, group.prediction_oracle)[0])
        assert metric.oracle_gap == pytest.approx(metric.rho_oracle - metric.rho_source)
    summary = pd.read_csv(a / "summary.csv")
    np.testing.assert_allclose(summary.oracle_gap, summary.rho_oracle - summary.rho_source, atol=1e-14)


def test_summary_does_not_mix_incomplete_seed_populations():
    rows = []
    for assay in ("a", "b"):
        for seed in (0, 1):
            good = assay == "a" or seed == 0
            rows.append(dict(assay_id=assay, protein_id=assay, task="task", super_cluster=assay, fold=0, seed=seed,
                             rho_source=0.9 if assay == "a" else -0.9, rho_oracle=0.6 if good else np.nan,
                             oracle_gap=-0.3 if assay == "a" else (1.5 if good else np.nan),
                             weight_cosine_source_oracle=0.2, status="ok" if good else "undefined_correlation"))
    per_assay, summary = matched_summary(pd.DataFrame(rows), [0, 1])
    assert per_assay.loc[per_assay.assay_id.eq("b"), "rho_source"].isna().all()
    assert summary.n_valid_paired_assays.iloc[0] == 1
    assert summary.rho_source.iloc[0] == pytest.approx(0.9)
    assert summary.oracle_gap.iloc[0] == pytest.approx(-0.3)


def test_fold_local_fits_are_separate_from_other_folds(data, tmp_path):
    a, b = tmp_path / "fold-local", tmp_path / "changed"
    args = arguments("cosine", data, a)
    main(args)
    frame = pd.read_csv(data / "variants.csv")
    frame.loc[frame.fold.eq(0), "score"] *= -1
    changed = tmp_path / "changed.csv"
    frame.to_csv(changed, index=False)
    args[args.index("--metadata") + 1], args[args.index("--out") + 1] = str(changed), str(b)
    main(args)
    with np.load(a / "probe_weights.npz") as wa, np.load(b / "probe_weights.npz") as wb:
        np.testing.assert_allclose(wa["weights"][wa["fold"] != 0], wb["weights"][wb["fold"] != 0], rtol=0, atol=0)
        np.testing.assert_allclose(wa["weights"][wa["fold"] == 0], -wb["weights"][wb["fold"] == 0], atol=1e-12)
        pairwise = pd.read_csv(a / "pairwise_cosine.csv")
        for row in pairwise.itertuples():
            w_a = wa["weights"][(wa["task"] == row.task) & (wa["fold"] == row.fold_a)][0]
            w_b = wa["weights"][(wa["task"] == row.task) & (wa["fold"] == row.fold_b)][0]
            assert row.weight_cosine == pytest.approx(weight_cosine(w_a, w_b))
    assert len(pairwise) == 20  # Ten fold pairs in each of two tasks.
    assert pairwise.label_protocol.str.contains("descriptive").all()


def test_oracle_requires_explicit_label_access(data, tmp_path):
    args = arguments("diagnose", data, tmp_path / "forbidden")
    args.remove("--allow-target-support")
    with pytest.raises(SystemExit):
        main(args)
    assert not (tmp_path / "forbidden").exists()
