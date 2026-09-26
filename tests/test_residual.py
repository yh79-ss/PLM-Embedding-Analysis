"""Independent small-array checks for prior-anchored residual controls."""
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from vep_diagnostics.residual import (PREDICTIONS, assay_ranks, choose_residual_setting,
                                      combine_prior, run_residual, select_residual)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


def fixture():
    rng = np.random.default_rng(381)
    rows, features = [], []
    for fold in (2, 5, 9):
        for assay_n in range(2):
            x = rng.normal(size=(9, 3)) * np.array([2., 0.7, 4.])
            prior = rng.normal(size=9)
            y = .7 * prior + x[:, 0] - .3 * x[:, 1] + rng.normal(size=9) * .8
            for i in range(9):
                rows.append({"assay_id": f"a_{fold}_{assay_n}", "protein_id": f"p_{fold}_{assay_n}",
                             "super_cluster": f"c_{fold}_{assay_n}", "task": "test", "fold": fold,
                             "mutant": f"A{i+1}C", "score": y[i], "baseline": prior[i]})
            features.append(x)
    return pd.DataFrame(rows), np.vstack(features)


def manual_ranks(frame, column):
    out = np.zeros(len(frame))
    for indices in frame.groupby("assay_id").indices.values():
        values = frame.iloc[indices][column].to_numpy(dtype=float)
        out[indices] = (rankdata(values, method="average") - 1) / (len(values) - 1)
    return out


def normal_equation(x_train, response, x_test, alpha):
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1
    standardized = (x_train - mean) / scale
    response_mean = response.mean()
    beta = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(x_train.shape[1]),
                           standardized.T @ (response - response_mean))
    return (x_test - mean) / scale @ beta + response_mean


def manual_rho(y, prediction):
    return np.corrcoef(rankdata(y), rankdata(prediction))[0, 1]


def write_fixture(tmp_path, frame, x):
    metadata, embeddings = tmp_path / "metadata.csv", tmp_path / "embeddings.npz"
    frame.to_csv(metadata, index=False)
    # Reverse NPZ order to require the independent keyed loader to align it.
    np.savez(embeddings, X=x[::-1], assay_id=frame.assay_id.to_numpy(dtype=str)[::-1],
             mutant=frame.mutant.to_numpy(dtype=str)[::-1])
    return metadata, embeddings


def args(metadata, embeddings, out):
    return Namespace(command="residual", metadata=str(metadata), embeddings=str(embeddings), out=str(out),
                     baseline_column="baseline", baseline_provenance="synthetic fixed predictor independent of effects",
                     confirm_no_target_labels=True, alphas=[.1, 5., 100.], lambdas=[0., .25, 1.], fixed_alpha=1.,
                     representation="synthetic 3-D", scope="synthetic audit", seed=4, bootstrap=25)


def test_formula_and_lambda_zero_identity():
    prior = np.array([0., .25, 1.])
    residual = np.array([.5, -.5, 1.])
    np.testing.assert_array_equal(combine_prior(prior, residual, .2), prior + .2 * residual)
    np.testing.assert_array_equal(combine_prior(prior, None, 0), prior)
    frame = pd.DataFrame({"assay_id": ["a", "a", "a", "b", "b", "b"],
                          "baseline": [9, 9, -1, 1, 8, 3]})
    np.testing.assert_array_equal(assay_ranks(frame, "baseline"), [.75, .75, 0., 0., 1., .5])


def test_true_maximum_ties_and_undefined_zero_alpha():
    table = pd.DataFrame({"alpha": [100., 10., 1.], "multiplier": [1., 1., 1.],
                          "mean_rho": [1 - 1.5e-12, 1 - .75e-12, 1.]})
    assert choose_residual_setting(table)["alpha"] == 10.
    tied = pd.DataFrame({"alpha": [10., 100., np.nan, 1.], "multiplier": [.5, .5, 0., 1.],
                         "mean_rho": [.2] * 4})
    selected = choose_residual_setting(tied)
    assert selected["multiplier"] == 0 and np.isnan(selected["alpha"])
    selected = choose_residual_setting(tied[tied.multiplier.gt(0)])
    assert selected["multiplier"] == .5 and selected["alpha"] == 100.
    with pytest.raises(ValueError, match="Undefined"):
        choose_residual_setting(table.assign(mean_rho=np.nan))


def test_source_selection_matches_independent_fold_reference():
    frame, x = fixture()
    alphas, multipliers = [.1, 5., 100.], [0., .25, 1.]
    joint, unit, ledger, candidate_scores = select_residual(frame, x, alphas, multipliers)
    reference = []
    for fold in (2, 5, 9):
        training, valid = np.flatnonzero(frame.fold.ne(fold)), np.flatnonzero(frame.fold.eq(fold))
        source, held = frame.iloc[training], frame.iloc[valid]
        response = manual_ranks(source, "score") - manual_ranks(source, "baseline")
        prior = manual_ranks(held, "baseline")
        for alpha in [None] + alphas:
            pred = None if alpha is None else normal_equation(x[training], response, x[valid], alpha)
            for multiplier in ([0.] if alpha is None else [.25, 1.]):
                combined = prior if alpha is None else prior + multiplier * pred
                for assay, indices in held.groupby("assay_id").indices.items():
                    reference.append({"alpha": np.nan if alpha is None else alpha, "multiplier": multiplier,
                                      "inner_fold": fold, "assay_id": assay,
                                      "rho": manual_rho(held.iloc[indices].score, combined[indices])})
    ref = pd.DataFrame(reference)
    columns = ["inner_fold", "alpha", "multiplier", "assay_id"]
    np.testing.assert_allclose(ledger.sort_values(columns).rho, ref.sort_values(columns).rho, atol=5e-15)
    means = ref.groupby(["alpha", "multiplier"], dropna=False, as_index=False).rho.mean()
    maximum = means.rho.max()
    winner = means[means.rho >= maximum - 1e-12].sort_values(["multiplier", "alpha"], ascending=[True, False]).iloc[0]
    assert joint["multiplier"] == winner.multiplier
    if winner.multiplier > 0:
        assert joint["alpha"] == winner.alpha
    units = means[means.multiplier.eq(1)]
    expected_unit = units[units.rho >= units.rho.max() - 1e-12].alpha.max()
    assert unit["alpha"] == expected_unit
    assert candidate_scores.n_assays.eq(frame.assay_id.nunique()).all()
    assert ledger[ledger.selected_joint].assay_id.nunique() == frame.assay_id.nunique()
    assert ledger[ledger.selected_unit].assay_id.nunique() == frame.assay_id.nunique()


def test_predictions_match_closed_form_and_target_labels_do_not_select(tmp_path):
    frame, x = fixture()
    metadata, embeddings = write_fixture(tmp_path, frame, x)
    settings = args(metadata, embeddings, tmp_path / "original")
    run_residual(settings)
    output = pd.read_csv(tmp_path / "original/predictions.csv")
    selected = pd.read_csv(tmp_path / "original/selected_hyperparameters.csv")
    for fold in (2, 5, 9):
        training, held = np.flatnonzero(frame.fold.ne(fold)), np.flatnonzero(frame.fold.eq(fold))
        source, target = frame.iloc[training], frame.iloc[held]
        response = manual_ranks(source, "score") - manual_ranks(source, "baseline")
        prior = manual_ranks(target, "baseline")
        actual, chosen = output[output.fold.eq(fold)], selected[selected.outer_fold.eq(fold)].iloc[0]
        np.testing.assert_allclose(actual.prior_prediction, prior, atol=1e-15)
        for col, alpha in [("fixed_prediction", 1.), ("unit_prediction", chosen.unit_alpha)]:
            expected = prior + normal_equation(x[training], response, x[held], alpha)
            np.testing.assert_allclose(actual[col], expected, atol=1e-13)
        expected = prior if chosen.selected_lambda == 0 else prior + chosen.selected_lambda * normal_equation(
            x[training], response, x[held], chosen.selected_alpha)
        np.testing.assert_allclose(actual.prediction, expected, atol=1e-13)
    changed = frame.copy()
    changed.loc[changed.fold.eq(2), "score"] *= -19
    changed.to_csv(tmp_path / "changed.csv", index=False)
    changed_args = args(tmp_path / "changed.csv", embeddings, tmp_path / "changed")
    run_residual(changed_args)
    after = pd.read_csv(tmp_path / "changed/predictions.csv")
    np.testing.assert_array_equal(output.loc[output.fold.eq(2), PREDICTIONS], after.loc[after.fold.eq(2), PREDICTIONS])
    before_selection = pd.read_csv(tmp_path / "original/source_selection.csv")
    after_selection = pd.read_csv(tmp_path / "changed/source_selection.csv")
    pd.testing.assert_frame_equal(before_selection[before_selection.outer_fold.eq(2)].reset_index(drop=True),
                                  after_selection[after_selection.outer_fold.eq(2)].reset_index(drop=True))
    manifest = json.loads((tmp_path / "original/manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "not an exact historical replay" in manifest["historical_equivalence"]
    assert len(pd.read_csv(tmp_path / "original/contrasts.csv")) == 6


def test_zero_residual_selects_prior_without_artificial_alpha(tmp_path):
    frame, x = fixture()
    x[:] = 0
    metadata, embeddings = write_fixture(tmp_path, frame, x)
    run_residual(args(metadata, embeddings, tmp_path / "zero"))
    selection = pd.read_csv(tmp_path / "zero/selected_hyperparameters.csv")
    assert selection.selected_lambda.eq(0).all()
    assert selection.selected_alpha.isna().all()
    assert not selection.joint_alpha_defined.any()
    assert not selection.joint_alpha_at_max.any()
    assert selection.unit_alpha.eq(100).all()
    output = pd.read_csv(tmp_path / "zero/predictions.csv")
    np.testing.assert_array_equal(output.prediction, output.prior_prediction)


def test_undefined_candidate_cannot_change_selection_population():
    frame, x = fixture()
    frame.loc[frame.assay_id.eq("a_2_0"), "baseline"] = 1
    with pytest.raises(ValueError, match="undefined Spearman"):
        select_residual(frame, x, [1., 10.], [0., 1.])
    with pytest.raises(ValueError, match="include 0 and 1"):
        select_residual(frame, x, [1., 10.], [.5, 1.])
    with pytest.raises(ValueError, match="two source folds"):
        select_residual(frame.iloc[:9], x[:9], [1., 10.], [0., 1.])


def test_provenance_guards(tmp_path):
    frame, x = fixture()
    metadata, embeddings = write_fixture(tmp_path, frame, x)
    settings = args(metadata, embeddings, tmp_path / "unused")
    settings.confirm_no_target_labels = False
    with pytest.raises(ValueError, match="confirm-no-target-labels"):
        run_residual(settings)
    settings.confirm_no_target_labels = True
    settings.baseline_provenance = " "
    with pytest.raises(ValueError, match="baseline-provenance"):
        run_residual(settings)
    settings.baseline_provenance = "a valid declaration"
    frame["label_protocol"] = "target-support"
    frame.to_csv(metadata, index=False)
    with pytest.raises(ValueError, match="relabelled"):
        run_residual(settings)
