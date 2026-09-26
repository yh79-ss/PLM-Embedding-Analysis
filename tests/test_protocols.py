"""Synthetic tests for protocol boundaries, identities and statistical estimands."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from vep_diagnostics.cli import main
from vep_diagnostics.data import (attach_split, load_embeddings, load_metadata, read_csv,
                                  sha256, single_substitutions, validate_metadata, verify_split)
from vep_diagnostics.demo import make_demo
from vep_diagnostics.metrics import js_divergence, mmd_squared, paired_cluster_interval
from vep_diagnostics.probes import choose_alpha, paired_support_split


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    path = tmp_path_factory.mktemp("fixture") / "data"
    make_demo(path, 17)
    return path


def run_args(command, demo, out):
    args = [command, "--metadata", str(demo / "variants.csv"), "--scope", "synthetic demo", "--out", str(out)]
    if command in ("probe", "support", "shift"):
        args += ["--embeddings", str(demo / "embeddings.npz"), "--representation", "synthetic 8-D"]
    return args


def test_frozen_split():
    path = Path(__file__).resolve().parents[1] / "splits/protein_fold_lookup_c08_v3_seed893.csv"
    split, summary = verify_split(path)
    assert sha256(path) == "fca42ef22d6c305982c64f3eb0032b68757251262357edb9fcf231d4c4ed5280"
    assert (summary["assays"], summary["proteins"], summary["superclusters"]) == (217, 186, 163)
    assert list(summary["assays_per_fold"].values()) == [43, 44, 44, 43, 43]
    metadata = pd.DataFrame({"assay_id": [split.DMS_id.iloc[0]], "mutant": ["A1C"], "task": ["custom"]})
    attached = attach_split(metadata, split)
    assert attached.protein_id.iloc[0] == split.UniProt_ID.iloc[0]
    with pytest.raises(ValueError, match="Unknown assay"):
        attach_split(metadata.assign(assay_id="unknown"), split)
    with pytest.raises(ValueError, match="disagrees"):
        attach_split(metadata.assign(protein_id="wrong"), split)


def test_keyed_embeddings_and_population_rejection(demo, tmp_path):
    frame = load_metadata(demo / "variants.csv")
    x = load_embeddings(demo / "embeddings.npz", frame)
    np.testing.assert_array_equal(x[0], x[1])  # Same synthetic variant, two tasks; archive was shuffled.
    with pytest.raises(ValueError, match="population mismatch"):
        load_embeddings(demo / "embeddings.npz", frame.iloc[:-1])
    bad = tmp_path / "bad.npz"
    np.savez(bad, X=x, assay_id=frame.assay_id.to_numpy(dtype=object), mutant=frame.mutant.to_numpy(dtype=str))
    with pytest.raises(ValueError):
        load_embeddings(bad, frame)


def test_fold_leakage_and_duplicates(demo):
    frame = load_metadata(demo / "variants.csv")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_metadata(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    bad = frame.copy()
    bad.loc[bad.assay_id.eq("synthetic_Activity_0"), "fold"] = 1
    with pytest.raises(ValueError, match="Leakage"):
        validate_metadata(bad)
    bad = frame.copy()
    bad.loc[bad.fold.eq(1), "super_cluster"] = "synthetic_cluster_0"
    with pytest.raises(ValueError, match="super_cluster spans"):
        validate_metadata(bad)


def test_selection_ties_use_true_maximum():
    assert choose_alpha([1 - 1.5e-12, 1 - 0.75e-12, 1], [1, 10, 100]) == 10
    assert choose_alpha([1, 1, 1], [1, 10, 100], prefer_larger=True) == 100
    with pytest.raises(ValueError, match="Undefined"):
        choose_alpha([np.nan, 1], [1, 10])


def test_position_support_identity_and_singleton_gate(demo):
    frame = single_substitutions(load_metadata(demo / "variants.csv"))
    group = frame[frame.assay_id.eq("synthetic_Activity_0")].copy()
    # Keep only one variant at a site to exercise the corrected replacement gate.
    group = group[~group.position.eq(1) | group.mut_aa.eq("C")]
    for seed in range(8):
        ev, arms, stats = paired_support_split(group, seed, 64)
        pos, types = group.position.to_numpy(), group.substitution.to_numpy()
        assert not set(ev) & set(arms["overlap"])
        assert not set(ev) & set(arms["disjoint"])
        assert not set(pos[ev]) & set(pos[arms["disjoint"]])
        assert sorted(types[arms["overlap"]]) == sorted(types[arms["disjoint"]])
        added = np.setdiff1d(arms["overlap"], arms["disjoint"])
        assert set(pos[added]) <= set(pos[ev])
        assert stats["n_support"] == 64


def test_cluster_interval_preserves_assay_macro():
    frame = pd.DataFrame({"super_cluster": ["A", "A", "B"], "delta": [0., 0., 1.]})
    result = paired_cluster_interval(frame, 100, 0)
    assert result["delta"] == pytest.approx(1 / 3)  # Not cluster-equal 1/2.
    assert result["n_clusters"] == 2
    assert np.isnan(paired_cluster_interval(frame.iloc[:2], 100, 0)["ci_low"])


def test_mmd_and_jsd():
    x = np.arange(12).reshape(6, 2)
    assert mmd_squared(x, x) == pytest.approx(0, abs=1e-12)
    assert mmd_squared(x, x + 100) > 0
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1)
    assert js_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)


def test_outer_target_labels_do_not_change_target_predictions(demo, tmp_path):
    out_a, out_b = tmp_path / "original", tmp_path / "changed"
    args = run_args("probe", demo, out_a) + ["--alphas", "0.1", "10", "--source-task", "Activity", "--target-task", "Binding"]
    main(args)
    frame = pd.read_csv(demo / "variants.csv")
    frame.loc[frame.fold.eq(0), "score"] *= -19
    changed = tmp_path / "changed_labels.csv"
    frame.to_csv(changed, index=False)
    args[args.index("--metadata") + 1] = str(changed)
    args[args.index("--out") + 1] = str(out_b)
    main(args)
    a, b = pd.read_csv(out_a / "predictions.csv"), pd.read_csv(out_b / "predictions.csv")
    a, b = a[a.fold.eq(0)], b[b.fold.eq(0)]
    np.testing.assert_allclose(a.prediction, b.prediction, atol=0, rtol=0)
    np.testing.assert_array_equal(a.selected_alpha, b.selected_alpha)
    assert json.loads((out_a / "manifest.json").read_text())["status"] == "complete"


def test_all_analysis_commands(demo, tmp_path):
    probe = tmp_path / "probe"
    main(run_args("probe", demo, probe) + ["--alphas", "0.1", "10"])
    main(run_args("shift", demo, tmp_path / "shift") + ["--max-samples", "16", "--projection-dim", "4"])
    main(run_args("composition", demo, tmp_path / "composition"))
    main(run_args("context", demo, tmp_path / "context") + ["--bootstrap", "20"])
    main(run_args("support", demo, tmp_path / "support") + ["--allow-target-support", "--support-size", "64", "--seeds", "0", "--alphas", "0.1", "10"])
    prediction = str(probe / "predictions.csv")
    main(run_args("compare", demo, tmp_path / "compare") + ["--predictions-a", prediction, "--predictions-b", prediction,
         "--column-b", "fixed_prediction", "--label-protocol", "no-target-label", "--bootstrap", "20"])
    for name in ("probe", "shift", "composition", "context", "support", "compare"):
        assert json.loads((tmp_path / name / "manifest.json").read_text())["status"] == "complete"
    assert len(pd.read_csv(tmp_path / "context/pair_concordance.csv")) == 5
    assert json.loads((tmp_path / "support/manifest.json").read_text())["n_eligible_assays"] > 0
    assert pd.read_csv(tmp_path / "shift/held_out.csv").reference_percentile.between(0, 100).all()
    with pytest.raises(SystemExit):
        main(run_args("probe", demo, probe))  # Existing output is never overwritten.


def test_descriptive_commands_ignore_scores(demo, tmp_path):
    frame = read_csv(demo / "variants.csv").drop(columns=["score", "baseline"])
    no_labels = tmp_path / "metadata.csv"
    frame.to_csv(no_labels, index=False)
    for command in ("composition", "shift"):
        args = run_args(command, demo, tmp_path / command)
        args[args.index("--metadata") + 1] = str(no_labels)
        if command == "shift":
            args += ["--max-samples", "8"]
        main(args)


def test_compare_rejects_relabelled_support(demo, tmp_path):
    frame = read_csv(demo / "variants.csv")
    frame["prediction"] = frame.score
    frame["label_protocol"] = "target-support diagnostic"
    path = tmp_path / "support.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(SystemExit):
        main(run_args("compare", demo, tmp_path / "compare") + ["--predictions-a", str(path), "--predictions-b", str(path),
             "--label-protocol", "no-target-label"])


def test_shift_single_fold_reports_missing_reference(demo, tmp_path):
    frame = read_csv(demo / "variants.csv")
    frame["fold"] = "0"
    path = tmp_path / "single_fold.csv"
    frame.to_csv(path, index=False)
    args = run_args("shift", demo, tmp_path / "shift")
    args[args.index("--metadata") + 1] = str(path)
    main(args + ["--max-samples", "8"])
    result = pd.read_csv(tmp_path / "shift/held_out.csv")
    assert result.status.eq("insufficient_rows").all()
    assert result.reference_percentile.isna().all()
