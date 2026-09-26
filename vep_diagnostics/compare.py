"""Exact-key paired evaluation of two previously generated prediction tables."""
import numpy as np
import pandas as pd

from .data import IDENTITY, KEY, Run, keys, load_metadata, read_csv, require_columns, scores, stable_seed
from .metrics import paired_cluster_interval, rho


def read_prediction(path, column, metadata):
    frame = read_csv(path)
    require_columns(frame, KEY + [column])
    if frame.duplicated(KEY).any():
        raise ValueError("Prediction keys are not unique; select one seed and one model before comparison")
    frame[column] = scores(frame, column)
    # When provided, identities must agree with the evaluation metadata.
    for col in IDENTITY[1:]:
        if col in frame:
            joined = frame[KEY + [col]].merge(metadata[KEY + [col]], on=KEY, suffixes=("_p", "_m"))
            if not joined[col + "_p"].astype(str).eq(joined[col + "_m"].astype(str)).all():
                raise ValueError(f"Prediction identity disagrees with metadata: {col}")
    return frame


def run_compare(args):
    metadata = load_metadata(args.metadata)
    metadata["score"] = scores(metadata)
    a = read_prediction(args.predictions_a, args.column_a, metadata)
    b = read_prediction(args.predictions_b, args.column_b, metadata)
    if args.label_protocol == "no-target-label":
        for table in (a, b):
            if "label_protocol" in table and not table.label_protocol.str.startswith("no-target-label").all():
                raise ValueError("Cannot relabel supplied diagnostic/adaptation predictions as no-target-label")
    km, ka, kb = keys(metadata), keys(a), keys(b)
    common = km.intersection(ka).intersection(kb)
    if not args.allow_intersection and (len(common) != len(km) or len(common) != len(ka) or len(common) != len(kb)):
        raise ValueError("Populations differ; align beforehand or explicitly use --allow-intersection and report reduced scope")
    if len(common) == 0:
        raise ValueError("No common variant keys")
    run = Run(args, args.label_protocol, [args.metadata, args.predictions_a, args.predictions_b])
    exclusions = []
    for name, idx in (("metadata", km), ("predictions_a", ka), ("predictions_b", kb)):
        for assay, mutant in idx.difference(common):
            exclusions.append({"input": name, "assay_id": assay, "mutant": mutant, "reason": "outside_three_way_intersection"})
    run.table("excluded_keys.csv", pd.DataFrame(exclusions, columns=["input", *KEY, "reason"]))
    matched = metadata.iloc[km.get_indexer(common)].copy()
    matched["a"] = a.iloc[ka.get_indexer(common)][args.column_a].to_numpy()
    matched["b"] = b.iloc[kb.get_indexer(common)][args.column_b].to_numpy()
    rows = []
    for assay, group in matched.groupby("assay_id", sort=True):
        ra, rb = rho(group.score, group.a), rho(group.score, group.b)
        rows.append({"assay_id": assay, "task": group.task.iloc[0], "super_cluster": group.super_cluster.iloc[0],
                     "n_variants": len(group), "rho_a": ra, "rho_b": rb, "delta": ra - rb,
                     "status": "ok" if np.isfinite(ra) and np.isfinite(rb) else "undefined_constant_or_too_small"})
    table = pd.DataFrame(rows)
    summaries = []
    for task, group in table.groupby("task", sort=True):
        valid = group[group.status.eq("ok")]
        interval = paired_cluster_interval(valid, args.bootstrap, stable_seed(args.seed, task))
        summaries.append({"task": task, "n_assays": len(group), "n_valid_pairs": len(valid),
                          "mean_rho_a": valid.rho_a.mean(), "mean_rho_b": valid.rho_b.mean(), **interval})
    run.table("per_assay.csv", table)
    run.table("summary.csv", pd.DataFrame(summaries))
    run.finish(contrast="A minus B, paired assay-macro Spearman on identical variants",
               uncertainty="95% percentile cluster bootstrap conditional on fixed predictions; no refitting or multiplicity correction",
               declared_protocol_not_independently_certified=True)
