"""Command line interface; each run requires an explicit population scope."""
import argparse
import json

import numpy as np
from threadpoolctl import threadpool_limits

from .data import Run, attach_split, json_safe, load_embeddings, load_metadata, read_csv, verify_split


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Create a small synthetic dataset and keyed embeddings")
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int, default=17)
    check = sub.add_parser("check-split", help="Validate a ProteinGym-format protein/cluster split")
    check.add_argument("--split", required=True)
    attach = sub.add_parser("attach-split", help="Attach frozen folds by exact assay ID")
    attach.add_argument("--metadata", required=True)
    attach.add_argument("--split", required=True)
    attach.add_argument("--out", required=True)
    validate = sub.add_parser("validate", help="Validate metadata, split and optional keyed embeddings")
    validate.add_argument("--metadata", required=True)
    validate.add_argument("--embeddings")
    for command, help_text in (
        ("probe", "Nested source-only Ridge and fixed-alpha control"),
        ("support", "Explicit target-support position-overlap/disjoint diagnostic"),
        ("shift", "Embedding MMD with source-assay calibration"),
        ("composition", "Label-free single-substitution library audit"),
        ("context", "Descriptive exact-background assay agreement"),
        ("compare", "Paired saved-prediction comparison with cluster intervals"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--metadata", required=True)
        p.add_argument("--out", required=True, help="New directory; existing paths are refused")
        p.add_argument("--scope", required=True, help="Population name, including Binding subset when relevant")
        p.add_argument("--seed", type=int, default=0)
        if command in ("probe", "support", "shift"):
            p.add_argument("--embeddings", required=True)
            p.add_argument("--representation", required=True, help="Model, layer, pooling and mutant/WT/delta convention")
        if command in ("probe", "support"):
            p.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
            p.add_argument("--fixed-alpha", type=float, default=1.0)
        if command == "probe":
            p.add_argument("--source-task")
            p.add_argument("--target-task")
            p.add_argument("--shuffle-source-labels", action="store_true")
        if command == "support":
            p.add_argument("--allow-target-support", action="store_true", required=True,
                           help="Explicitly declare that target-support labels may be fitted")
            p.add_argument("--support-size", type=positive_int, default=512)
            p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        if command == "shift":
            p.add_argument("--projection-dim", type=int, default=128, help="Fixed Gaussian projection dimension; 0 uses full features")
            p.add_argument("--max-samples", type=positive_int, default=1000)
        if command in ("context", "compare"):
            p.add_argument("--bootstrap", type=positive_int, default=2000)
        if command == "context":
            p.add_argument("--min-variants", type=positive_int, default=100)
            p.add_argument("--min-positions", type=positive_int, default=20)
        if command == "compare":
            p.add_argument("--predictions-a", required=True)
            p.add_argument("--predictions-b", required=True)
            p.add_argument("--column-a", default="prediction")
            p.add_argument("--column-b", default="prediction")
            p.add_argument("--allow-intersection", action="store_true")
            p.add_argument("--label-protocol", choices=["no-target-label", "target-support", "few-shot", "descriptive"], required=True)
    plot = sub.add_parser("plot", help="Plot a saved diagnostic summary (requires matplotlib)")
    plot.add_argument("--summary", required=True)
    plot.add_argument("--output", required=True)
    return root


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if hasattr(args, "alphas"):
        values = [*args.alphas, args.fixed_alpha]
        if not all(np.isfinite(a) and a > 0 for a in values) or len(set(args.alphas)) != len(args.alphas):
            p.error("Alphas must be unique, finite and strictly positive")
        args.alphas = sorted(args.alphas)
    if hasattr(args, "projection_dim") and args.projection_dim < 0:
        p.error("--projection-dim cannot be negative")
    if hasattr(args, "seeds") and len(args.seeds) != len(set(args.seeds)):
        p.error("--seeds must be unique")
    if hasattr(args, "seed") and args.seed < 0:
        p.error("--seed must be nonnegative")
    if hasattr(args, "scope") and not args.scope.strip():
        p.error("--scope must be nonempty")
    try:
        with threadpool_limits(limits=1):
            if args.command == "demo":
                from .demo import make_demo
                make_demo(args.out, args.seed)
            elif args.command == "check-split":
                _, summary = verify_split(args.split)
                print(json.dumps(json_safe(summary), indent=2))
            elif args.command == "attach-split":
                split, summary = verify_split(args.split)
                frame = attach_split(read_csv(args.metadata), split)
                run = Run(args, "metadata split attachment; no effect-label use", [args.metadata, args.split])
                # This is an input table, so avoid adding output-specific scope columns.
                frame.to_csv(run.path / "variants.csv", index=False)
                run.finish(split_summary=summary)
            elif args.command == "validate":
                frame = load_metadata(args.metadata)
                info = {"n_variants": len(frame), "n_assays": frame.assay_id.nunique(),
                        "n_proteins": frame.protein_id.nunique(), "n_superclusters": frame.super_cluster.nunique(),
                        "folds": sorted(frame.fold.unique().tolist()), "tasks": sorted(frame.task.unique())}
                if args.embeddings:
                    x = load_embeddings(args.embeddings, frame)
                    info["embedding_shape"] = list(x.shape)
                print(json.dumps(info, indent=2))
            elif args.command in ("probe", "support"):
                from .probes import run_probe, run_support
                (run_probe if args.command == "probe" else run_support)(args)
            elif args.command in ("shift", "composition", "context"):
                from .descriptive import run_shift, run_composition, run_context
                {"shift": run_shift, "composition": run_composition, "context": run_context}[args.command](args)
            elif args.command == "compare":
                from .compare import run_compare
                run_compare(args)
            elif args.command == "plot":
                from .plot import plot_summary
                plot_summary(args)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f"error: {exc}\n")
