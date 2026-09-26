"""Label-free embedding/provenance views and assay-level descriptive associations."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .data import IDENTITY, KEY, Run, load_embeddings, load_metadata, read_csv, require_columns, stable_seed
from .descriptive import sample_indices
from .metrics import rho


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_geometry(args):
    frame = load_metadata(args.metadata)
    x = load_embeddings(args.embeddings, frame)
    if min(x.shape) < 2:
        raise ValueError("Two-dimensional PCA needs at least two rows and features")
    idx = np.concatenate([sample_indices(g.index.to_numpy(), args.max_per_assay, stable_seed(args.seed, a, "pca"))
                          for a, g in frame.groupby("assay_id", sort=True)])
    sampled = x[idx]
    if np.sum(np.var(sampled, axis=0)) == 0:
        raise ValueError("PCA is undefined for a constant embedding matrix")
    run = Run(args, "descriptive all-population embedding visualization; no effect-label use", [args.metadata, args.embeddings])
    pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
    coordinates = pca.fit_transform(sampled)
    table = frame.iloc[idx][list(dict.fromkeys(KEY + IDENTITY))].copy()
    table["PC1"], table["PC2"] = coordinates[:, 0], coordinates[:, 1]
    run.table("coordinates.csv", table)
    run.table("explained_variance.csv", pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": pca.explained_variance_ratio_}))
    rows = []
    for assay, group in frame.groupby("assay_id", sort=True):
        values = x[group.index.to_numpy()]
        rows.append({**{c: group[c].iloc[0] for c in IDENTITY}, "n_variants": len(group),
                     "dimension": x.shape[1], "n_zero_vectors": int((np.linalg.norm(values, axis=1) == 0).sum()),
                     "n_constant_features": int((np.ptp(values, axis=0) == 0).sum()),
                     "mean_vector_norm": float(np.linalg.norm(values, axis=1).mean()),
                     "mean_feature_variance": float(np.var(values, axis=0).mean())})
    run.table("embedding_qc.csv", pd.DataFrame(rows))
    np.savez_compressed(run.path / "pca_basis.npz", components=pca.components_, mean=pca.mean_)
    if args.plot:
        plt = plotting()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col in zip(axes, ("task", "fold")):
            for value, group in table.groupby(col, sort=True):
                ax.scatter(group.PC1, group.PC2, s=7, alpha=.45, label=str(value))
            ax.set(xlabel="PC1", ylabel="PC2", title=f"Sampled embeddings by {col}")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(run.path / "pca.png", dpi=160)
        plt.close(fig)
    run.finish(sampling="independent label-free uniform cap per assay; exact sampled keys saved",
               preprocessing="centered PCA, no feature scaling; one shared all-population basis",
               limitation="transductive descriptive picture, not a train-only feature transform, separation test, or proof of absent shift")


def assay_table(path, metadata, columns, model=None):
    frame = read_csv(path)
    if model is not None:
        require_columns(frame, ["model"])
        frame = frame[frame.model.eq(model)].copy()
    require_columns(frame, ["assay_id"])
    if set(columns) - set(frame):
        raise ValueError(f"Missing metric columns: {sorted(set(columns) - set(frame))}")
    if frame.assay_id.duplicated().any():
        raise ValueError("Assay table has repeated rows; select one model/seed or use a documented per-assay aggregate")
    for col in IDENTITY[1:]:
        source_col = "outer_fold" if col == "fold" and "outer_fold" in frame else col
        if source_col in frame:
            left = frame[["assay_id", source_col]].rename(columns={source_col: "provided"})
            joined = left.merge(metadata[["assay_id", col]], on="assay_id")
            if not joined.provided.astype(str).eq(joined[col].astype(str)).all():
                raise ValueError(f"Assay table identity disagrees: {col}")
    for col in columns:
        # Blank exported cells are undefined; arbitrary text is not silently dropped.
        frame[col] = pd.to_numeric(frame[col].replace("", np.nan), errors="raise")
        if np.isinf(frame[col]).any():
            raise ValueError(f"Infinite assay metric: {col}")
    return frame


def run_associate(args):
    metadata = load_metadata(args.metadata).drop_duplicates("assay_id")[IDENTITY]
    cov = assay_table(args.covariates, metadata, args.x)
    performance = assay_table(args.metrics, metadata, args.y, args.model)
    populations = {"metadata": set(metadata.assay_id), "covariates": set(cov.assay_id), "metrics": set(performance.assay_id)}
    common = set.intersection(*populations.values())
    if not args.allow_intersection and any(s != common for s in populations.values()):
        raise ValueError("Assay populations differ; align first or use --allow-intersection")
    if not common:
        raise ValueError("No common assays")
    joined = metadata[metadata.assay_id.isin(common)].merge(cov[["assay_id", *args.x]], on="assay_id", validate="one_to_one")
    joined = joined.merge(performance[["assay_id", *args.y]], on="assay_id", validate="one_to_one")
    run = Run(args, "descriptive assay-level association; performance labels used only through saved metrics", [args.metadata, args.covariates, args.metrics])
    run.table("excluded_assays.csv", pd.DataFrame([{"input": name, "assay_id": a} for name, ids in populations.items() for a in sorted(ids - common)], columns=["input", "assay_id"]))
    run.table("joined_assays.csv", joined)
    summaries = []
    for task, group in joined.groupby("task", sort=True):
        for x in args.x:
            for y in args.y:
                finite = np.isfinite(group[x]) & np.isfinite(group[y])
                summaries.append({"task": task, "x": x, "y": y, "n_assays": len(group), "n_finite_pairs": int(finite.sum()),
                                  "spearman": rho(group.loc[finite, x], group.loc[finite, y])})
    run.table("summary.csv", pd.DataFrame(summaries))
    if args.plot:
        plt = plotting()
        for i, x in enumerate(args.x):
            for j, y in enumerate(args.y):
                fig, ax = plt.subplots(figsize=(6, 4))
                for task, group in joined.groupby("task", sort=True):
                    ax.scatter(group[x], group[y], label=task, s=20, alpha=.7)
                ax.set(xlabel=x, ylabel=y)
                ax.legend(fontsize=8)
                fig.tight_layout()
                fig.savefig(run.path / f"association_{i}_{j}.png", dpi=160)
                plt.close(fig)
    run.finish(interpretation="within-task descriptive Spearman; no pooled task coefficient, significance tests, or causal driver categories",
               comparison_scope="assay IDs/identities checked, within-assay variant population equivalence must be verified by the caller")


def run_provenance(args):
    frame = load_metadata(args.metadata)
    require_columns(frame, args.group_by)
    for col in args.group_by:
        if frame.groupby("assay_id")[col].nunique().gt(1).any():
            raise ValueError(f"Provenance column {col} varies within an assay")
    assays = frame.groupby(IDENTITY + args.group_by, as_index=False).size().rename(columns={"size": "n_variants"})
    run = Run(args, "descriptive provenance concentration; no effect-label use", [args.metadata])
    rows, summaries = [], []
    for task, group in assays.groupby("task", sort=True):
        for col in args.group_by:
            counts = group.groupby(col).agg(n_assays=("assay_id", "size"), n_variants=("n_variants", "sum"), n_folds=("fold", "nunique"))
            for value, record in counts.iterrows():
                rows.append({"task": task, "group_column": col, "group_value": value, **record.to_dict(),
                             "assay_fraction": record.n_assays / len(group), "variant_fraction": record.n_variants / group.n_variants.sum()})
            fraction = counts.n_assays.to_numpy(float) / len(group)
            summaries.append({"task": task, "group_column": col, "n_assays": len(group), "n_groups": len(counts),
                              "largest_assay_fraction": fraction.max(), "effective_group_count": 1 / np.sum(fraction ** 2),
                              "n_groups_spanning_folds": int((counts.n_folds > 1).sum())})
    run.table("assays.csv", assays)
    run.table("group_counts.csv", pd.DataFrame(rows))
    run.table("summary.csv", pd.DataFrame(summaries))
    run.finish(limitation="protein-disjoint folds need not be provenance-disjoint; shared provenance is not by itself label leakage")
