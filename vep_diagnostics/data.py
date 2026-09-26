"""Strict keyed inputs, fold guards, and write-once run provenance."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd

from . import __version__

KEY = ["assay_id", "mutant"]
IDENTITY = ["assay_id", "protein_id", "task", "super_cluster", "fold"]
AA = "ACDEFGHIKLMNPQRSTVWY"
SINGLE = re.compile(r"^([" + AA + r"])([1-9][0-9]*)([" + AA + r"])$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(seed, *parts):
    payload = json.dumps([int(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def require_columns(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input table is empty")
    for col in columns:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing/empty values in {col}")


def read_csv(path):
    # Do not let identifiers such as 'NA' become missing values.
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def keys(frame):
    return pd.MultiIndex.from_frame(frame[KEY])


def validate_metadata(frame):
    frame = frame.copy()
    require_columns(frame, KEY + IDENTITY[1:])
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate (assay_id, mutant) keys; resolve replicates before analysis")
    fold = pd.to_numeric(frame.fold, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(fold).all() or (fold < 0).any() or (fold != np.floor(fold)).any():
        raise ValueError("fold must contain nonnegative integers")
    frame["fold"] = fold.astype(int)
    for col in IDENTITY[1:]:
        if frame.groupby("assay_id")[col].nunique().gt(1).any():
            raise ValueError(f"An assay maps to multiple {col} values")
    for col in ("protein_id", "super_cluster"):
        if frame.groupby(col).fold.nunique().gt(1).any():
            raise ValueError(f"Leakage: {col} spans multiple folds")
    if frame.groupby("protein_id").super_cluster.nunique().gt(1).any():
        raise ValueError("A protein maps to multiple superclusters")
    return frame.reset_index(drop=True)


def load_metadata(path):
    return validate_metadata(read_csv(path))


def scores(frame, column="score"):
    require_columns(frame, [column])
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} contains nonfinite values; define a finite analysis population first")
    return values


def load_embeddings(path, frame):
    with np.load(path, allow_pickle=False) as archive:
        if not {"X", *KEY}.issubset(archive.files):
            raise ValueError("NPZ requires X, assay_id, and mutant arrays")
        x = archive["X"]
        ids = [archive[col] for col in KEY]
        if x.ndim != 2 or x.shape[1] == 0 or x.dtype.kind not in "fiu":
            raise ValueError("X must be a real numeric [variants, features] matrix")
        for value in ids:
            if value.ndim != 1 or value.dtype.kind != "U" or len(value) != len(x):
                raise ValueError("NPZ keys must be 1-D Unicode arrays with one key per X row")
        indexed = pd.DataFrame(dict(zip(KEY, ids)))
        if indexed.duplicated(KEY).any():
            raise ValueError("Duplicate embedding keys")
        source, target = keys(indexed), keys(frame)
        missing, extra = target.difference(source), source.difference(target)
        if len(missing) or len(extra):
            raise ValueError(f"Embedding population mismatch: {len(missing)} missing, {len(extra)} extra keys")
        x = np.asarray(x[source.get_indexer(target)], dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Embeddings contain nonfinite values")
    return x


def single_substitutions(frame):
    parsed = frame.mutant.str.extract(SINGLE)
    valid = parsed.notna().all(axis=1) & parsed[0].ne(parsed[2])
    result = frame.loc[valid].copy()
    result["wt_aa"] = parsed.loc[valid, 0]
    result["position"] = parsed.loc[valid, 1].astype(int)
    result["mut_aa"] = parsed.loc[valid, 2]
    result["substitution"] = result.wt_aa + ">" + result.mut_aa
    return result


def verify_split(path):
    split = read_csv(path)
    require_columns(split, ["DMS_id", "UniProt_ID", "super_cluster", "fold_protein_5"])
    if split.DMS_id.duplicated().any():
        raise ValueError("Split contains duplicate assay IDs")
    normalized = split.rename(columns={"DMS_id": "assay_id", "UniProt_ID": "protein_id", "fold_protein_5": "fold"})
    normalized["mutant"] = "split_record"
    normalized["task"] = "split_record"
    normalized = validate_metadata(normalized)
    summary = {"sha256": sha256(path), "assays": len(split), "proteins": split.UniProt_ID.nunique(),
               "superclusters": split.super_cluster.nunique(),
               "assays_per_fold": normalized.groupby("fold").size().to_dict(),
               "superclusters_per_fold": normalized.groupby("fold").super_cluster.nunique().to_dict()}
    return split, summary


def attach_split(metadata, split):
    require_columns(metadata, KEY + ["task"])
    lookup = split.rename(columns={"DMS_id": "assay_id", "UniProt_ID": "protein_id", "fold_protein_5": "fold"})
    lookup = lookup[["assay_id", "protein_id", "super_cluster", "fold"]]
    merged = metadata.merge(lookup, on="assay_id", how="left", validate="many_to_one", suffixes=("", "_lookup"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError(f"Unknown assay IDs in split: {merged.loc[merged._merge.ne('both'), 'assay_id'].unique().tolist()}")
    for col in ("protein_id", "super_cluster", "fold"):
        if col + "_lookup" in merged:
            if not merged[col].astype(str).eq(merged[col + "_lookup"].astype(str)).all():
                raise ValueError(f"Metadata disagrees with frozen split: {col}")
            merged = merged.drop(columns=col + "_lookup")
    return validate_metadata(merged.drop(columns="_merge"))


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


class Run:
    """Reserve a new directory; a failed run never looks complete."""
    def __init__(self, args, protocol, inputs):
        self.path = Path(args.out)
        self.path.mkdir(parents=True, exist_ok=False)
        versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn")}
        self.manifest = {"status": "started", "version": __version__, "python": platform.python_version(),
                         "packages": versions, "started_utc": datetime.now(timezone.utc).isoformat(),
                         "label_protocol": protocol, "arguments": vars(args),
                         "inputs": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in inputs],
                         "source_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                         "decision_impact": "none; reusable analysis software, no study conclusions regenerated"}
        write_json(self.path / "manifest.json", self.manifest)

    def table(self, name, frame):
        # Scope and label access travel with every table, including exclusions.
        frame = frame.copy()
        frame["analysis_scope"] = self.manifest["arguments"].get("scope", "split metadata")
        frame["label_protocol"] = self.manifest["label_protocol"]
        frame.to_csv(self.path / name, index=False)

    def finish(self, **details):
        self.manifest.update(details, status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        self.manifest["outputs"] = {p.name: sha256(p) for p in sorted(self.path.iterdir()) if p.name != "manifest.json"}
        write_json(self.path / "manifest.json", self.manifest)
