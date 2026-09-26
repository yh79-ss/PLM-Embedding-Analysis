"""Small synthetic fixture, not biological data or a performance benchmark."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import write_json


def make_demo(out, seed):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    rows, features = [], []
    coefficient = rng.normal(size=8)
    for fold in range(5):
        background = "A" * 35 + "CDEFG"[fold]
        for position in range(1, 36):
            for alternative in "CDEFGHIK":
                x = rng.normal(size=8)
                shared = float(x @ coefficient + rng.normal(scale=0.3))
                for task in ("Activity", "Binding"):
                    rows.append({"assay_id": f"synthetic_{task}_{fold}", "mutant": f"A{position}{alternative}",
                                 "protein_id": f"synthetic_protein_{fold}", "task": task,
                                 "super_cluster": f"synthetic_cluster_{fold}", "fold": fold,
                                 "protein_length": len(background), "score": shared + rng.normal(scale=0.2),
                                 "baseline": shared + rng.normal(scale=1.0), "background_sequence": background,
                                 "mutated_sequence": background[:position - 1] + alternative + background[position:]})
                    features.append(x.copy())
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "variants.csv", index=False)
    x = np.asarray(features, dtype=np.float32)
    order = rng.permutation(len(frame))
    np.savez_compressed(out / "embeddings.npz", X=x[order],
                        assay_id=frame.assay_id.to_numpy(dtype=str)[order], mutant=frame.mutant.to_numpy(dtype=str)[order])
    write_json(out / "ABOUT.json", {"synthetic": True, "seed": seed, "n_variants": len(frame),
                                    "note": "Binding scope is synthetic, not any ProteinGym Binding population"})
