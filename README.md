# PLM Embedding Analysis

Run protein-embedding diagnostics on your own precomputed variant vectors.
This repository includes our protein-level split and CPU tools for source/oracle
prediction, embedding MMD, probe-weight cosine, and related assay analyses.

## Install

```bash
git clone https://github.com/yh79-ss/PLM-Embedding-Analysis.git
cd PLM-Embedding-Analysis
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot]'
```

Requires Python 3.10+. No GPU or protein-model download is needed.

## Prepare your data

Supply a CSV with one row per variant and these columns:

| Column | Meaning |
|---|---|
| `assay_id`, `mutant` | Unique assay–variant key, matching your embedding keys |
| `protein_id` | Protein identifier |
| `task` | Task label, such as Activity, Binding, or Stability |
| `super_cluster` | Related-protein group kept within one fold |
| `fold` | Integer outer-fold assignment |
| `score` | Finite measured effect; needed for prediction and cosine, optional for MMD |

Store embeddings in a keyed `.npz` archive:

```python
import numpy as np

# embedding_rows describes the CURRENT row order of your N-by-D matrix X.
np.savez_compressed(
    "my_embeddings.npz",
    X=np.asarray(X, dtype=np.float32),
    assay_id=np.asarray(embedding_rows["assay_id"], dtype=str),
    mutant=np.asarray(embedding_rows["mutant"], dtype=str),
)
```

Use one vector per variant: mutation-site, sequence-mean, or another declared
pooling rule. Embeddings are aligned by keys, not assumed row order. Both files
must contain exactly the same keys, without duplicates or nonfinite values.
Keep all assays from the same protein and all related proteins in one fold.
See [input details](docs/METHODS.md#prepare-your-inputs) for additional columns,
score conventions, and supported mutation identifiers.

To try every command below with synthetic data:

```bash
python -m vep_diagnostics demo --out demo
```

For your own data, replace `demo/variants.csv` and `demo/embeddings.npz` in
the recipes. Replace the representation and scope descriptions too. The demo
contains synthetic Activity and Binding labels; it is not a ProteinGym result.

## Use our protein-level split

[The split CSV](splits/protein_fold_lookup_c08_v3_seed893.csv) covers
217 assays, 186 protein identifiers, and 163 operational sequence clusters
across five folds. [Split construction and checksum](splits/README.md).

```bash
python -m vep_diagnostics check-split \
  --split splits/protein_fold_lookup_c08_v3_seed893.csv

python -m vep_diagnostics attach-split \
  --metadata my_variants_without_folds.csv \
  --split splits/protein_fold_lookup_c08_v3_seed893.csv \
  --out outputs/attached-split
```

The input needs `assay_id`, `mutant`, and `task`; other columns are preserved.
Use `outputs/attached-split/variants.csv` downstream. Unknown assay IDs fail.
For different proteins, supply your own cluster-respecting folds.

## Check your inputs

```bash
python -m vep_diagnostics validate \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz
```

This checks key alignment and prevents proteins or clusters from spanning folds.

## Calculate rho_source, rho_oracle, and the oracle gap

```bash
python -m vep_diagnostics diagnose \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D features' --scope 'synthetic demo' \
  --allow-target-support --support-fraction 0.5 --seeds 0 1 2 3 4 \
  --fixed-alpha 1 --out outputs/diagnose
```

The source probe fits other folds of the same task. The oracle fits a random
support subset of the target assay. Both are scored on the **same remaining
target variants**.

Open `outputs/diagnose/summary.csv`:

| Output column | Calculation |
|---|---|
| `rho_source` | Spearman correlation between measured effects and source-probe predictions |
| `rho_oracle` | Spearman correlation between measured effects and target-support-probe predictions |
| `oracle_gap` | `rho_oracle - rho_source` on identical evaluation variants |
| `weight_cosine_source_oracle` | Cosine between source and oracle weights in the original embedding coordinates |
| `n_valid_paired_assays` | Number of assays contributing finite, complete paired results |

`per_seed.csv` and `per_assay.csv` contain the detailed metrics;
`predictions.csv` contains `prediction_source` and `prediction_oracle`.
`oracle_splits.csv` records exactly which rows were support versus evaluation.

Defaults use alpha 1 for both probes. To choose their shared alpha using source
folds only, add `--source-selection nested --alphas 0.1 1 10 100 1000`.
That mode needs at least two source folds for every outer fit.

The oracle explicitly uses target-support labels; it is a diagnostic, not a
no-target-label result or guaranteed performance ceiling. Random support can
share positions with evaluation variants. Metrics average seeds within assays,
then complete paired assays equally. [Method details](docs/METHODS.md#matched-sourceoracle-diagnostic).

### Source-only prediction without an oracle

```bash
python -m vep_diagnostics probe \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D features' --scope 'synthetic demo' \
  --alphas 0.1 1 10 --out outputs/source
```

Read `outputs/source/source_scores.csv`: `rho_source` is the source-selected
readout and `rho_source_fixed` the fixed-alpha control. Optional input
`baseline` scores produce `rho_baseline`. These scores use all held-out rows,
whereas `diagnose` uses the oracle's held-out subset.

For cross-task transfer, add both `--source-task Activity --target-task Binding`.
For a shuffled-source-label control, add `--shuffle-source-labels --seed 1`
and use a new output directory.

## Calculate embedding distance: MMD

```bash
python -m vep_diagnostics shift \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D features' --scope 'synthetic demo' \
  --projection-dim 0 --max-samples 64 --out outputs/mmd
```

Read `outputs/mmd/held_out.csv`:

| Output column | Meaning |
|---|---|
| `mmd_squared` | Multi-scale RBF MMD² between a held-out assay and its same-task source pool |
| `reference_percentile` | Its location in the matched source-assay distance reference |
| `n_pool_sample`, `n_assay_sample` | Actual sample sizes used |

`source_reference.csv` contains the calibration distances;
`summary.csv` contains task medians. The score column is not used.

`--projection-dim 0` uses full embeddings; the default is a seeded 128D
Gaussian projection. The demo caps each side at 64 rows; for a larger run, use
a prespecified cap such as `--max-samples 1000`. Keep these settings fixed when
comparing runs. The exported statistic is **MMD²**; take its square root if you
need MMD. The reference percentile is descriptive, not a p-value.
[Kernel and sampling details](docs/METHODS.md#embedding-shift-and-source-calibration).

## Calculate probe-weight cosine across folds

```bash
python -m vep_diagnostics cosine \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D features' --scope 'synthetic fold-local diagnostic' \
  --alpha 1 --out outputs/cosine
```

This fits a separate probe **within each fold**, then compares same-task
probe weights between folds. Read `pairwise_cosine.csv` for `weight_cosine`
and `summary.csv` for `mean_weight_cosine`. The fitted weights are saved in
`probe_weights.npz`, with matching task/fold identifiers; counts and intercepts
are in `probe_metadata.csv`.

Weights are converted back from each scaler to the same original embedding
coordinates before computing the cosine. The default label transform is
within-assay rank-Gaussian; `--label-transform percentile` changes it.

This is the fold-local cosine diagnostic. The
`weight_cosine_source_oracle` from `diagnose` compares different models and
must be labeled separately. Neither low cosine nor a large oracle gap alone
identifies a biological mechanism. [Definitions and limitations](docs/METHODS.md#probe-weight-cosine).

## Other analysis recipes

| Tool | Inputs in addition to standard metadata | Output to inspect |
|---|---|---|
| `composition` | `protein_length`; no labels or embeddings needed | `coverage.csv`, `per_assay.csv`: coverage and substitution/position JSD |
| `context` | `background_sequence`, `mutated_sequence`, `score` | `pair_concordance.csv`: exact-background assay agreement |
| `support` | Embeddings and explicit target-support access | `per_assay.csv`: overlap/disjoint support, fixed/tuned readouts |
| `compare` | Two prediction CSVs plus evaluation metadata | `summary.csv`: paired Spearman difference and cluster interval |

```bash
python -m vep_diagnostics composition \
  --metadata demo/variants.csv --scope 'synthetic demo' --out outputs/composition

python -m vep_diagnostics context \
  --metadata demo/variants.csv --scope 'synthetic context pairs' --out outputs/context

python -m vep_diagnostics support \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D features' --scope 'synthetic position-support diagnostic' \
  --allow-target-support --support-size 64 --seeds 0 1 --out outputs/support

python -m vep_diagnostics compare \
  --metadata demo/variants.csv \
  --predictions-a outputs/source/predictions.csv --column-a prediction \
  --predictions-b outputs/source/predictions.csv --column-b baseline \
  --label-protocol no-target-label --scope 'synthetic demo' --out outputs/comparison

python -m vep_diagnostics plot \
  --summary outputs/comparison/summary.csv --output outputs/comparison.png
```

The comparison example uses the demo's optional `baseline` column. For your
data, point to your own two prediction files. Use identical variant populations;
`--allow-intersection` explicitly permits a reduced population and logs exclusions.

## Outputs and help

Every `--out` must name a **new** directory. Check `manifest.json` for
`status: complete`; it records options, input hashes, versions and label access.
Blank metric cells mean undefined values, not zero. Retain validity counts.
State your population in `--scope`, including the actual Binding subset.

```bash
python -m vep_diagnostics diagnose --help
python -m vep_diagnostics cosine --help
python -m vep_diagnostics shift --help
python -m pytest -q
```

[Detailed methods and troubleshooting](docs/METHODS.md) ·
[Split provenance](splits/README.md) · [Implementation provenance](PROVENANCE.md)

This is a portable analysis toolkit, not an exact replay of the study. It
contains no raw assay data, embeddings, checkpoints or study result tables.
