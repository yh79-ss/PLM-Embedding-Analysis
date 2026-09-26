# Additional diagnostic recipes

Run these from the repository root after installation and
`python -m vep_diagnostics demo --out demo`. The examples use synthetic data;
replace paths, representation descriptions and `--scope` for your own inputs.
Every output directory must be new. These are analyses of supplied embeddings
or predictions, not PLM extraction/training or biological-data download jobs.

## Fixed baselines and ranking controls

First generate source-only predictions, including the fixed-alpha control:

```bash
python -m vep_diagnostics probe \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic demo' \
  --alphas 0.1 1 10 --out outputs/source-controls

python -m vep_diagnostics evaluate \
  --metadata demo/variants.csv --predictions outputs/source-controls/predictions.csv \
  --columns prediction fixed_prediction --baseline-column baseline \
  --label-protocol no-target-label --scope 'synthetic demo' \
  --permutations 1000 --bootstrap 2000 --out outputs/evaluation
```

`summary.csv` reports assay-macro Spearman, conditional supercluster intervals,
random-ranking reference quantiles and one-sided permutation p-values.
`baseline_comparisons.csv` reports predictor-minus-baseline Spearman and the
fraction of paired assays below the baseline. `per_assay.csv` retains the
individual scores and undefined counts.

The null shuffles evaluation effects within each assay, preserving ties and
holding predictions fixed. It tests an exchangeability assumption; correlated
variants can violate it. It is **not** the shuffled-source-label refit control.
For that distinct control, rerun `probe` with `--shuffle-source-labels --seed 1`
(and other prespecified seeds), then evaluate each run. Never choose control
seeds after seeing their performance.

`permutation_p_holm` adjusts all task/model ranking tests in this one call,
including baseline columns. Baseline-difference intervals are not adjusted.
Positive ranking is not evidence of an improvement over the baseline. Separate
runs/representations do not automatically form a corrected joint test family.

To package supplied fixed scores (ESM2 LLR, MSA log-odds, etc.):

```bash
python -m vep_diagnostics baselines \
  --metadata demo/variants.csv --score-columns baseline \
  --scope 'synthetic fixed prior' --out outputs/fixed-priors
```

Your real scalar columns must already be computed without target-label fitting.
List their names after `--score-columns`; output is `predictions.csv`. A baseline
column for `evaluate` can be in the prediction file or the metadata. Supplied
prediction-file values take precedence. Scores must have a prespecified
direction: larger predicts a larger measured effect. No evaluation-based sign
flipping is performed.

Optional BLOSUM62 substitution floor (canonical single substitutions only):

```bash
python -m pip install -e '.[baseline]'
python -m vep_diagnostics baselines \
  --metadata demo/variants.csv --blosum62 \
  --scope 'synthetic single substitutions' --out outputs/blosum
python -m vep_diagnostics evaluate \
  --metadata demo/variants.csv --predictions outputs/blosum/predictions.csv \
  --columns blosum62 --label-protocol no-target-label \
  --scope 'synthetic single substitutions' --out outputs/blosum-evaluation
```

This reads BioPython's bundled substitution matrix, not a protein-model output.
MSA feature generation, LLR extraction and replicate-noise-ceiling estimation
are not performed by this command.

## Normalized recovery

Use identical evaluation variants for source and oracle:

```bash
python -m vep_diagnostics diagnose \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic local-support diagnostic' \
  --allow-target-support --seeds 0 1 --out outputs/local

python -m vep_diagnostics evaluate \
  --metadata demo/variants.csv --predictions outputs/local/predictions.csv \
  --prediction-seed 0 --allow-intersection \
  --columns prediction_source --oracle-column prediction_oracle \
  --baseline-column baseline --oracle-min-rho 0.05 \
  --label-protocol target-support --scope 'synthetic local evaluation, seed 0' \
  --out outputs/recovery
```

`recovery.csv` reports `(rho_source - random_mean) / (rho_oracle - random_mean)`.
`recovery_summary.csv` averages finite assay-level ratios, not task-level ratios.
Here the random mean is the source-prediction permutation mean. The oracle
must reach the declared minimum rho and have a positive nonzero denominator;
otherwise recovery is blank and counted as invalid. Values can be negative or
above one; this is not a fraction of a known biological performance ceiling.

The explicit intersection selects the held-out rows of **one** oracle seed;
`excluded_keys.csv` records the removed support rows. It does not select assays
by outcome. Do not concatenate seeds and treat repeated variants as independent.
Use `diagnose/summary.csv` for its already paired complete-seed oracle-gap mean.

## Calibration and error modes

```bash
python -m vep_diagnostics errors \
  --metadata demo/variants.csv --predictions outputs/source-controls/predictions.csv \
  --columns prediction fixed_prediction --percentile-predictions \
  --label-protocol no-target-label --scope 'synthetic source prediction errors' \
  --bins 10 --tail-fraction 0.1 --max-pairs 20000 --out outputs/errors
```

Inspect:

- `per_assay.csv`: Spearman, pair-ranking accuracy, rank RMSE, top/bottom AP and
  AUROC, and optional original-prediction bias plus calibration slope/intercept.
- `calibration_bins.csv`: mean prediction and effect percentile rank in each
  within-assay prediction-rank bin. Ties can leave empty bins; none are invented.
- `error_strata.csv`: rank errors by effect-rank quartile, observed-position
  quartile and WT residue class, separately for each assay and predictor.

`--percentile-predictions` declares that original predictions were fitted to
percentile labels, as in `probe`. Do not use it for raw LLRs, BLOSUM scores or
rank-Gaussian probes. Bias is `mean(effect percentile rank - original prediction)`;
it is not the near-zero residual after fitting an intercept on those same rows.
Slope/intercept use evaluation labels **descriptively**. No calibrated predictor
or deployment-performance claim is produced. Rank-error summaries remain valid
without the flag, but are not raw-score calibration. Tail membership preserves
ties, so the positive fraction need not equal the requested tail fraction.

## Shift–performance associations

```bash
python -m vep_diagnostics shift \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic demo' \
  --projection-dim 0 --max-samples 64 --out outputs/shift-controls

python -m vep_diagnostics associate \
  --metadata demo/variants.csv --covariates outputs/shift-controls/held_out.csv \
  --metrics outputs/local/per_assay.csv \
  --x mmd_squared reference_percentile centroid_distance covariance_distance \
  --y rho_source oracle_gap --scope 'synthetic descriptive association' \
  --plot --out outputs/associations
```

`summary.csv` gives **within-task** Spearman with finite-pair counts;
`joined_assays.csv` is the audited join. Optional PNGs show the scatterplots.
You can substitute composition or other numeric assay-level covariates. Tables
must contain one row per assay; `--model prediction` selects one model from
`evaluate/per_assay.csv`. Repeated seeds/folds must be explicitly aggregated first.

The tool checks assay identities and population equality. `--allow-intersection`
permits a logged subset. It cannot infer whether two same-assay summaries used
identical variants; state that relationship. In the example, MMD uses the full
assay embeddings and local scores use held-out support-split rows. There is no
pooled-task coefficient, causal classification or significance test.

## Embedding visualization and quality checks

```bash
python -m vep_diagnostics geometry \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic embedding geometry' \
  --max-per-assay 50 --plot --out outputs/geometry
```

Outputs: `pca.png`, keyed `coordinates.csv`, `explained_variance.csv`,
`pca_basis.npz`, and `embedding_qc.csv` (zero vectors, constant features,
vector norms and feature variance). The shared PCA basis is fitted to a
label-free capped sample from every assay, centered without feature scaling.
It is an all-population visualization, **not** a source-only preprocessing step
for predictive evaluation. Keep the sampling rule fixed; 2D overlap or separation
does not establish transferability. No task/assay panels are selected by results.

## Cosine reference controls

```bash
python -m vep_diagnostics cosine \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic fold-local geometry' \
  --alpha 1 --reference-repeats 20 --out outputs/cosine-controls
```

Alongside observed `pairwise_cosine.csv`, inspect `reference_draws.csv` and
`reference_summary.csv`. For each task/fold, two independently assay-resampled
fits provide a same-pool bootstrap reference; two independently within-assay
label-shuffled fits provide a null reference. Scaling is refitted each time,
coefficients are converted to original coordinates, and the observed label
transform/alpha is retained. Entire assays, not individual variants, are
resampled; already transformed labels remain fixed within each copied assay.

At least two assays within a task/fold are needed for its bootstrap reference.
The small demo has only one, so it deliberately reports missing bootstrap
values while still producing shuffle references. This exposes an inadequate
reference population rather than returning the tautological cosine of one.
These within-pool brackets are not a guaranteed upper bound or a null matched
to every between-fold comparison. No normalized causal score is exported.

## Representation and complementarity comparisons

For a standalone interface comparison, run `probe` separately on two keyed
embedding files with the same metadata/folds and source alpha grid, then use
`compare` on their predictions. Compare the same pooling conventions when that
is the intended contrast; changes in model, pretraining and pooling remain part
of the complete interface comparison.

To ask whether block B adds predictive information beyond block A:

```bash
# Replace both NPZ paths/descriptions with the real, distinct feature blocks.
# The demo deliberately repeats the same block as a runnable dimension control.
python -m vep_diagnostics probe \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --aux-embeddings demo/embeddings.npz --aux-representation 'duplicate-block control' \
  --representation 'synthetic base plus duplicate' --scope 'synthetic demo' \
  --alphas 0.1 1 10 --out outputs/augmentation

python -m vep_diagnostics compare \
  --metadata demo/variants.csv \
  --predictions-a outputs/augmentation/predictions.csv \
  --predictions-b outputs/source-controls/predictions.csv \
  --label-protocol no-target-label --scope 'synthetic same-row augmentation' \
  --out outputs/augmentation-comparison
```

Auxiliary keys must match exactly. Both arms independently select alpha using
source folds; each coordinate is source-standardized. There is no separate
block-scale tuning. Do not pick a representation/combination using evaluation
results and then call it confirmatory. Learned auxiliary features must themselves
respect outer and inner fold boundaries; this generic global-NPZ interface does
not certify out-of-fold construction of a supervised encoder or learned stacker.
Frozen embeddings and label-independent fixed scalar features are the simplest
inputs. Duplicating features changes Ridge regularization geometry, so the
duplicate-block demo is not expected to produce an identical fit.

## Cross-task transfer

```bash
python -m vep_diagnostics cross-task \
  --metadata demo/variants.csv --embeddings demo/embeddings.npz \
  --representation 'synthetic 8D' --scope 'synthetic task-transfer matrix' \
  --baseline-column baseline --alphas 0.1 1 10 \
  --permutations 1000 --bootstrap 2000 --out outputs/cross-task
```

Read `rho_matrix.csv` and `increment_matrix.csv` (source tasks in rows), with
counts and intervals in `summary.csv`. A source-task model never sees any label
or fitted preprocessing from the target outer fold, even when the same protein
has assays in both tasks. Diagonals are same-task source-only controls. Each
source-task/outer-fold fit has independent inner source-fold alpha selection;
too few source folds fails rather than silently using target labels.

`ranking_p_holm` adjusts within-assay permutation ranking tests across all
off-diagonal directions in this **one representation**. Baseline-difference
intervals are conditional supercluster intervals and **not simultaneous**.
Thus this command does not reproduce the study's reliable-increment decision
rule or its historical pair-specific fold geometry. Do not count ordinary
interval exclusions as multiplicity-corrected reliable directions.

`predictions.csv` repeats variant keys for different `source_task` values.
Select a single source task before using those rows with `evaluate`/`errors`.
Use the simpler `probe --source-task Activity --target-task Binding` when only
one directed contrast is wanted. Supply a matched fixed prior in advance.

## Subgroups, provenance and external predictions

```bash
python -m vep_diagnostics evaluate \
  --metadata demo/variants.csv --predictions outputs/source-controls/predictions.csv \
  --columns prediction --baseline-column baseline --group-by provenance platform \
  --label-protocol no-target-label --scope 'synthetic provenance-stratified evaluation' \
  --out outputs/subgroups

python -m vep_diagnostics provenance \
  --metadata demo/variants.csv --group-by provenance platform \
  --scope 'synthetic source composition' --out outputs/provenance
```

`subgroups.csv` reports assay-equal scores and conditional cluster intervals
within each task/annotation level. `provenance/group_counts.csv` reports both
assay and variant fractions; `summary.csv` reports concentration, effective
group count and groups spanning folds. Annotations must be nonempty and constant
within each assay. The annotation names are user-defined (e.g., publication,
assay platform, PPI/DTI class, taxon or experimental context), not inferred from
assay names. This reports source structure; it does not prove a source effect.

For an external corpus, supply its keyed metadata and **already generated**
predictions, then use the same `evaluate`/`compare` commands. Fold and cluster
columns describe the external evaluation units; a single fold is accepted for
saved-prediction evaluation, with no refit. Cross-corpus model training, sequence
overlap screening, label-direction harmonization and dataset ingestion are not
automatically performed. Document those separately. Different external scopes
(e.g., coarse Binding and PPI-only Binding) must not be compared as if matched.

[Back to README](../README.md) · [Methods](METHODS.md) · [Coverage map](COVERAGE.md)
