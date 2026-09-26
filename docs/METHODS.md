# Method and input reference

For runnable commands and output columns, start with the [README](../README.md).
This reference contains the longer input, protocol and interpretation details.
The [additional recipes](RECIPES.md) document the v0.3 baseline, recovery,
prediction-error, geometry, cosine-reference, augmentation, cross-task and
provenance tools. The [coverage map](COVERAGE.md) traces them to the slides/code
and explains the excluded historical procedures.

## Prepare your inputs

### Variant metadata: one CSV row per assay–variant key

All analyses use the same metadata format. Column names are case-sensitive.

| Column | Required for | Meaning |
|---|---|---|
| `assay_id` | All | Stable assay identifier, exactly matching the embedding key |
| `mutant` | All | Variant identifier; unique within each assay |
| `protein_id` | All | Protein identity shared by all assays of that protein |
| `task` | All | Task label, e.g. `Activity`, `Binding`, `Expression`, `OrganismalFitness`, `Stability`; custom labels work |
| `super_cluster` | All | Operational sequence-cluster group that must stay within one fold |
| `fold` | All | Nonnegative integer outer-fold assignment |
| `score` | `probe`, `diagnose`, `cosine`, `support`, `context`, `compare` | Finite measured effect, already oriented in the intended direction |
| `baseline` | Optional for `probe` | Existing finite fixed-prior score, already oriented so larger means larger effect |
| `protein_length` | `composition` | Positive integer full reference-protein length |
| `background_sequence` | `context` | Full wild-type sequence for that assay |
| `mutated_sequence` | `context` | Full sequence for that variant |

Minimal example (a few rows illustrate the schema; this is too small to fit):

```csv
assay_id,mutant,protein_id,task,super_cluster,fold,score
assay_A,A10V,protein_A,Activity,cluster_A,0,0.42
assay_A,G25D,protein_A,Activity,cluster_A,0,-0.13
assay_B,L8P,protein_B,Activity,cluster_B,1,-0.77
```

Identifiers are exact strings. The tools do not perform fuzzy matching, alias
repair, residue renumbering, or automatic score-direction inference. Resolve
replicates into one prespecified value per key before analysis. Do not pick the
replicate, sign, population, layer, or pooling rule using held-out performance.

An assay must map to one protein, task, supercluster and fold; a protein must map
to one supercluster; proteins and superclusters must never span folds. An assay
measuring multiple proteins is outside this one-protein-per-assay schema.

For `composition`, `support`, and `context`, substitutions use canonical
one-based full-sequence numbering such as `A10V`. They must use the 20 standard
amino-acid letters and change the amino acid. `composition` and `context`
explicitly count excluded nonsingle/noncanonical rows; `support` rejects them.
`probe` and `shift` treat `mutant` as an opaque key and can analyze other variant
types if the supplied embeddings and study protocol support them.

`context` checks the reference residue and the entire mutated sequence.
`composition` checks the position against `protein_length`. `support` uses
positions parsed from identifiers, so validate full-sequence numbering upstream.

### Embeddings: a keyed NumPy archive

Provide an `.npz` file with these arrays:

| Array | Shape and dtype | Meaning |
|---|---|---|
| `X` | `[N, D]`, real numeric, finite | One vector per variant |
| `assay_id` | `[N]`, NumPy Unicode | Assay key for each vector |
| `mutant` | `[N]`, NumPy Unicode | Variant key for each vector |

Export from your own feature-generation pipeline:

```python
import numpy as np

# embedding_rows describes X in its CURRENT order, not the metadata CSV's order.
np.savez_compressed(
    "inputs/my_embeddings.npz",
    X=np.asarray(X, dtype=np.float32),
    assay_id=np.asarray(embedding_rows["assay_id"], dtype=str),
    mutant=np.asarray(embedding_rows["mutant"], dtype=str),
)
```

The parent directory must already exist. Use Unicode strings, not object arrays;
loading pickled arrays is disabled. Embeddings are joined by the two keys, so
their row order may differ from the CSV. Duplicate keys, missing keys, extra
keys, nonfinite features, and non-vector features are errors. **Equal row counts
alone do not establish alignment.**

For residue-level arrays, choose pooling before export: mutation-site, sequence
mean, or another explicitly defined vector. Keep mutant vectors, wild-type
vectors and mutant-minus-wild-type vectors distinct. If using deltas, compute
them only after matching the same residue/variant keys. Do not silently flatten
residue-by-feature tensors. Learned projections, normalization or feature
selection must respect the source folds; do not fit them using target labels.

Every embedding command requires `--representation`, for example:
`'ESM2 layer33 mutant mutation-site, 1280D'`. This description is stored in the
manifest; it is not an automatic verification of how features were made.

The input population is exactly the rows you supply: no automatic row cap,
nonfinite-value dropping, label-availability filtering for descriptive tools,
or model-coverage intersection is performed. Record upstream exclusions and
sampling seeds. For fair interface comparisons, explicitly fix common variant
keys before fitting as well as before evaluation; the toolkit cannot infer
missing source-feature availability in another run.

## Use our protein-level split

The frozen file is
[splits/protein_fold_lookup_c08_v3_seed893.csv](../splits/protein_fold_lookup_c08_v3_seed893.csv).
It has one row per assay, with `DMS_id`, `UniProt_ID`, `super_cluster`, and
`fold_protein_5`. It contains no effect labels or sequences.

| Fold | Assays | Superclusters |
|---|---:|---:|
| 0 | 43 | 30 |
| 1 | 44 | 33 |
| 2 | 44 | 34 |
| 3 | 43 | 33 |
| 4 | 43 | 33 |
| Total | 217 | 163 |

There are **186 distinct saved protein identifiers**. All assays and variants
for a protein, and all proteins in an operational supercluster, stay in the same
fold. Fold `k` is evaluation; other folds supply training and inner validation.
Inner validation must exclude its whole fold and fit preprocessing only on its
inner-training rows.

```bash
python -m vep_diagnostics check-split \
  --split splits/protein_fold_lookup_c08_v3_seed893.csv

python -m vep_diagnostics attach-split \
  --metadata inputs/variants_before_split.csv \
  --split splits/protein_fold_lookup_c08_v3_seed893.csv \
  --out outputs/attached-split
```

The second command expects at least `assay_id`, `mutant`, and `task`; other
columns, including `score`, are preserved. It writes
`outputs/attached-split/variants.csv`. An existing protein/cluster/fold column
must agree with the lookup. Unknown assay IDs fail; missing assays are not
silently assigned to folds or discarded. Subsets of known assays are allowed,
but their results must name the subset rather than imply all 217 assays.

For ProteinGym input, map each processed assay's `DMS_score` to `score`, use its
`DMS_id` as `assay_id`, and obtain `task` from the assay metadata. Preserve the
released processed score direction; do not invert an already adjusted score
again based on a raw-directionality field. The original downloaded data are not
included. You must obtain and prepare the relevant release separately.

The split was built from MMseqs2 sequence relationships, starting with 30%
identity and 80% **target-sequence** coverage, followed by recorded bridge
merges. These are operational sequence groups, not proof of biological-family,
domain, structural-fold, or remote-homology separation. A fold contains many
groups. The original MMseqs2 version and complete bridge-search coverage flags
were not preserved. Use the released lookup for exact assignments; rerunning
clustering with a current tool is a new split. Full provenance, limitations,
the seed-selection rule, and checksum are in [splits/README.md](../splits/README.md).

For your own proteins, provide your own validated `protein_id`, `super_cluster`,
and `fold` columns. This package validates group separation but does not discover
homology, generate new sequence clusters, or map unknown proteins into our split.
Do not assign related proteins independently to random folds.


## Matched source/oracle diagnostic

The `diagnose` command uses random-row target support, separately from the
position-controlled `support` command. The default support fraction is 0.5 and
seeds are 0–4. Support size is `round(n * fraction)`; at least three support and
three evaluation rows are required. Splits depend on row order, assay identity
and the seed, never on measured effects. The saved split manifest is the exact
row-assignment authority. Keep input ordering fixed for exact repeats.

For each task and outer fold, fit the source probe on other folds of that task.
Fit the oracle on support rows from one target assay only. Both models use
independently fitted StandardScalers, percentile-rank fitting labels, and SVD
Ridge. Labels are ranked only within the fitting assay or support set. Their
predictions are scored against the same held-out target variants.

Since v0.4, `--fixed-alpha` controls the fixed source and `--oracle-alpha`
independently controls the oracle; both default to 1. With
`--source-selection nested`, source alpha is selected on source folds using the
`probe` rule, while the oracle stays fixed. The fixed source, selected source
and oracle share evaluation rows. Thus source improvement and oracle-gap
reduction isolate the source-readout change. The oracle is not optimized on
evaluation labels. Use the separately labeled `support` command for the
position-controlled support-tuned design. Versions before v0.4 shared the
selected alpha with the oracle; those nested runs answer a different question.

Definitions for one assay and support seed:

- `rho_source = Spearman(evaluation effects, source predictions)`.
- `rho_oracle = Spearman(evaluation effects, oracle predictions)`.
- `oracle_gap = rho_oracle - rho_source`.
- `weight_cosine_source_oracle` is cosine between the source and oracle
  coefficients after undoing their own feature standardization.

The oracle is a target-support readout, not a mathematical upper bound; its
correlation may be below the source correlation. Random-row support can contain
different substitutions at evaluation positions. It is not a position-disjoint
diagnostic or a replay of the study's frozen population.

Per-assay rho values average over the declared seeds only if every seed has
finite values for both models. Task summaries average those complete paired
assays equally. Counts are disclosed. Cosines have their own finite-assay count
and may have a different valid population. Seeds are repeated measurements, not
independent assays. `paired_per_assay.csv` and `paired_summary.csv` report
paired differences and 95% supercluster-bootstrap intervals after averaging
within assay over complete seeds. They use a common finite population across
fixed source, selected source, and oracle. Intervals condition on fitted models,
selected alphas, and realized support splits; they are not refit/support-sampling
intervals or simultaneous discoveries. No cosine interval or gap p-value is inferred.

## Probe-weight cosine

For a model fitted on standardized features `z = (x - mean) / scale`, compare
weights in the original common embedding coordinates:

```text
w_raw = w_standardized / scale
b_raw = b_standardized - dot(mean, w_raw)
cosine(w_a, w_b) = dot(w_a, w_b) / (norm(w_a) * norm(w_b))
```

The intercept is not part of the cosine. Zero-norm or nonfinite vectors produce
an undefined value, not zero. Only compare vectors from the same representation,
layer, pooling convention, feature ordering and coordinate basis. Undoing each
scaler makes coordinates comparable within a run; it does not make cosine
invariant to arbitrary feature rescaling or rotation across different models.

The `cosine` command implements the fold-local diagnostic: fit each task's
probe using labels **inside one fold**, then compare all distinct fold pairs
within that task. These are disjoint fitting populations, unlike the heavily
overlapping training sets of leave-one-fold-out source models. Each fold contains
multiple operational protein clusters; this is not one biological family per
probe and not a predictive cross-family evaluation.

The default label transform matches the historical fold-local producer:
`norm.ppf(clip((average_rank - 0.5) / n, 1e-6, 1 - 1e-6))`, separately within
each fitting assay. `--label-transform percentile` instead uses
`(average_rank - 1) / (n - 1)` (singleton 0.5). It changes the diagnostic.
The fixed alpha defaults to 1; no selection uses performance outcomes.

`probe_weights.npz` saves the raw-coordinate weights with matching `task` and
`fold` arrays; `probe_metadata.csv` saves intercepts, norms and counts.
`pairwise_cosine.csv` contains `weight_cosine`; `summary.csv` averages finite
fold pairs within task and reports their count. No confidence interval is
computed because fold pairs share probes.

The direct source/oracle cosine in `diagnose` is a different estimand. Neither
cosine has a universal cutoff for biological mapping shift. Correlated features,
regularization, sample size and coefficient instability affect these numbers.
The original study later restricted direct source/oracle cosine to historical
method context; these new generic outputs do not revive its withdrawn numeric
claims or causal thresholds.

## Source-only readout diagnostics

```bash
python -m vep_diagnostics probe \
  --metadata inputs/variants.csv --embeddings inputs/my_embeddings.npz \
  --representation 'my model / layer / pooling / mutant-or-delta convention' \
  --scope 'my prespecified assay population' \
  --alphas 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
  --fixed-alpha 1 --out outputs/my-probe
```

By default, each task is fitted and evaluated separately. For every outer fold:

1. Exclude the entire evaluation fold from source fitting and selection.
2. Leave out each remaining source fold in turn. Fit StandardScaler on the
   inner-training features and transform its labels within each assay to
   `(average_rank - 1) / (n - 1)`; a singleton receives 0.5.
3. Fit Ridge for each declared alpha. Select the largest mean **per-assay**
   validation Spearman across the source validation assays, not a correlation
   pooled across assays. Ties within `1e-12` of the true maximum choose the
   smaller alpha.
4. Refit scaler and readout on all permitted source rows, then predict the outer
   evaluation rows. Also fit the fixed-alpha control on those same source rows.
5. Score only after predictions are fixed. Target labels never enter fitting,
   scaling, label normalization, or alpha selection for their outer fold.

At least two source folds with usable validation assays are needed for each
outer fit. Thus a task generally needs at least three populated folds. Constant
source validation labels or predictions produce an error rather than changing
the selection population silently. Inspect alphas selected at the grid boundary;
this does not prove the optimum is inside the tested range. Choosing a new grid
from target performance makes subsequent evaluation exploratory.

Ridge fits use equal row weights; task-level reporting gives each assay equal
weight. Larger source assays can therefore influence fitting more strongly.
No fitted models are saved: outputs are diagnostic predictions and tables.

To test cross-task transfer, add both `--source-task Binding --target-task
Stability` to the command. Task names must match your metadata. The entire
target outer fold is still excluded, even from the different source task.
Hyperparameters are selected using source-task labels only. A positive target
correlation and a positive increment over a fixed prior are separate questions.
This command does not reproduce the study's cross-task permutation or
multiple-comparison calibration.

For a shuffled-source-label control, repeat a prespecified run with
`--shuffle-source-labels --seed 1` and a new output directory. Labels are shuffled
within each source assay before source selection and fitting; held-out labels
remain unchanged. One shuffled run is a control realization, not a null interval
or p-value. Use a prespecified collection of seeds if constructing a null
distribution, without choosing seeds based on their results.

## Embedding shift and source calibration

`shift` compares each held-out assay with the same-task source pool outside its
fold. For calibration, each source assay is compared with the source pool after
excluding that assay's **entire supercluster**. It uses only embeddings and
metadata; the score column can be absent.

By default, a seeded Gaussian projection maps features to 128 dimensions. Use
`--projection-dim 0` to keep the full space. No data-fitted scaling is applied.
The same projection is used throughout a run. Each side of a distance comparison
is uniformly sampled without replacement to at most `--max-samples` rows
(default 1,000); source-pool sampling is row-weighted, not assay-weighted.

The same sampled rows also produce centroid and covariance distances:
`||mean(X) - mean(Y)||₂ / sqrt(d)` and `||cov(X) - cov(Y)||F / sqrt(d)`,
using sample covariances (`n - 1` denominator) in the declared full or projected
coordinates. They are not standardized effect sizes. For large feature spaces
the covariance norm uses an equivalent Gram-matrix calculation to avoid D²
storage. Negative squared values and cancellation within a relative floating-
point roundoff tolerance are set to zero.

The statistic is biased RBF **MMD squared**, including within-sample diagonals,
averaged over squared-bandwidth multipliers `0.5, 1, 2, 4`. The base squared
bandwidth is the median positive squared pairwise distance in the two sampled
sets, recomputed for each comparison. If all distances vanish, the base is 1.

`held_out.csv` reports a midrank percentile relative to the finite source-assay
reference values in the same task and outer fold. A percentile near 50 means
the observed distance is near the center of this descriptive reference; it is
**not a null-test probability**. Correlated references, different pool sizes and
adaptive bandwidths limit inference. Global distance does not identify or rule
out task-relevant covariate shift, and cannot assign a cause to a transfer gap.
Keep representation, projection, sample cap and population fixed for comparisons.

## Variant-library composition

`composition` uses metadata only, restricted to canonical nonsynonymous single
substitutions. It reports per-assay site coverage (`observed positions / length`)
and substitution coverage (`unique single substitutions / (19 × length)`).
Coverage refers to the supplied rows. To distinguish original-library coverage
from a retained row cap, run it separately on full metadata and retained metadata
with explicit scopes; do not call a capped table the full library.

For each target assay, the source reference averages normalized per-assay
histograms over same-task assays outside the target fold. Each source assay has
equal mass. Outputs include base-2 Jensen–Shannon divergence for the 380 WT→mutant
types and relative-position deciles, plus target substitution mass absent from
the source. Position deciles use `floor(10 × (position - 1) / length)`.

These deciles are not aligned homologous sites. Zero unsupported substitution
mass means marginal substitution types occur somewhere in the source; it does
not establish matching sequence contexts, interfaces or measured effects.
No source reweighting, effect prediction, or causal decomposition is performed.

## Explicit target-support position diagnostic

`support` requires `--allow-target-support` and its own population scope.
Default support budget is 512 labels per arm, seeds are `0 1 2 3 4`, and the
alpha grid is the same nine-value grid as `probe`. Use smaller budgets only as a
separately declared design, as in the synthetic example.

For each assay and seed, the tool selects one third of observed positions and
approximately half the variants at each selected position with at least two
variants for evaluation. It then samples position-disjoint support. The overlap
arm replaces support rows with other variants at actual evaluation positions,
preserving the support count and exact WT→mutant type histogram. Evaluation
variants themselves are never support rows. The overlap arm is enriched for
evaluation positions; complete overlap is not guaranteed.

Eligibility is determined from metadata before fitting: the full requested
support budget, at least 30 evaluation variants, four evaluation positions, ten
matched replacements, and three distinct support positions in each arm. An
assay must pass every declared seed. Inspect `eligibility.csv` and
`support_splits.csv`; an empty eligible population is reported explicitly and
produces no predictions. Do not adjust eligibility using measured outcomes.

Both arms report the fixed-alpha primary result **and** an independently tuned
readout. Tuning uses three position-grouped inner folds within support only,
reranking training labels in each inner fit. This diagnostic preserves the
support producer's `average_rank / n` convention, distinct from `probe`'s
endpoint-scaled ranks. It selects pooled support out-of-fold Spearman and breaks
ties toward larger alpha. Scaling is support-fitted; evaluation labels are used
only for the final metrics. The input CSV is read into memory, so this is a
computational label boundary rather than a file-access isolation guarantee.

Metrics average seeds within each assay, then assays within each task. Undefined
seed metrics make that assay/model mean undefined. Paired summaries report
overlap-minus-disjoint and selected-minus-fixed contrasts on complete paired
seeds before averaging and supercluster resampling. Optional
`--source-predictions` with `--source-label-protocol no-target-label` joins a
frozen source predictor, and `--baseline-column` supplies a fixed metadata prior.
Comparators must cover every evaluation key (and seed, if provided). Both are
scored on exactly the local models' held-out rows; no source refit is performed.
These target-support comparisons remain distinct from no-target-label results.
The intervals condition on predictions and splits and omit support resampling
and model-selection uncertainty. See the [paired recipes](RECIPES.md#paired-multi-seed-contrasts).

Accessible local signal is evidence about the declared support design. It does
not establish label-free transfer, deployable gain without support labels, or a
unique biological mechanism. This tool's eligible subset is not automatically
the study's five-assay subset, historical Binding11, all-13, curated-8, or
MSA-valid Binding population.

## Exact-background context agreement

`context` is descriptive and fits no predictor. Candidate pairs share a protein
ID or the exact normalized full background sequence. A pair is eligible only
when its full backgrounds match, its variants validate against those backgrounds,
and it shares at least 100 variants and 20 positions by default. Case and
whitespace are normalized for sequences; mutation identifiers remain exact.
Different backgrounds are excluded even when protein identifiers match.

Signed Spearman is computed on identical shared substitutions. The 95% interval
resamples **positions**, carrying all variants at each selected position together
and reranking after each resample (default 2,000 draws). Any undefined bootstrap
draw suppresses that pair's interval and is disclosed in the finite-draw count.
No cross-pair/task aggregate or significance test is produced. Position-block
intervals do not account for dependence between different positions, shared raw
measurements between assays, or independent-protein uncertainty.

Different measured rankings may reflect assay context, measurement noise,
processing or dynamic range. Without independent replicate evidence, this
analysis cannot identify an irreducible context effect or explain the size of a
prediction gap. Ensure score directions are meaningful before comparing assays.

## Matched comparisons and uncertainty

`compare` takes the metadata and two CSVs with `assay_id`, `mutant`, and a numeric
prediction column (default `prediction`). Change columns with `--column-a` and
`--column-b`. Input key populations must match exactly by default. Use
`--allow-intersection` only for an explicitly reduced population; all excluded
keys are written to `excluded_keys.csv`. Correlations are recomputed on the
common variants, rather than comparing scores from different row populations.

Each per-assay delta is `Spearman(A, score) - Spearman(B, score)`. Task estimates
average finite paired assay deltas, with included/undefined counts disclosed.
Intervals resample operational superclusters with replacement and retain every
assay in each sampled cluster, including multiplicity. Each draw recomputes the
assay-macro mean, so this is not a cluster-equal point estimate. Fewer than two
clusters yield no interval; very small cluster counts remain descriptive.

The 95% percentile intervals condition on the already fitted predictions and
their chosen models. They omit source refitting, hyperparameter selection,
support sampling, and dependence caused by overlapping training folds. There is
no multiplicity correction across tasks, interfaces or comparisons. An interval
containing zero is unresolved evidence, not equivalence. Positive standalone
performance is not evidence of improvement over a fixed prior.

`--label-protocol` is mandatory. Existing diagnostic tags cannot be relabeled as
`no-target-label`. For externally generated untagged predictions, the declaration
is your responsibility; inspecting predictions alone cannot certify their
training history.

## Outputs, provenance and interpretation

Every analysis creates a new output directory. Existing directories are refused
even if empty; partially failed runs keep a `started` manifest rather than a
false completion marker. Inspect `manifest.json` for `status: complete` before
using results. Inputs and source modules are SHA256-hashed; arguments, population
scope, representation description, package versions, label protocol, and output
hashes are recorded. Preserve manifests alongside results.

| Command | Main output tables |
|---|---|
| `probe` | `predictions.csv`, `source_selection.csv`, `per_assay.csv`, `source_scores.csv`, `summary.csv` |
| `residual` | `predictions.csv`, `source_selection.csv`, `source_candidate_scores.csv`, `selected_hyperparameters.csv`, `per_assay.csv`, `summary.csv`, `contrast_per_assay.csv`, `contrasts.csv` |
| `diagnose` | `oracle_splits.csv`, `eligibility.csv`, `predictions.csv`, `per_seed.csv`, `per_assay.csv`, `summary.csv`, `paired_per_assay.csv`, `paired_summary.csv`; `source_selection.csv` in nested mode |
| `cosine` | `pairwise_cosine.csv`, `probe_metadata.csv`, `probe_weights.npz`, `summary.csv` |
| `shift` | `held_out.csv`, `source_reference.csv`, `summary.csv` |
| `composition` | `coverage.csv`, `per_assay.csv`, `summary.csv` |
| `context` | `candidate_pairs.csv`, `pair_concordance.csv` |
| `support` | `eligibility.csv`, `support_splits.csv`, `support_selection.csv`, and, if eligible, predictions, per-seed/per-assay summaries and `paired_summary.csv`/`paired_per_assay.csv` |
| `paired-seeds` | `population.csv`, `per_seed.csv`, `per_assay.csv`, `summary.csv` |
| `cross-task` | `predictions.csv`, `source_selection.csv`, `per_assay.csv`, `summary.csv`, `rho_matrix.csv`, `increment_matrix.csv`, `matrix_columns.csv` |
| `calibrate-transfer` | `calibration.csv`, `per_assay.csv`, `input_runs.csv`, `population_differences.csv` |
| `compare` | `excluded_keys.csv`, `per_assay.csv`, `summary.csv` |

The [controlled-comparison recipes](RECIPES.md#prior-anchored-residual-controls)
specify the residual estimator, complete source-only selection population,
prior provenance and rank universe. The
[joint-family recipe](RECIPES.md#joint-family-transfer-calibration) specifies
Holm ranking tests, one-sided Bonferroni conditional bounds, family/threshold
declarations and simulation-resolution limits. Ordinary `compare` intervals
are not retroactively made simultaneous by either addition.

Scope and label protocol are attached to analysis tables. Blank numerical CSV
cells mean undefined values, not zero. Assays with fewer than three rows,
constant effects or constant predictions have undefined Spearman. Reporting
must retain valid-assay counts and exclusions. A method with fewer valid assays
should not receive a headline comparison from unmatched task summaries.

For Binding, explicitly name the population and label access. The original
study distinguishes all-13 no-target-label Binding, curated-8 few-shot Binding,
MSA-valid Binding, historical target-support populations, and a separate local
position-support subset. They are different estimands. Custom user populations
should use their own explicit names and counts. The package does not infer
those memberships from the word `Binding` or from a coincidental assay count.

Low embedding distance, a source/support gap, composition differences and
imperfect assay agreement are complementary observations. These tools do not
add them into an attributable fraction of transfer loss or infer a dominant
mechanism. The `diagnose` source/oracle coefficient cosine and the `cosine` fold-local
comparison are separate descriptive geometries. Neither is a calibrated mapping-shift
test. SAE feature mechanism claims remain outside this portable package.

## Troubleshooting

| Message / observation | What to check |
|---|---|
| Embedding population mismatch | Export keys from the actual feature row order; explicitly subset both inputs to the intended population |
| Duplicate keys | Resolve replicate policy or select one model/seed; do not arbitrarily keep the first row |
| Protein or cluster spans folds | Correct the split upstream; never fix it by hiding group identifiers |
| Unknown assay ID when attaching split | Confirm the release and exact identifier; new assays require an explicit new split policy |
| Nonfinite score or embedding | Define and document exclusions upstream, applying matching keys to all compared interfaces |
| Undefined source validation Spearman | Inspect assay size, constant labels, and constant features/predictions; revise eligibility before inspecting target performance |
| No eligible support assays | Inspect label-free support coverage; a smaller declared budget defines a different protocol |
| Existing output directory | Select a new run directory; completed or partial results are not overwritten |
| Memory/time pressure | Predefine a smaller population, fewer candidate alphas, or an explicit feature interface; reduce MMD `--max-samples`; disclose these changes |

## Repository scope and reuse

Only the Python analysis package, synthetic-test code, installation metadata,
documentation and frozen split belong in this repository. Generated outputs and
embedding files are ignored by Git. Obtain biological inputs under their source
terms; the split does not redistribute ProteinGym effect measurements.

No new code license is asserted by this extraction. The repository owner should
choose an appropriate license before presenting it as an open-source release.
For attribution and the source-study lineage, see [PROVENANCE.md](../PROVENANCE.md).
