# v0.4.0 — correctness repairs and controlled comparisons

This release fixes verified software defects and adds reusable controls missing
from v0.3. It does not regenerate the originating study's scientific results.

## Corrections and migration

- **Permutation significance:** upper-tail counts now include numerical ties
  across different floating-point correlation calculations. Earlier p-values
  could be severely too small. Re-evaluate saved predictions with `evaluate`,
  or completed task-transfer runs with `calibrate-transfer`, writing to new
  directories. Do not silently overwrite old outputs. Holm-adjusted p-values
  must be recalculated too. This fix does not require model refitting.
- **SciPy 1.9:** correlation extraction uses the compatible tuple accessor;
  the declared dependency floor is retained and tested.
- **Subgroup annotations:** reserved metric/provenance names, including `status`,
  are rejected rather than silently overwritten. Rename user annotations such
  as `status` to a distinct name like `study_status`.
- **Task matrices:** arbitrary task labels are represented by safe
  `target_0000`-style columns, with `matrix_columns.csv` recording the exact
  mapping. Update scripts that expected raw task labels as matrix headers.
  Long-form summaries keep the original labels.
- **Oracle semantics:** nested source selection no longer changes the oracle
  penalty. `--oracle-alpha` defaults to 1 independently of `--fixed-alpha` and
  the source alpha grid. The legacy `alpha` output denotes source alpha only.
  Old shared-alpha nested outputs represent a different comparison; do not
  relabel them as frozen-oracle controls.

The numerical tie rule is documented as `100 * float64_eps * max(1, abs(rho))`
for these bounded correlation statistics. This is a roundoff safeguard, not an
effect-size threshold. Within-assay exchangeability remains a scientific
assumption; a corrected p-value does not establish that assumption.

## Added tools and outputs

| Tool | Use | Main output |
|---|---|---|
| `residual` | Test embeddings' value beyond an independently fixed prior, with source-only alpha/lambda selection | `predictions.csv`, selection ledgers, `contrasts.csv` |
| `diagnose --source-selection nested` | Compare fixed and tuned source predictions against one frozen support oracle | `paired_summary.csv` and `paired_per_assay.csv` |
| `paired-seeds` | Calculate matched multi-seed differences without treating seeds as independent assays | `summary.csv`, `per_assay.csv`, `per_seed.csv` |
| `support --source-predictions ... --baseline-column ...` | Add matched source/prior controls to position-support comparisons | `paired_summary.csv` |
| `calibrate-transfer` | Joint-family ranking tests and simultaneous conditional prior-increment bounds | `calibration.csv`, `input_runs.csv`, population audit |

Start with the [runnable recipes](RECIPES.md). Prior provenance, family/threshold
prespecification and support-label permissions are required scientific choices,
not options to select after seeing evaluation results. Numerical/protocol tests
use synthetic inputs and the frozen split lookup; they do not validate new
biological conclusions. Remaining exclusions are listed in the
[coverage map](COVERAGE.md).

[Back to README](../README.md)
