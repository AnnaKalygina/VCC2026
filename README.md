# VCC 2026

The repo currently contains a snakemake workflow for the download and extraction of the 21-context summary-statistics pipeline and an internal H1 (VCC2025) evaluation.

## Setup

```bash
pixi install --locked
pixi run smoke
pixi run dry-run
```

## Prepare H1 or process all data

```bash
# Downloads the full original Training, Validation, and Test H5AD files (~34 GB).
sbatch scripts/workflow.sbatch prepare_h1

# Generates all prerequisites and computes all 21 context summaries.
sbatch scripts/workflow.sbatch

# A single context, including its prerequisites.
sbatch scripts/workflow.sbatch data/adamson__k562.h5
```

Snakemake builds a dependency graph from the requested outputs. The default `all`
target prepares the source metadata, vocabulary and gene mapping, then runs the
summary rule for each of the 21 contexts in `workflow/config.yaml`. `prepare_h1`
downloads and validates only the three H1 splits. Existing outputs are reused;
missing or outdated outputs trigger their dependent rules. The Slurm profile submits
individual rules with their declared CPU, memory and runtime limits; per-rule logs
record progress and failures. H1 prediction evaluation is launched separately.

Summary outputs are `data/<context>.h5` and `data/<context>.json`.
The HDF5 contains the ordered VCC2026 vocabulary, measured-gene mask, control statistics,
and per-target mean CPM, LFC, NB dispersion, MWU statistics, cell/depth summaries,
and split-half reliability.
- Unmeasured genes remain NaN.
- Library sizes use measured VCC2026 vocabulary genes
- cells below 1,000 UMI are removed.

The 21 contexts cover H1, Adamson, three Frangieh conditions, two Feng screens,
DLD1, hESC, VIPerturb, two Orion lines, Replogle GWPS plus four Replogle/Nadig contexts,
KOLF, and three Marson CD4 T-cell conditions. 

Feng annotation pairs come from the authors' LFC table; effect estimates are not used.

Staged non-H1 count files are deleted after successful summary generation. KOLF,
Replogle/Nadig, and CD4 inputs are streamed through HTTP ranges. Orion shards are staged
one at a time with prefetch. Receipts in `data/provenance/` retain source URLs, sizes,
SHA-256 hashes, and available upstream MD5s. 
Failed jobs retain partial downloads for inspection/resume.

## Evaluate supplied H1 predictions

Prediction NPZ fields:

| Field | Shape | Meaning |
| --- | --- | --- |
| `genes` | `[G]` | Exact native H1 gene axis, in H5AD order |
| `perts` | `[300]` | All unique H1 target symbols; any order |
| `lfc` | `[300, G]` | Finite predicted log2 fold changes |
| `source` | scalar string | Prediction provenance |
| `lfc_bulk` | `[300, G]`, optional | Separate pooled-bulk LFC channel |

```bash
pixi run python scripts/example_predictions.py --out results/zero.npz
sbatch scripts/evaluate_h1.sbatch --predictions results/zero.npz --out results/zero_eval --seed 0
```

The example uses only gene/target labels. The evaluator uses the shared H1 controls once, scores all 300
targets, and retains reference split identity. The evaluator uses original UMI counts for controls and perturbed references, with seed 0,
up to 400 reference cells and 400 generated cells per target. Cell selection does not
alter the counts of retained cells. Predicted LFCs are not clipped; the separate bulk
channel retains its gain of 0.25. Old evaluation settings cannot reuse completed blocks.

DE runs in 12 blocks of 25 targets. PDS ranking and corrected MSE are calculated once
on the complete 300-target panel using the blocks' sufficient statistics. Outputs include
`per_pert.csv`, `agg.csv`, `score.csv`, `summary.json`, `metadata.json`, and `DONE`.
Completed blocks can resume only with identical recorded inputs/settings.

## Small verification only

`pixi run smoke` uses 24 genes, 32 controls, and 40 synthetic perturbed cells, plus tiny
metadata fixtures. It checks CPM/LFC/MWU calculations, integer count generation,
prediction validation, preservation of large LFCs, stale/incomplete runs, shared controls, and global
moment merging. `pixi run dry-run` constructs the DAG without downloads or Slurm submission.
Routine verification never processes real matrices or launches full H1 scoring.
