"""Held-out predictions for the H1 evaluator, for questions such as the bulk gain (AUDIT A1) and the scale check (T6).

    pixi run python scripts/h1_heldout_predictions.py --seed 0 --out results/h1_heldout

Each target's H1 cells are split at random into halves A and B (seeded). Writes to <out>/seed<seed>/:

  half_oracle.npz    each target's log2 fold change measured on its half-A cells against half of the control cells
                     (log2 of mean CPM over mean CPM, eps 1e-9; 0 for genes with no counts in that control
                     half): a strong model with real sampling noise
  mean_response.npz  the mean of those fold changes over all 300 targets, the same row for every target: what the
                     scorer's baseline predicts
  holdout.npz        per target, the half-A rows of its source file; `evaluate_h1.py --holdout` never samples them as
                     the reference, so prediction and truth share no target cells (they share the other control half
                     only through the evaluator's control fit, which uses all controls)
"""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CTRL, EPS = "non-targeting", 1e-9


def mean_cpm(X):
    X = X.tocsr().astype(np.float64)
    L = np.asarray(X.sum(1)).ravel()
    return np.asarray(X.multiply(1e6 / L[:, None]).mean(0)).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/h1_heldout"))
    a = ap.parse_args()
    rng = np.random.default_rng([a.seed, 777])
    paths = [ROOT / f"data/vcc2025/adata_{s}.h5ad" for s in ("Training", "Validation", "Test")]
    srcs = [ad.read_h5ad(p, backed="r") for p in paths]
    genes = srcs[0].var_names.to_numpy().astype(str)
    labels0 = srcs[0].obs["target_gene"].astype(str).to_numpy()
    ctrl = np.flatnonzero(labels0 == CTRL)
    ctrl_a = np.sort(rng.permutation(ctrl)[: len(ctrl) // 2])
    c_mean = mean_cpm(srcs[0].X[ctrl_a])
    targets, lfc, hold = [], [], {}
    for s in srcs:
        lab = s.obs["target_gene"].astype(str).to_numpy()
        for t in sorted(set(lab) - {CTRL}):
            rows = np.flatnonzero(lab == t)
            half = np.sort(rng.permutation(rows)[: len(rows) // 2])
            targets.append(t)
            # A gene absent from the control half has no measurable change; its ratio to eps would be about 2^32.
            lfc.append(np.where(c_mean > 0, np.log2((mean_cpm(s.X[half]) + EPS) / (c_mean + EPS)), 0))
            hold[t] = half
    order = np.argsort(targets)
    targets, lfc = np.array(targets)[order], np.array(lfc, np.float32)[order]
    assert len(targets) == 300 and len(set(targets)) == 300
    out = Path(a.out) / f"seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "half_oracle.npz", genes=genes, perts=targets, lfc=lfc,
                        source=f"half-split oracle, seed {a.seed}: half-A H1 cells vs half of the H1 controls")
    np.savez_compressed(out / "mean_response.npz", genes=genes, perts=targets,
                        lfc=np.repeat(lfc.mean(0, keepdims=True), len(targets), 0),
                        source=f"mean response over the 300 half-split oracle rows, seed {a.seed}")
    np.savez_compressed(out / "holdout.npz", **hold)
    print(f"wrote {out}: {len(targets)} targets, median half size {int(np.median([len(v) for v in hold.values()]))}")
    for s in srcs:
        s.file.close()


if __name__ == "__main__":
    main()
