"""H1 as a 2026-style reference: 400 cells per target, 18,400 controls, thinned to a 20k control depth."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
CTRL, N_CELLS, N_CTRL, DEPTH = "non-targeting", 400, 18_400, 20_000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "data/vcc2025/h1_reference_20k.h5ad"))
    a = ap.parse_args()
    rng = np.random.default_rng([a.seed, 2026])
    hold = np.load(a.holdout) if a.holdout else None
    srcs = [ad.read_h5ad(ROOT / f"data/vcc2025/adata_{s}.h5ad", backed="r") for s in ("Training", "Validation", "Test")]
    genes = srcs[0].var_names
    parts, obs = [], []
    for k, s in enumerate(srcs):
        lab = s.obs["target_gene"].astype(str).to_numpy()
        batch = s.obs["batch"].astype(str).to_numpy()
        pick = []
        if k == 0:
            c = np.flatnonzero(lab == CTRL)
            pick.append(np.sort(rng.choice(c, N_CTRL, replace=False)))
        for t in sorted(set(lab) - {CTRL}):
            rows = np.flatnonzero(lab == t)
            if hold is not None:
                rows = np.setdiff1d(rows, hold[t])
            pick.append(rows if len(rows) <= N_CELLS else np.sort(rng.choice(rows, N_CELLS, replace=False)))
        rows = np.sort(np.concatenate(pick))
        parts.append(sp.csr_matrix(s.X[rows]))
        obs.append(pd.DataFrame(dict(target_gene=lab[rows], batch=batch[rows])))
    X = sp.vstack(parts, format="csr").astype(np.int64)
    obs = pd.concat(obs, ignore_index=True)
    if (X.data < 0).any() or not np.array_equal(X.data, np.round(X.data)):
        raise SystemExit("H1 counts are not raw integers")
    L = np.asarray(X.sum(1)).ravel()
    p = DEPTH / np.median(L[obs.target_gene.to_numpy() == CTRL])
    if p >= 1:
        raise SystemExit(f"H1 controls are already at or below {DEPTH} UMI (p = {p:.3f})")
    X.data = rng.binomial(X.data, p)
    X.eliminate_zeros()
    keep = np.asarray(X.sum(1)).ravel() > 0
    out = ad.AnnData(X[keep].astype(np.float32), obs=obs[keep].reset_index(drop=True), var=pd.DataFrame(index=genes))
    out.obs.index = out.obs.index.astype(str)
    out.uns["h1_reference"] = dict(thinning_p=float(p), seed=a.seed, holdout=str(a.holdout), n_dropped_empty=int((~keep).sum()))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(a.out)
    Lc = np.asarray(out.X[(out.obs.target_gene == CTRL).to_numpy()].sum(1)).ravel()
    print(f"wrote {a.out}: {out.n_obs:,} cells ({int((out.obs.target_gene == CTRL).sum()):,} controls, "
          f"{out.obs.target_gene.nunique() - 1} targets), thinning p {p:.4f}, median control depth {np.median(Lc):,.0f}")


if __name__ == "__main__":
    main()
