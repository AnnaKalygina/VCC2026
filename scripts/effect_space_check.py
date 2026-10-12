"""Split-half agreement of context pairs recomputed in the scorer's CPM and group-sum spaces."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/clean"), str(ROOT / "scripts")]
from genes import GeneMap, load_vocab  # noqa: E402
from labels import TargetLabels  # noqa: E402
from readers import READERS, check_counts  # noqa: E402
from stats import MIN_UMI  # noqa: E402
from mergeability import MIN_REL, _pearson, _sb, boot_median, deming  # noqa: E402

PAIRS = ["vcc2025__H1:gse295214__hesc", "adamson__k562:replogle_nadig__k562",
         "replogle_nadig__k562:replogle_nadig__rpe1"]
SPACES = ("cp10k", "cpm_lfc", "bulk")
EPS, BULK, GATE = 1e-9, 5e4, 5.0


def half_sums(context, keep_targets, vocab, seed):
    """Per target half and control half: summed counts, CPM fractions, log1p CP10k, cells and depths."""
    cfg = yaml.safe_load(open(ROOT / "workflow/config.yaml"))["contexts"][context]
    src = READERS[cfg["reader"]](tmp=ROOT / "data/raw/_clean_tmp" / context, **cfg.get("args", {}))
    assert not hasattr(src, "run_blocks"), f"{context}: gene-major readers are not supported here"
    gm = GeneMap(src.genes, src.ensembl, vocab, label=context)
    tl = TargetLabels(vocab, gm, src.genes)
    min_umi = cfg.get("min_umi", MIN_UMI)
    rng = np.random.default_rng([seed, 31])
    G = len(gm.measured_idx)
    K = len(keep_targets) * 2 + 2
    acc = {k: np.zeros((K, G)) for k in ("S", "F", "Q")}
    n, Lsum = np.zeros(K), np.zeros(K)

    def add(X, L, grp):
        ok = grp >= 0
        X, L, grp = X[np.flatnonzero(ok)], L[ok], grp[ok]
        if not len(grp):
            return
        ind = sp.csr_matrix((np.ones(len(grp)), (grp, np.arange(len(grp)))), shape=(K, len(grp)))
        F = sp.diags(1 / L) @ X
        Q = F.copy()
        Q.data = np.log1p(Q.data * 1e4)
        acc["S"] += (ind @ X).toarray()
        acc["F"] += (ind @ F).toarray()
        acc["Q"] += (ind @ Q).toarray()
        n[:] += np.bincount(grp, minlength=K)
        Lsum[:] += np.bincount(grp, L, minlength=K)

    for X, _ in src.controls():
        check_counts(X, context)
        Xm = gm.apply(X).tocsr().astype(np.float64)
        L = np.asarray(Xm.sum(1)).ravel()
        grp = np.where(L >= min_umi, K - 2 + rng.integers(0, 2, len(L)), -1)
        add(Xm, L, grp)
    t_of = {}
    for raw in src.targets:
        name = tl(raw)[0]
        if name in keep_targets:
            t_of[raw] = keep_targets.index(name)
    for X, lab, _ in src.cells():
        check_counts(X, context)
        Xm = gm.apply(X).tocsr().astype(np.float64)
        L = np.asarray(Xm.sum(1)).ravel()
        t = np.array([t_of.get(x, -1) for x in lab], np.int64)
        grp = np.where((t >= 0) & (L >= min_umi), 2 * t + rng.integers(0, 2, len(L)), -1)
        add(Xm, L, grp)
    V = len(vocab)
    full = {}
    for k, a in acc.items():
        f = np.full((K, V), np.nan)
        f[:, gm.measured_idx] = a
        full[k] = f
    return dict(n=n, L=Lsum, **full)


def effects(H, space, t):
    """Half A, half B and full effect of target t in one space, on the vocab axis."""
    c = [len(H["n"]) - 2, len(H["n"]) - 1]
    g = [2 * t, 2 * t + 1]

    def eff(p, q):
        np_, nq = H["n"][p].sum(), H["n"][q].sum()
        if space == "cp10k":
            return H["Q"][p].sum(0) / np_ - H["Q"][q].sum(0) / nq
        if space == "cpm_lfc":
            return np.log2((H["F"][p].sum(0) / np_ * 1e6 + EPS) / (H["F"][q].sum(0) / nq * 1e6 + EPS))
        return np.log1p(BULK * H["S"][p].sum(0) / H["L"][p].sum()) - np.log1p(BULK * H["S"][q].sum(0) / H["L"][q].sum())

    return eff([g[0]], [c[0]]), eff([g[1]], [c[1]]), eff(g, c)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="*", default=PAIRS)
    ap.add_argument("--out", default=str(ROOT / "results/mergeability/space_check"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import h5py
    vocab = load_vocab()
    vi = {g: i for i, g in enumerate(vocab)}
    pairs = [p.split(":") for p in a.pairs]
    tg = {}
    for c in {c for p in pairs for c in p}:
        with h5py.File(ROOT / f"data/{c}.h5") as h:
            tg[c] = {x.decode() if isinstance(x, bytes) else str(x) for x in h["pert/targets"][:]}
    need = {c: sorted(set().union(*[tg[x] & tg[y] for x, y in pairs if c in (x, y)])) for c in tg}
    H = {}
    for c in sorted(need):
        print(f"{c}: reading cells for {len(need[c])} targets", flush=True)
        H[c] = half_sums(c, need[c], vocab, a.seed)
    rows = []
    for x, y in pairs:
        A, B = H[x], H[y]
        c_ok = []
        for X in (A, B):
            ctrl = X["F"][-2:].sum(0) / X["n"][-2:].sum() * 1e6
            c_ok.append(np.isfinite(ctrl) & (ctrl > GATE))
        keep = c_ok[0] & c_ok[1]
        for t in sorted(tg[x] & tg[y]):
            ia, ib = need[x].index(t), need[y].index(t)
            if min(A["n"][2 * ia:2 * ia + 2].min(), B["n"][2 * ib:2 * ib + 2].min()) < 8:
                continue
            m = keep.copy()
            if t in vi:
                m[vi[t]] = False
            for sp_ in SPACES:
                aA, aB, af = effects(A, sp_, ia)
                bA, bB, bf = effects(B, sp_, ib)
                ra, rb = _sb(_pearson(aA[m], aB[m])), _sb(_pearson(bA[m], bB[m]))
                r = _pearson(af[m], bf[m])
                lam = np.var(bA[m] - bB[m]) / max(np.var(aA[m] - aB[m]), 1e-30)
                rows.append(dict(a=x, b=y, target=t, space=sp_, n_genes=int(m.sum()), rel_a=ra, rel_b=rb, r=r,
                                 r_disatt=r / np.sqrt(ra * rb) if ra > 0 and rb > 0 else np.nan,
                                 slope=deming(af[m], bf[m], lam)))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_target.csv", index=False)
    rng = np.random.default_rng(0)
    summ = []
    for (x, y, sp_), g in df.groupby(["a", "b", "space"], sort=False):
        rel = (g.rel_a >= MIN_REL) & (g.rel_b >= MIN_REL)
        row = dict(a=x, b=y, space=sp_, n_targets=len(g), n_reliable=int(rel.sum()), rel_a=g.rel_a.median(),
                   rel_b=g.rel_b.median(), r=g.r.median(), ctrl_depth_mean_a=H[x]["L"][-2:].sum() / H[x]["n"][-2:].sum(),
                   ctrl_depth_mean_b=H[y]["L"][-2:].sum() / H[y]["n"][-2:].sum())
        for k in ("r_disatt", "slope"):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = boot_median(g[k][rel].to_numpy(float), rng)
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
