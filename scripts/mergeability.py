"""Can two contexts be pooled? Agreement of the effect vectors of the targets they share, against the agreement
each context has with itself.

    pixi run python scripts/mergeability.py [--summaries data] [--out results/mergeability] [--min-shared 20]

Effects are the split-0 half deltas of mean log1p CP10k against the context's own controls (pert/half_delta_A/B),
so within- and cross-context agreement use one statistic and one gene set: per pair, the genes measured in both
and above 5 mean CPM in both control pools, minus the target's own gene. Per pair and shared target:

  r              Pearson of the full effect vectors (mean of the two halves)
  r_centered     the same after removing each context's mean response over the shared targets
  rel_a, rel_b   each context's own reliability on the same genes: Spearman-Brown of the half-vs-half Pearson
  r_disatt       r / sqrt(rel_a rel_b): about 1 when the two contexts measure the same effect up to noise
  slope          Deming slope of b on a, error variance ratio from the halves (var(dA - dB) / 4 per context)
  sign_agree     share of genes significant in both (BH per target at 0.05 on the summary's p) with equal sign
  r_batchmatched Pearson of the batch-matched log2 fold changes on the genes significant in either context

Targets are binned by effect strength in the weaker context (its significant genes). r_disatt and slope are
summarized only over targets with rel >= 0.3 in both contexts (with less, noise in the denominator dominates).
Writes pairs.csv (pair x target) and pair_summary.csv (medians per pair and strength bin).
"""
import argparse
import itertools
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
GATE, ALPHA = 5.0, 0.05
BINS = [0, 10, 100, 1000, np.inf]
MIN_REL = 0.3


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def load(path, targets):
    """Rows of the requested targets (in that order) and the context-level vectors."""
    with h5py.File(path) as h:
        all_t = _s(h["pert/targets"][:])
        pos = pd.Series(np.arange(len(all_t)), index=all_t)[targets].to_numpy()
        order = np.argsort(pos)
        back = np.argsort(order)
        rows = pos[order]
        get = lambda k: h[k][rows][back].astype(np.float64)     # h5py fancy index wants increasing rows
        return dict(genes=_s(h["genes"][:]), measured=h["measured"][:].astype(bool),
                    gated=h["control/mean_cpm"][:] > GATE, dA=get("pert/half_delta_A"), dB=get("pert/half_delta_B"),
                    nlp=get("pert/nlog10p"), lfc_bm=get("pert/lfc_batchmatched") if "pert/lfc_batchmatched" in h
                    else None)


def _pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def _sb(r):
    return 2 * r / (1 + r) if np.isfinite(r) and r > -1 else np.nan


def deming(x, y, lam):
    """Slope of y on x with error variance ratio lam = var(err_y) / var(err_x)."""
    sxx, syy = np.var(x), np.var(y)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    if sxy == 0:
        return np.nan
    return float((syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy))


def sig(nlp, cols):
    out = np.zeros(nlp.shape, bool)
    for i in range(len(nlp)):
        q = false_discovery_control(np.clip(10 ** -nlp[i, cols], 0, 1))
        out[i, cols[q < ALPHA]] = True
    return out


def compare(name_a, name_b, A, B, targets):
    genes = A["genes"]
    keep = A["measured"] & B["measured"] & A["gated"] & B["gated"]
    cols = np.flatnonzero(keep)
    gi = pd.Series(np.arange(len(genes)), index=genes)
    sa, sb = sig(A["nlp"], cols), sig(B["nlp"], cols)
    fa, fb = (A["dA"] + A["dB"]) / 2, (B["dA"] + B["dB"]) / 2
    ca = fa[:, cols] - np.nanmean(fa[:, cols], 0)
    cb = fb[:, cols] - np.nanmean(fb[:, cols], 0)
    rows = []
    for i, t in enumerate(targets):
        m = np.ones(len(cols), bool)
        if t in gi.index and keep[gi[t]]:
            m[np.searchsorted(cols, gi[t])] = False
        c = cols[m]
        if not (np.isfinite(fa[i, c]).all() and np.isfinite(fb[i, c]).all()):
            continue
        ra = _sb(_pearson(A["dA"][i, c], A["dB"][i, c]))
        rb = _sb(_pearson(B["dA"][i, c], B["dB"][i, c]))
        r = _pearson(fa[i, c], fb[i, c])
        lam = np.var(B["dA"][i, c] - B["dB"][i, c]) / max(np.var(A["dA"][i, c] - A["dB"][i, c]), 1e-30)
        both = sa[i, c] & sb[i, c]
        either = sa[i, c] | sb[i, c]
        rbm = np.nan
        if A["lfc_bm"] is not None and B["lfc_bm"] is not None and either.sum() >= 10:
            x, y = A["lfc_bm"][i, c][either], B["lfc_bm"][i, c][either]
            ok = np.isfinite(x) & np.isfinite(y)
            rbm = _pearson(x[ok], y[ok]) if ok.sum() >= 10 else np.nan
        rows.append(dict(a=name_a, b=name_b, target=t, n_genes=len(c), n_sig_min=int(min(sa[i, c].sum(),
                         sb[i, c].sum())), n_sig_both=int(both.sum()),
                         r=r, r_centered=_pearson(ca[i, m], cb[i, m]), rel_a=ra, rel_b=rb,
                         r_disatt=r / np.sqrt(ra * rb) if ra > 0 and rb > 0 else np.nan,
                         slope=deming(fa[i, c], fb[i, c], lam),
                         sign_agree=float((np.sign(fa[i, c][both]) == np.sign(fb[i, c][both])).mean())
                         if both.any() else np.nan,
                         r_batchmatched=rbm))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/mergeability"))
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--contexts", nargs="*", help="restrict to these contexts")
    a = ap.parse_args()
    files = {f.stem: f for f in sorted(Path(a.summaries).glob("*.h5")) if f.with_suffix(".json").exists()
             and (not a.contexts or f.stem in a.contexts)}
    tg = {}
    for c, f in files.items():
        with h5py.File(f) as h:
            if "pert/half_delta_A" in h:
                tg[c] = set(_s(h["pert/targets"][:]))
    rows = []
    for ca, cb in itertools.combinations(sorted(tg), 2):
        shared = sorted(tg[ca] & tg[cb])
        if len(shared) < a.min_shared:
            continue
        A, B = load(files[ca], shared), load(files[cb], shared)
        res = compare(ca, cb, A, B, shared)
        rows += res
        print(f"{ca} vs {cb}: {len(shared)} shared targets, {len(res)} compared", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pairs.csv", index=False)
    df["strength"] = pd.cut(df.n_sig_min, BINS, right=False,
                            labels=["<10 DE", "10-99 DE", "100-999 DE", ">=1000 DE"])
    weak = (df.rel_a < MIN_REL) | (df.rel_b < MIN_REL)
    df["n_reliable"] = (~weak).astype(int)
    df.loc[weak, ["r_disatt", "slope"]] = np.nan
    cols = ["r", "r_centered", "rel_a", "rel_b", "r_disatt", "slope", "sign_agree", "r_batchmatched"]
    allbins = df.groupby(["a", "b"])[cols].median().assign(n_targets=df.groupby(["a", "b"]).size(),
                                                             n_reliable=df.groupby(["a", "b"]).n_reliable.sum())
    allbins["strength"] = "all"
    byb = df.groupby(["a", "b", "strength"], observed=True)[cols].median()
    byb["n_targets"] = df.groupby(["a", "b", "strength"], observed=True).size()
    byb["n_reliable"] = df.groupby(["a", "b", "strength"], observed=True).n_reliable.sum()
    summ = pd.concat([allbins.reset_index(), byb.reset_index()]).sort_values(["a", "b", "strength"])
    summ.to_csv(out / "pair_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(allbins.round(3).sort_values("r_disatt", ascending=False).to_string())


if __name__ == "__main__":
    main()
