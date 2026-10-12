"""Pairwise agreement of shared targets' effects between contexts, against each context's own reliability."""
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
N_BOOT = 1000
LN2 = np.log(2.0)


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def load(path, targets, effects="auto"):
    """Rows of the requested targets (in that order) and the context-level vectors."""
    with h5py.File(path) as h:
        has_bm = "pert/half_delta_bm_A" in h
        if effects == "batchmatched" and not has_bm:
            raise ValueError(f"{path.stem} has no batch-matched half deltas")
        bm = "_bm" if has_bm and effects != "pooled" else ""
        all_t = _s(h["pert/targets"][:])
        pos = pd.Series(np.arange(len(all_t)), index=all_t)[targets].to_numpy()
        order = np.argsort(pos)
        back = np.argsort(order)
        rows = pos[order]
        get = lambda k: h[k][rows][back].astype(np.float64)
        return dict(genes=_s(h["genes"][:]), measured=h["measured"][:].astype(bool),
                    gated=h["control/mean_cpm"][:] > GATE, dA=get(f"pert/half_delta{bm}_A"), dB=get(f"pert/half_delta{bm}_B"),
                    effects="batch-matched" if bm else "pooled",
                    nlp=get("pert/nlog10p"), lfc_bm=get("pert/lfc_batchmatched") if "pert/lfc_batchmatched" in h
                    else None, lfc=get("pert/lfc_batchmatched" if bm else "pert/lfc"), mean=get("pert/mean_cpm"),
                    se=get("pert/se_cpm"), c_mean=h["control/mean_cpm"][:].astype(np.float64),
                    c_se=h["control/se_cpm"][:].astype(np.float64), depth=float(h["control/depth_median"][()]))


def lfc_err_var(X, i, c):
    """Sampling variance of each gene's log2 fold change: delta method on the two mean CPMs."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (X["se"][i, c] / (X["mean"][i, c] * LN2)) ** 2 + (X["c_se"][c] / (X["c_mean"][c] * LN2)) ** 2


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
        slope_lfc = np.nan
        g = c[both]
        if len(g) >= 10:
            va, vb = lfc_err_var(A, i, g), lfc_err_var(B, i, g)
            x, y = A["lfc"][i, g], B["lfc"][i, g]
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(va) & np.isfinite(vb)
            if ok.sum() >= 10:
                slope_lfc = deming(x[ok], y[ok], vb[ok].mean() / max(va[ok].mean(), 1e-30))
        rows.append(dict(a=name_a, b=name_b, target=t, effects=f"{A['effects']}/{B['effects']}", n_genes=len(c), n_sig_min=int(min(sa[i, c].sum(),
                         sb[i, c].sum())), n_sig_both=int(both.sum()),
                         r=r, r_centered=_pearson(ca[i, m], cb[i, m]), rel_a=ra, rel_b=rb,
                         r_disatt=r / np.sqrt(ra * rb) if ra > 0 and rb > 0 else np.nan,
                         slope=deming(fa[i, c], fb[i, c], lam),
                         sign_agree=float((np.sign(fa[i, c][both]) == np.sign(fb[i, c][both])).mean())
                         if both.any() else np.nan,
                         r_batchmatched=rbm, slope_lfc=slope_lfc))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/mergeability"))
    ap.add_argument("--min-shared", type=int, default=20)
    ap.add_argument("--contexts", nargs="*", help="restrict to these contexts")
    ap.add_argument("--effects", choices=["auto", "pooled", "batchmatched"], default="auto")
    a = ap.parse_args()
    files = {f.stem: f for f in sorted(Path(a.summaries).glob("*.h5")) if f.with_suffix(".json").exists()
             and (not a.contexts or f.stem in a.contexts)}
    tg, depth = {}, {}
    for c, f in files.items():
        with h5py.File(f) as h:
            if "pert/half_delta_A" in h and (a.effects != "batchmatched" or "pert/half_delta_bm_A" in h):
                tg[c] = set(_s(h["pert/targets"][:]))
                depth[c] = float(h["control/depth_median"][()])
    rows = []
    for ca, cb in itertools.combinations(sorted(tg), 2):
        shared = sorted(tg[ca] & tg[cb])
        if len(shared) < a.min_shared:
            continue
        A, B = load(files[ca], shared, a.effects), load(files[cb], shared, a.effects)
        res = compare(ca, cb, A, B, shared)
        rows += res
        print(f"{ca} vs {cb}: {len(shared)} shared targets, {len(res)} compared", flush=True)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "pairs.csv", index=False)
    df["strength"] = pd.cut(df.n_sig_min, BINS, right=False,
                            labels=["<10 DE", "10-99 DE", "100-999 DE", ">=1000 DE"]).astype(str)
    weak = (df.rel_a < MIN_REL) | (df.rel_b < MIN_REL)
    df["n_reliable"] = (~weak).astype(int)
    df.loc[weak, ["r_disatt", "slope", "slope_lfc"]] = np.nan
    summ = pd.concat([summarize(df.assign(strength="all")), summarize(df)]).sort_values(["a", "b", "strength"])
    eff = df.groupby(["a", "b"]).effects.first()
    summ["effects"] = [eff[(x, y)] for x, y in zip(summ.a, summ.b)]
    summ["depth_a"], summ["depth_b"] = summ.a.map(depth), summ.b.map(depth)
    summ.to_csv(out / "pair_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summ[summ.strength == "all"].round(3).sort_values("r_disatt", ascending=False).to_string())


def boot_median(v, rng):
    """Median and bootstrap 95% interval of the median over targets (NaN-free values)."""
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return (np.median(v) if len(v) else np.nan), np.nan, np.nan
    m = np.median(v[rng.integers(0, len(v), (N_BOOT, len(v)))], 1)
    return np.median(v), *np.percentile(m, [2.5, 97.5])


def summarize(df):
    """Medians per pair and strength bin; bootstrap intervals for r_disatt and the two slopes."""
    rng = np.random.default_rng(0)
    cols = ["r", "r_centered", "rel_a", "rel_b", "sign_agree", "r_batchmatched"]
    rows = []
    for (x, y, st), g in df.groupby(["a", "b", "strength"], sort=True):
        row = dict(a=x, b=y, strength=st, **g[cols].median().to_dict(), n_targets=len(g),
                   n_reliable=int(g.n_reliable.sum()))
        for k in ("r_disatt", "slope", "slope_lfc"):
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = boot_median(g[k].to_numpy(float), rng)
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    main()
