"""Known-answer check of one context's summary (AUDIT T2, T3, A6): the same cells, read with the same reader, gene
map and label rule as src/clean/run.py, recomputed (T2) with dense NumPy/SciPy and (T3) with cell-eval2's own DE
call at the vcc2026 preset (pdex backend, CPU). Compares with data/<context>.h5.

    sbatch ... --wrap "bash scripts/run_job.sh pixi run python -u tests/known_answer.py --context adamson__k562"

T2: mean CPM, log2 fold change and the two-sided Mann-Whitney p (scipy, asymptotic, continuity and tie corrected,
on log1p CPM) of every target and measured gene. T3: cell-eval2 compute_de on the same cells; the significant set
per target (5 CPM gate on the controls, BH per target at 0.05) from the summary's nlog10p must equal cell-eval2's.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml
from scipy.stats import false_discovery_control, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/clean")]
import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import pandas as pd  # noqa: E402
from genes import GeneMap, load_vocab  # noqa: E402
from labels import TargetLabels  # noqa: E402
from readers import READERS  # noqa: E402
from stats import MIN_UMI  # noqa: E402

EPS, GATE, ALPHA = 1e-9, 5.0, 0.05


def cells(ctx):
    """(counts [n, measured genes] csr, label per cell: target name or 'non-targeting', measured vocab positions)."""
    cfg = yaml.safe_load(open(ROOT / "workflow/config.yaml"))["contexts"][ctx]
    vocab = load_vocab()
    src = READERS[cfg["reader"]](tmp=ROOT / "data/raw/_clean_tmp" / ctx, **cfg.get("args", {}))
    if hasattr(src, "run_blocks") or hasattr(src, "target_symbol"):
        raise SystemExit(f"{ctx}: gene-major and TSS-group readers are not covered by this check")
    gm = GeneMap(src.genes, src.ensembl, vocab, label=ctx)
    tl = TargetLabels(vocab, gm, src.genes)
    min_umi = cfg.get("min_umi", MIN_UMI)
    Xs, labs = [], []
    for X, _ in src.controls():
        Xm = gm.apply(X)
        ok = np.asarray(Xm.sum(1)).ravel() >= min_umi
        Xs.append(Xm[np.flatnonzero(ok)])
        labs.append(np.full(ok.sum(), "non-targeting", dtype=object))
    for X, lab, _ in src.cells():
        Xm = gm.apply(X)
        name = np.array([tl(t)[0] for t in lab], dtype=object)
        ok = (np.asarray(Xm.sum(1)).ravel() >= min_umi) & (name != "")
        Xs.append(Xm[np.flatnonzero(ok)])
        labs.append(name[ok])
    return sp.vstack(Xs, format="csr"), np.concatenate(labs), gm.measured_idx, vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="adamson__k562")
    a = ap.parse_args()
    X, lab, midx, vocab = cells(a.context)
    with h5py.File(ROOT / f"data/{a.context}.h5") as h:
        targets = np.array([t.decode() if isinstance(t, bytes) else t for t in h["pert/targets"][:]], dtype=object)
        lfc_s = h["pert/lfc"][:][:, midx]
        nlp_s = h["pert/nlog10p"][:][:, midx]
        cpm_s = h["pert/mean_cpm"][:][:, midx]
        ctrl_s = h["control/mean_cpm"][:][midx]
    L = np.asarray(X.sum(1)).ravel()
    cpm = sp.diags(1e6 / L) @ X
    ctrl = lab == "non-targeting"
    C = cpm[ctrl].toarray()
    c_mean = C.mean(0)
    logC = np.log1p(C)
    gated = c_mean > GATE
    out = dict(context=a.context, n_cells=int(len(lab)), n_controls=int(ctrl.sum()), n_targets=len(targets),
               n_genes=len(midx), max_abs_diff={})
    d = out["max_abs_diff"]
    d["control_mean_cpm_rel"] = float(np.max(np.abs(ctrl_s - c_mean) / np.maximum(c_mean, 1e-3)))
    worst_cpm = worst_lfc = worst_p = 0.0
    p_ind = np.full(lfc_s.shape, np.nan)
    for i, t in enumerate(targets):
        P = cpm[lab == t].toarray()
        m = P.mean(0)
        lfc = np.log2((m + EPS) / (c_mean + EPS))
        worst_cpm = max(worst_cpm, float(np.max(np.abs(cpm_s[i] - m) / np.maximum(m, 1e-3))))
        worst_lfc = max(worst_lfc, float(np.nanmax(np.abs(lfc_s[i] - lfc))))
        p = mannwhitneyu(np.log1p(P), logC, axis=0, alternative="two-sided", method="asymptotic",
                         use_continuity=True).pvalue
        p = np.where(np.isnan(p), 1.0, p)                   # all-zero gene in both groups
        p_ind[i] = p
        fin = (p > 1e-290) & (nlp_s[i] < 290)            # both sides below double-precision underflow
        worst_p = max(worst_p, float(np.max(np.abs(-np.log10(p[fin]) - nlp_s[i][fin]), initial=0)))
    d.update(mean_cpm_rel=worst_cpm, lfc=worst_lfc, nlog10p_vs_scipy=worst_p)

    # T3: cell-eval2's own DE (vcc2026 preset values) on the same counts
    from cell_eval2.de_compute import compute_de
    adata = ad.AnnData(X.astype(np.float32), obs=pd.DataFrame({"target": lab}),
                       var=pd.DataFrame(index=vocab[midx]))
    de = compute_de(adata, backend="pdex", groupby="target", reference="non-targeting", mean_calc="arithmetic",
                    epsilon=EPS, input_type="counts", target_sum=1e6, filter_gene_min_cpm_cell=GATE,
                    fdr_scope="per_pert", threads=4, device="cpu").to_pandas()
    gi = pd.Series(np.arange(len(midx)), index=vocab[midx])
    agree = dict(targets=0, sig_pipeline=0, sig_celleval=0, sig_both=0, max_abs_nlog10p_vs_celleval=0.0,
                 max_abs_lfc_vs_celleval=0.0)
    for i, t in enumerate(targets):
        r = de[de.target == t]
        if r.empty:
            continue
        agree["targets"] += 1
        col = gi[r.feature.to_numpy()].to_numpy()
        pc = r.p_value.to_numpy()
        fin = (pc > 1e-290) & (nlp_s[i, col] < 290)
        agree["max_abs_nlog10p_vs_celleval"] = max(agree["max_abs_nlog10p_vs_celleval"], float(np.max(np.abs(
            -np.log10(pc[fin]) - nlp_s[i, col][fin]), initial=0)))
        lc = r.filter(like="fold_change").iloc[:, 0].to_numpy()
        ok = np.isfinite(lc)
        agree["max_abs_lfc_vs_celleval"] = max(agree["max_abs_lfc_vs_celleval"],
                                               float(np.max(np.abs(lc[ok] - lfc_s[i, col][ok]), initial=0)))
        sig_c = set(r.feature[r.p_adj.to_numpy() < ALPHA])
        g = np.flatnonzero(gated)
        q = false_discovery_control(np.clip(10 ** -nlp_s[i, g].astype(np.float64), 0, 1))
        sig_p = set(vocab[midx][g[q < ALPHA]])
        agree["sig_pipeline"] += len(sig_p)
        agree["sig_celleval"] += len(sig_c)
        agree["sig_both"] += len(sig_p & sig_c)
    out["celleval_de"] = agree
    out["de_columns"] = list(de.columns)
    print(json.dumps(out, indent=2))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/known_answer_{a.context}.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
