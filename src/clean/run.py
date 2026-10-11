"""One context -> data/{context}.h5 + .json (provenance, QC counts, validation checks).

    pixi run python -u src/clean/run.py --context vcc2025__H1
"""
import argparse
import os
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from genes import GeneMap, load_vocab  # noqa: E402
from labels import TargetLabels  # noqa: E402
from readers import READERS, check_counts  # noqa: E402
from stats import MIN_UMI, Engine, batch_codes, write_h5  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def git_state():
    """Commit and dirty flag of the code that produced an output (Snakemake does not rerun on code edits)."""
    import subprocess
    run = lambda *c: subprocess.run(["git", "-C", str(ROOT), *c], capture_output=True, text=True).stdout.strip()
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))


def target_index(raw, name_of):
    """Raw target labels -> (unique names, index per raw label; -1 where the label is dropped)."""
    names = np.array([name_of(t) for t in raw], dtype=object)
    keep = names != ""
    utg, inv = np.unique(names[keep].astype(str), return_inverse=True)
    idx = np.full(len(raw), -1, np.int64)
    idx[keep] = inv
    return utg, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", required=True)
    ap.add_argument("--config", default=str(ROOT / "workflow/config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        ap.error("Real statistics processing requires Slurm")
    os.chdir(ROOT)
    cfg = yaml.safe_load(open(a.config))["contexts"][a.context]
    vocab = load_vocab()
    tmp = ROOT / "data/raw/_clean_tmp" / a.context            # staged raw files; deleted by src.cleanup()
    src = READERS[cfg["reader"]](tmp=tmp, **cfg.get("args", {}))
    gm = GeneMap(src.genes, src.ensembl, vocab, label=a.context)
    tl = TargetLabels(vocab, gm, src.genes)
    # TSS-group readers (DLD1) keep their group names; their gene symbols go through the same rule below
    name_of = (lambda t: t) if hasattr(src, "target_symbol") else (lambda t: tl(t)[0])
    cells_of = {}                                         # raw label -> perturbed cells seen
    min_umi = cfg.get("min_umi", MIN_UMI)
    qc = dict(ctrl_seen=0, ctrl_low_umi=0, pert_seen=0, pert_low_umi=0, pert_unlabelled=0)

    if hasattr(src, "run_blocks"):
        utg, t_of = target_index(np.asarray(src.targets), name_of)
        lb, n = np.unique(src.lab[~src.is_ctrl], return_counts=True)
        cells_of.update(zip(lb, n))
        bl = src.batch_labels
        batches = None if bl is None else sorted(set(bl[src.is_ctrl]))
        eng = Engine(len(gm.measured_idx), len(utg), seed=a.seed, batches=batches)
        src.run_blocks(eng, gm, t_of, qc, log, min_umi)
    else:
        # pass 1: control pool (needed in full before any perturbed cell: MWU reference, control halves)
        Xs, Ls, Bs = [], [], []
        for k, (X, bat) in enumerate(src.controls()):
            if k % 5 == 0:
                log(f"  pass 1: chunk {k}, {qc['ctrl_seen']:,} controls so far")
            check_counts(X, f"{a.context} controls chunk {k}")
            Xm = gm.apply(X)
            L = np.asarray(Xm.sum(1)).ravel()
            ok = L >= min_umi
            qc["ctrl_seen"] += len(L)
            qc["ctrl_low_umi"] += int((~ok).sum())
            Xs.append(Xm[np.flatnonzero(ok)].astype(np.float32))   # integer counts: exact in float32
            Ls.append(L[ok])
            Bs.append(None if bat is None else np.asarray(bat).astype(str)[ok])
        Xc, Lc = sp.vstack(Xs, format="csr"), np.concatenate(Ls)
        assert (src.batch_key is None) == all(b is None for b in Bs), "batches must come with every chunk or none"
        Bc = None if src.batch_key is None else np.concatenate(Bs)
        batches = None if Bc is None else sorted(set(Bc))
        del Xs
        log(f"controls: {Xc.shape[0]:,} kept of {qc['ctrl_seen']:,} ({qc['ctrl_low_umi']:,} < {min_umi} UMI), "
            f"median depth {np.median(Lc):,.0f}")
        targets_raw = np.asarray(src.targets)
        utg, inv = target_index(targets_raw, name_of)
        t_of = {t: i for t, i in zip(targets_raw, inv) if i >= 0}
        eng = Engine(len(gm.measured_idx), len(utg), seed=a.seed, batches=batches)
        eng.set_controls(Xc, Lc, batch=None if Bc is None else batch_codes(Bc, batches))
        del Xc
        log(f"engine: {len(utg):,} targets x {len(gm.measured_idx):,} genes"
            + ("" if batches is None else f", {len(batches)} control batches ({src.batch_key})"))
        # pass 2: perturbed cells
        k = 0
        for X, lab, bat in src.cells():
            check_counts(X, f"{a.context} cells chunk {k}")
            lb, n = np.unique(np.asarray(lab).astype(str), return_counts=True)
            for t, c in zip(lb, n):
                cells_of[t] = cells_of.get(t, 0) + int(c)
            Xm = gm.apply(X)
            L = np.asarray(Xm.sum(1)).ravel()
            g = np.array([t_of.get(t, -1) for t in lab], np.int64)
            qc["pert_seen"] += len(L)
            qc["pert_unlabelled"] += int((g < 0).sum())
            low = L < min_umi
            qc["pert_low_umi"] += int((low & (g >= 0)).sum())
            keep = np.flatnonzero(~low & (g >= 0))
            if len(keep):
                eng.add(Xm[keep], L[keep], g[keep],
                        batch=None if batches is None else batch_codes(np.asarray(bat)[keep], batches))
            k += 1
            if k % 20 == 0:
                log(f"  pass 2: {qc['pert_seen']:,} perturbed cells")
    log(f"QC: {qc}")
    if hasattr(src, "target_symbol"):
        sym = [tl(src.target_symbol(t))[0] for t in utg]
        for t in utg:
            g = src.target_symbol(t)
            cells_of[g] = cells_of.get(g, 0) + cells_of.get(t, 0)
    else:
        sym = list(utg)
    own = gm.vocab_index(sym, vocab)                    # measured index of each target's own gene
    res = eng.result(own)
    del eng
    keep_t = res["n_cells"] > 0
    if hasattr(src, "post_result"):        # e.g. DLD1: one TSS group per gene
        new_utg, idx = src.post_result(utg, res, own)
        new_utg = np.array([tl(g)[0] for g in new_utg], dtype=object)
        keep_t = np.zeros(len(utg), bool)
        keep_t[idx[new_utg != ""]] = True
        keep_t &= res["n_cells"] > 0
        utg = np.asarray(utg, object).copy()
        utg[idx] = new_utg
    if not keep_t.all():
        print(f"WARNING: {(~keep_t).sum()} target groups dropped (0 cells after QC, or non-chosen TSS): "
              f"{list(np.asarray(utg)[~keep_t][:20])}")
        for k, v in res.items():
            if isinstance(v, np.ndarray) and v.shape[:1] == keep_t.shape:
                res[k] = v[keep_t]
        utg = utg[keep_t]

    # validation (fatal where a violation means a bug, not a data property)
    c = res["control"]
    chk = dict(ctrl_cpm_sum=float(np.nansum(c["mean_cpm"].astype(np.float64))),
               pert_cpm_sum_minmax=[float(x) for x in np.percentile(np.nansum(res["mean_cpm"], 1), [0, 100])]
               if len(utg) else None,
               n_lfc_nonfinite=int((~np.isfinite(res["lfc"])).sum()),
               n_nlog10p_nonfinite=int((~np.isfinite(res["nlog10p"])).sum()),
               n_reliable_targets=int(np.isfinite(res["reliability_mean"]).sum()),
               median_batch_shift_rms=float(np.nanmedian(res["batch_shift_rms"]))
               if np.isfinite(res.get("batch_shift_rms", np.zeros(0))).any() else None,
               median_frac_no_ctrl=float(np.nanmedian(res["frac_no_ctrl"])) if "frac_no_ctrl" in res else None,
               median_reliability_sb=float(np.nanmedian(res["reliability_sb"])) if np.isfinite(res["reliability_sb"]).any() else None)
    assert abs(chk["ctrl_cpm_sum"] - 1e6) < 1e6 * 1e-4, chk
    assert chk["n_lfc_nonfinite"] == 0 and chk["n_nlog10p_nonfinite"] == 0, chk
    log(f"checks: {chk}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    attrs = dict(dataset=a.context.split("__")[0], context=a.context, min_umi=min_umi, L_def="vocab",
                 cpm="arithmetic mean of c/L*1e6", lfc="log2((pert+1e-9)/(ctrl+1e-9)), ungated",
                 pvalue="nlog10p = -log10 two-sided MWU (pdex formula, full control pool), ungated",
                 disp="NB method of moments, unclamped", reliability="8 half-splits, log1p CP10k, Pearson, SB",
                 se_cpm="sqrt(var(CPM) / n) per gene", half_delta="split 0 half deltas of mean log1p CP10k vs control",
                 batch_key=str(src.batch_key),
                 lfc_batchmatched="log2((pert+1e-9)/(ctrl_ref+1e-9)), ctrl_ref = per-batch control mean CPM weighted "
                                  "by the target's cells per batch; batches with < 20 controls excluded; "
                                  "MWU p stays against the pooled controls",
                 seed=a.seed, source=json.dumps(src.provenance))
    tmp = out / f"{a.context}.h5.tmp"
    write_h5(tmp, res, vocab, gm.measured_idx, utg, attrs)
    tmp.rename(out / f"{a.context}.h5")
    prov = dict(context=a.context, reader=cfg["reader"], args=cfg.get("args", {}), source=src.provenance,
                code=git_state(), batch_key=src.batch_key, n_control_batches=None if "batches" not in res else len(res["batches"]),
                genes=gm.report, target_labels=tl.report(cells_of), qc=qc, n_targets=int(len(utg)), n_ctrl=int(c["n_cells"]), checks=chk,
                minutes=round((time.time() - T0) / 60, 1),
                peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    json.dump(prov, open(out / f"{a.context}.json", "w"), indent=1)
    log(f"wrote {out / a.context}.h5: {len(utg):,} targets; peak RSS {prov['peak_rss_gb']} GB")
    if hasattr(src, "cleanup"):
        src.cleanup()
        log(f"removed staged raw files of {a.context}")


if __name__ == "__main__":
    main()
