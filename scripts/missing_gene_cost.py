"""Cost of each context's missing panel genes, measured on H1's significant gene sets."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

import coverage_tables as ct

ROOT = Path(__file__).resolve().parents[1]
ALPHA, MIN_SIG = 0.05, 10


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/coverage"))
    ap.add_argument("--reference", default="vcc2025__H1")
    a = ap.parse_args()
    bundle = Path(a.bundle)
    man = json.loads((bundle / "manifest.json").read_text())
    out = Path(a.out) / man["panel_id"]
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(Path(a.summaries) / f"{a.reference}.h5") as h:
        genes = ct._s(h["genes"][:])
        ref_meas = h["measured"][:].astype(bool)
        targets = ct._s(h["pert/targets"][:])
        gate = h["control/mean_cpm"][:] > ct.GATE_CPM
        lfc = h["pert/lfc"][:]
        nlp = h["pert/nlog10p"][:]
        delta = (h["pert/half_delta_A"][:].astype(np.float64) + h["pert/half_delta_B"][:]) / 2
    g = np.flatnonzero(gate & ref_meas)
    sig = np.zeros(lfc.shape, bool)
    for i in range(len(targets)):
        q = false_discovery_control(np.clip(10 ** -nlp[i, g].astype(np.float64), 0, 1))
        sig[i, g[q < ALPHA]] = True
    ok_l = np.isfinite(lfc) & ref_meas[None, :]
    absl = np.where(ok_l & sig, np.abs(lfc), 0.0)
    sq = np.where(np.isfinite(delta) & ref_meas[None, :], delta, 0.0) ** 2
    n_sig = sig.sum(1)

    rows = []
    for f in sorted(Path(a.summaries).glob("*.h5")):
        if f.stem == a.reference or not f.with_suffix(".json").exists():
            continue
        with h5py.File(f) as h:
            if not (ct._s(h["genes"][:]) == genes).all():
                raise SystemExit(f"{f}: gene axis differs")
            miss = ~h["measured"][:].astype(bool)
        set_share = (sig & miss).sum(1) / np.maximum(n_sig, 1)
        mag_share = (absl * miss).sum(1) / np.maximum(absl.sum(1), 1e-12)
        norm_share = (sq * miss).sum(1) / np.maximum(sq.sum(1), 1e-12)
        rows.append(pd.DataFrame(dict(context=f.stem, h1_target=targets, n_sig=n_sig, set_share=set_share,
                                      magnitude_share=mag_share, norm_share=norm_share,
                                      n_missing_genes=int(miss.sum()))))
    per = pd.concat(rows)
    per.to_csv(out / "missing_gene_cost.csv", index=False)
    use = per[per.n_sig >= MIN_SIG]
    summ = use.groupby("context").agg(
        n_missing_genes=("n_missing_genes", "first"), n_h1_targets=("h1_target", "size"),
        set_share_median=("set_share", "median"), set_share_mean=("set_share", "mean"),
        magnitude_share_median=("magnitude_share", "median"), magnitude_share_mean=("magnitude_share", "mean"),
        norm_share_median=("norm_share", "median"), norm_share_mean=("norm_share", "mean")).reset_index()
    summ.to_csv(out / "missing_gene_cost_summary.csv", index=False)

    gated = pd.DataFrame({c: ct.gate(bundle / f"context_{c}.h5ad", genes) for c in man["contexts"]}, index=genes)
    noprobe = ct.no_included_probe(genes)
    h1_missing = ~ref_meas
    bound = dict(h1_missing=int(h1_missing.sum()), h1_missing_no_included_probe=int((h1_missing & noprobe).sum()),
                 **{f"h1_missing_gated_in_{c}": int((h1_missing & gated[c].to_numpy()).sum()) for c in gated},
                 **{f"gated_in_{c}": int(gated[c].sum()) for c in gated})
    (out / "missing_gene_cost_h1_bound.json").write_text(json.dumps(bound, indent=2) + "\n")
    print(summ.round(3).to_string(index=False))
    print(json.dumps(bound, indent=2))
    print(f"H1 targets with >= {MIN_SIG} significant genes: {(n_sig >= MIN_SIG).sum()} of {len(targets)}; "
          f"median |S_p| {np.median(n_sig):.0f}")


if __name__ == "__main__":
    main()
