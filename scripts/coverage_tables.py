"""Coverage of a challenge panel by the summarized contexts: which panel targets each context perturbs, how well, and
which panel genes it does not measure.

    pixi run python scripts/coverage_tables.py --bundle <controls dir> [--summaries data] [--out results/coverage]

The panel comes from the bundle (gene_names.csv or the control h5ad var, pert_counts.csv, manifest.json, the control
h5ads), so the same command runs unchanged on the final D/E/F bundle. Writes to <out>/<panel_id>/:

  targets_by_context.csv   panel target x context: perturbed?, cells, reliability, depth, own-gene lfc and knockdown,
                           raw labels renamed to it, own gene above the 5 CPM DE gate in each bundle context
  genes_missing.csv        context x panel gene it does not measure, with the DE gate per bundle context and whether
                           the Flex v1.0.1 probe set has an included probe for it
  inventory.csv            one row per context
  batch_coverage.csv       context x batch: controls, perturbed cells, targets
  target_labels_report.csv every target label of every context that is not an exact panel symbol
  summary.md               panel targets x contexts and the gene gaps, one page
"""
import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data/raw/probeset/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2020-A.csv"
GATE_CPM = 5.0                                  # vcc2026 preset: genes tested only above 5 mean CPM in the controls


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def gate(path, genes):
    """Mean CPM (arithmetic mean of c / L * 1e6 over control cells) of each panel gene > 5 in one bundle context."""
    a = ad.read_h5ad(path)
    X = a.X.tocsr() if hasattr(a.X, "tocsr") else a.X
    L = np.asarray(X.sum(1)).ravel()
    cpm = np.asarray(X.multiply(1e6 / np.maximum(L, 1)[:, None]).mean(0)).ravel()
    return pd.Series(cpm, index=a.var_names.astype(str)).reindex(genes).to_numpy() > GATE_CPM


def no_included_probe(genes):
    p = pd.read_csv(PROBES, comment="#", usecols=["probe_id", "included"])
    sym = p.probe_id.str.split("|").str[1]
    ok = p.included.astype(str).str.upper().eq("TRUE") & ~p.probe_id.str.contains("DEPRECATED")
    return ~np.isin(genes, sym[ok].unique())


def read_context(path):
    with h5py.File(path) as h:
        c = dict(genes=_s(h["genes"][:]), measured=h["measured"][:].astype(bool), attrs=dict(h.attrs))
        if "pert" in h:
            p = h["pert"]
            c["targets"] = _s(p["targets"][:])
            for k in ("n_cells", "reliability_sb", "depth_median", "frac_no_ctrl"):
                c[k] = p[k][:] if k in p else np.full(len(c["targets"]), np.nan)
            gi = pd.Series(np.arange(len(c["genes"])), index=c["genes"])
            own = gi.reindex(c["targets"]).to_numpy()
            ok = ~np.isnan(own)
            lfc = np.full(len(own), np.nan)
            if ok.any():
                rows = np.flatnonzero(ok)
                cols = own[ok].astype(int)
                lfc[rows] = p["lfc"][:][rows, cols]
                lfc[rows[~c["measured"][cols]]] = np.nan
            c["own_lfc"] = lfc
            c["n_cells_by_batch"] = p["n_cells_by_batch"][:] if "n_cells_by_batch" in p else None
        if "batches" in h:
            c["batches"] = _s(h["batches"][:])
            c["ctrl_by_batch"] = h["control/n_cells_by_batch"][:]
        c["n_ctrl"] = int(h["control/n_cells"][()])
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/coverage"))
    a = ap.parse_args()
    bundle = Path(a.bundle)
    man = json.loads((bundle / "manifest.json").read_text())
    pid, bctx = man["panel_id"], man["contexts"]
    panel_targets = pd.read_csv(bundle / "pert_counts.csv")[man.get("pert_col", "target_gene")].astype(str).to_numpy()
    h5s = {c: bundle / f"context_{c}.h5ad" for c in bctx}
    genes = ad.read_h5ad(h5s[bctx[0]], backed="r").var_names.astype(str).to_numpy()
    gated = pd.DataFrame({f"gated_in_{c}": gate(p, genes) for c, p in h5s.items()}, index=genes)
    noprobe = pd.Series(no_included_probe(genes), index=genes)
    out = Path(a.out) / pid
    out.mkdir(parents=True, exist_ok=True)

    tbc, gm, inv, bcov, lab = [], [], [], [], []
    for f in sorted(Path(a.summaries).glob("*.h5")):
        js = f.with_suffix(".json")
        if not js.exists():
            continue
        ctx, meta = f.stem, json.loads(js.read_text())
        c = read_context(f)
        if not (c["genes"] == genes).all():
            raise SystemExit(f"{f}: gene axis differs from the bundle's")
        tl = meta.get("target_labels", {"changed": []})
        renamed = {}
        for r in tl["changed"]:
            lab.append(dict(context=ctx, **{k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()}))
            if r["action"].startswith("hgnc") or r["action"] in ("case", "underscore", "deposit_gene_axis"):
                renamed.setdefault(r["name"], []).append(r["raw"])
        targets = c.get("targets", np.array([], dtype=object))
        ti = pd.Series(np.arange(len(targets)), index=targets)
        for t in panel_targets:
            i = ti.get(t)
            row = dict(panel_target=t, context=ctx, perturbed=i is not None)
            if i is not None:
                lfc = c["own_lfc"][i]
                row.update(n_cells=int(c["n_cells"][i]), reliability_sb=float(c["reliability_sb"][i]),
                           depth_median=float(c["depth_median"][i]), own_gene_measured=bool(np.isfinite(lfc)),
                           on_target_lfc=float(lfc), knockdown_fraction=float(1 - 2 ** lfc),
                           renamed_from=";".join(renamed.get(t, [])))
            row.update({k: bool(v) for k, v in gated.loc[t].items()})
            tbc.append(row)
        miss = genes[~c["measured"]]
        for g in miss:
            gm.append(dict(context=ctx, gene=g, **{k: bool(v) for k, v in gated.loc[g].items()},
                           no_included_probe=bool(noprobe[g])))
        in_panel = np.isin(targets, panel_targets)
        own = c.get("own_lfc", np.array([]))
        any_gate = gated.any(axis=1).reindex(miss).to_numpy()
        inv.append(dict(
            context=ctx, reader=meta.get("reader"), batch_key=meta.get("batch_key"), n_ctrl=c["n_ctrl"],
            n_targets_total=len(targets), n_targets_panel_genes=int(np.isin(targets, genes).sum()),
            n_panel_targets_perturbed=int(in_panel.sum()),
            missing_panel_targets=";".join(sorted(set(panel_targets) - set(targets))),
            n_genes_measured=int(c["measured"].sum()), n_genes_missing=len(miss),
            n_genes_missing_gated_any=int(any_gate.sum()),
            n_genes_missing_no_included_probe=int(noprobe.reindex(miss).sum()),
            median_reliability_sb=float(np.nanmedian(c["reliability_sb"])) if len(targets) else np.nan,
            median_knockdown_fraction=float(np.nanmedian(1 - 2 ** own)) if np.isfinite(own).any() else np.nan,
            median_batch_shift_rms=meta.get("checks", {}).get("median_batch_shift_rms"),
            n_labels_renamed=sum(len(v) for v in renamed.values()),
            code=meta.get("code", {}).get("commit", "")[:7]))
        if "batches" in c and c.get("n_cells_by_batch") is not None:
            nb = c["n_cells_by_batch"][:, :len(c["batches"])]
            for b, name in enumerate(c["batches"]):
                bcov.append(dict(context=ctx, batch=name, n_controls=int(c["ctrl_by_batch"][b]),
                                 n_perturbed_cells=int(nb[:, b].sum()), n_targets=int((nb[:, b] > 0).sum())))
        print(f"{ctx}: {in_panel.sum()} of {len(panel_targets)} panel targets, {len(miss)} panel genes missing",
              flush=True)

    tbc, gm, inv = pd.DataFrame(tbc), pd.DataFrame(gm), pd.DataFrame(inv)
    tbc.to_csv(out / "targets_by_context.csv", index=False)
    gm.to_csv(out / "genes_missing.csv", index=False)
    inv.to_csv(out / "inventory.csv", index=False)
    pd.DataFrame(bcov).to_csv(out / "batch_coverage.csv", index=False)
    pd.DataFrame(lab).to_csv(out / "target_labels_report.csv", index=False)

    per_t = tbc[tbc.perturbed].groupby("panel_target").context.nunique().reindex(panel_targets, fill_value=0)
    good = tbc[tbc.perturbed & (tbc.reliability_sb > 0.5)].groupby("panel_target").context.nunique()
    good = good.reindex(panel_targets, fill_value=0)
    lines = [f"# Coverage of panel {pid}", "",
             f"{len(panel_targets)} panel targets, {len(genes)} panel genes, {len(inv)} contexts.", "",
             "## Panel targets", "",
             f"- perturbed in no context: {int((per_t == 0).sum())} "
             f"({', '.join(per_t.index[per_t == 0][:40])}{' ...' if (per_t == 0).sum() > 40 else ''})",
             f"- perturbed in 1 context: {int((per_t == 1).sum())}; 2 to 4: {int(per_t.between(2, 4).sum())}; "
             f"5 or more: {int((per_t >= 5).sum())}",
             f"- with reliability > 0.5 in at least one context: {int((good > 0).sum())}", "",
             "## Per context", "",
             "| context | panel targets perturbed | median reliability | median knockdown | genes measured | "
             "missing genes above the gate in A/B/C | missing without an included probe |",
             "|---|---|---|---|---|---|---|"]
    for r in inv.itertuples():
        lines.append(f"| {r.context} | {r.n_panel_targets_perturbed} | {r.median_reliability_sb:.2f} | "
                     f"{r.median_knockdown_fraction:.2f} | {r.n_genes_measured} | {r.n_genes_missing_gated_any} | "
                     f"{r.n_genes_missing_no_included_probe} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
