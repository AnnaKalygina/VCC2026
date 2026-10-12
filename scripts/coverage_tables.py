"""Coverage of a challenge panel by the summarized contexts: which panel targets each context perturbs, how well, and
which panel genes it does not measure.

    pixi run python scripts/coverage_tables.py --bundle <controls dir> [--summaries data] [--out results/coverage]
                                               [--compare <results/coverage/<old panel_id>>]

The panel comes from the bundle (gene_names.csv or the control h5ad var, pert_counts.csv, manifest.json, the control
h5ads), so the same command runs unchanged on the final D/E/F bundle. Context descriptions (cell line, condition,
modality, chemistry, lab, source, license) come from workflow/contexts.csv, one row per context in
workflow/config.yaml. A target counts as covered in a context when the context has cells for it. Writes plain UTF-8
CSV (lists joined by ';') to <out>/<panel_id>/:

  inventory.csv                one row per context, Michael's atlas inventory columns first; added_in_audit marks the
                               contexts that are not in Michael's pipeline (workflow/config.yaml on branch main)
  targets_coverage.csv         one row per panel target, least covered first (cell lines, then contexts, then cells)
  targets_by_context_cells.csv panel target x context matrix of cell counts (0 = not perturbed)
  genes_coverage.csv           one row per panel gene: contexts measuring it, the ones that do not, the A/B/C gates
  genes_by_context_measured.csv panel gene x context, 1 = the context measures the gene, with the gates and probe flag
  targets_by_context.csv       long form: cells, reliability, depth, own-gene lfc and knockdown, labels renamed to it
  genes_missing.csv            context x panel gene it does not measure, with the gates and the Flex probe flag
  batch_coverage.csv           context x batch: controls, perturbed cells, targets
  target_labels_report.csv     every target label of every context that is not an exact panel symbol
  panel_diff.csv               with --compare: targets new, dropped or kept relative to an earlier panel
  summary.md                   one page
"""
import argparse
import json
import subprocess
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data/raw/probeset/Chromium_Human_Transcriptome_Probe_Set_v1.0.1_GRCh38-2020-A.csv"
CONTEXTS = ROOT / "workflow/contexts.csv"
GATE_CPM = 5.0                                  # vcc2026 preset: genes tested only above 5 mean CPM in the controls
BASE = "main"                                   # Michael's pipeline before the audit (mibohl/VCC2026 31d1e88)
MICHAEL = ["dataset", "context", "source_accession_publication", "lab_group", "perturbation_modality",
           "sequencing_tech", "n_targets", "n_panel_targets", "median_cells_per_target", "median_control_lib_size",
           "mean_reliability"]


def _s(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a], dtype=object)


def gate(path, genes):
    """Mean CPM (arithmetic mean of c / L * 1e6 over control cells) of each panel gene > 5 in one bundle context."""
    a = ad.read_h5ad(path)
    X = a.X.tocsr() if hasattr(a.X, "tocsr") else a.X
    L = np.asarray(X.sum(axis=1)).ravel()
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
            for k in ("n_cells", "reliability_sb", "reliability_sb_batchmatched", "depth_median", "frac_no_ctrl"):
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
        c["ctrl_depth_median"] = float(h["control/depth_median"][()]) if "control/depth_median" in h else np.nan
    return c


def _nanstat(f, a):
    a = np.asarray(a, float)
    return float(f(a)) if np.isfinite(a).any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--summaries", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/coverage"))
    ap.add_argument("--compare", help="an earlier panel's output directory, for panel_diff.csv")
    a = ap.parse_args()
    bundle = Path(a.bundle)
    man = json.loads((bundle / "manifest.json").read_text())
    pid, bctx = man["panel_id"], man["contexts"]
    panel_targets = pd.read_csv(bundle / "pert_counts.csv")[man.get("pert_col", "target_gene")].astype(str).to_numpy()
    h5s = {c: bundle / f"context_{c}.h5ad" for c in bctx}
    genes = ad.read_h5ad(h5s[bctx[0]], backed="r").var_names.astype(str).to_numpy()
    gated = pd.DataFrame({f"gated_in_{c}": gate(p, genes) for c, p in h5s.items()}, index=genes)
    gate_cols = list(gated.columns)
    noprobe = pd.Series(no_included_probe(genes), index=genes)
    out = Path(a.out) / pid
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(CONTEXTS, dtype=str).set_index("context")
    configured = list(yaml.safe_load((ROOT / "workflow/config.yaml").read_text())["contexts"])
    if set(meta.index) != set(configured) or meta.index.duplicated().any():
        raise SystemExit(f"{CONTEXTS}: rows must match workflow/config.yaml exactly; differ: "
                         f"{sorted(set(meta.index) ^ set(configured))}")
    files = {f.stem: f for f in sorted(Path(a.summaries).glob("*.h5")) if f.with_suffix(".json").exists()}
    if set(files) != set(configured):
        raise SystemExit(f"summaries differ from workflow/config.yaml: missing {sorted(set(configured) - set(files))}, "
                         f"extra {sorted(set(files) - set(configured))}")
    base = set(yaml.safe_load(subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:workflow/config.yaml"],
                                             check=True, capture_output=True, text=True).stdout)["contexts"])
    unstim = meta["condition"].str.startswith("unstimulated")
    crispri = meta["perturbation_modality"].str.startswith("CRISPRi")

    tbc, gm, inv, bcov, lab = [], [], [], [], []
    cells = pd.DataFrame(0, index=panel_targets, columns=configured, dtype=np.int64)
    measured = pd.DataFrame(False, index=genes, columns=configured)
    for ctx in configured:
        f = files[ctx]
        js = json.loads(f.with_suffix(".json").read_text())
        c = read_context(f)
        if not (c["genes"] == genes).all():
            raise SystemExit(f"{f}: gene axis differs from the bundle's")
        measured[ctx] = c["measured"]
        tl = js.get("target_labels", {"changed": []})
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
                cells.loc[t, ctx] = int(c["n_cells"][i])
                row.update(n_cells=int(c["n_cells"][i]), reliability_sb=float(c["reliability_sb"][i]),
                           reliability_sb_batchmatched=float(c["reliability_sb_batchmatched"][i]),
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
        m = meta.loc[ctx]
        inv.append(dict(
            dataset=m["dataset"], context=ctx, source_accession_publication=m["source_accession_publication"],
            lab_group=m["lab_group"], perturbation_modality=m["perturbation_modality"],
            sequencing_tech=m["sequencing_tech"], n_targets=len(targets), n_panel_targets=int(in_panel.sum()),
            median_cells_per_target=_nanstat(np.nanmedian, c.get("n_cells", [])),
            median_control_lib_size=c["ctrl_depth_median"],
            mean_reliability=_nanstat(np.nanmean, c.get("reliability_sb", [])),
            cell_line=m["cell_line"], cell_type=m["cell_type"], condition=m["condition"], license=m["license"],
            added_in_audit=ctx not in base,
            n_controls=c["n_ctrl"], n_genes_measured=int(c["measured"].sum()), n_panel_genes_missing=len(miss),
            pct_panel_genes_measured=round(100 * c["measured"].mean(), 1),
            n_native_genes=js.get("genes", {}).get("n_native"),
            **{f"n_genes_matched_{k}": js.get("genes", {}).get(k) for k in ("exact", "ensembl", "case", "hgnc")},
            n_missing_genes_above_gate_ABC=int(gated.reindex(miss).any(axis=1).sum()),
            **{f"n_missing_genes_above_gate_{k.split('_')[-1]}": int(gated[k].reindex(miss).sum()) for k in gate_cols},
            n_missing_genes_without_included_probe=int(noprobe.reindex(miss).sum()),
            mean_reliability_batchmatched=_nanstat(np.nanmean, c.get("reliability_sb_batchmatched", [])),
            median_knockdown_fraction=_nanstat(np.nanmedian, 1 - 2 ** own) if len(own) else np.nan,
            median_batch_shift_rms=js.get("checks", {}).get("median_batch_shift_rms"),
            batch_key=js.get("batch_key"), n_labels_renamed=sum(len(v) for v in renamed.values()),
            n_genes_hgnc_renamed=len(js.get("genes", {}).get("hgnc_renamed", [])),
            missing_panel_targets=";".join(sorted(set(panel_targets) - set(targets))),
            code=js.get("code", {}).get("commit", "")[:7]))
        if "batches" in c and c.get("n_cells_by_batch") is not None:
            nb = c["n_cells_by_batch"][:, :len(c["batches"])]
            for b, name in enumerate(c["batches"]):
                bcov.append(dict(context=ctx, batch=name, n_controls=int(c["ctrl_by_batch"][b]),
                                 n_perturbed_cells=int(nb[:, b].sum()), n_targets=int((nb[:, b] > 0).sum())))
        print(f"{ctx}: {in_panel.sum()} of {len(panel_targets)} panel targets, {len(miss)} panel genes missing",
              flush=True)

    tbc, gm, inv = pd.DataFrame(tbc), pd.DataFrame(gm), pd.DataFrame(inv)
    write = lambda df, name: df.to_csv(out / name, index=False, lineterminator="\n")
    write(inv, "inventory.csv")
    write(tbc, "targets_by_context.csv")
    write(gm, "genes_missing.csv")
    write(pd.DataFrame(bcov), "batch_coverage.csv")
    write(pd.DataFrame(lab), "target_labels_report.csv")

    # targets: covered = the context has cells for the target
    order = meta.loc[configured].sort_values(["dataset", "cell_line"]).index
    cov = cells > 0
    tc = pd.DataFrame(index=pd.Index(panel_targets, name="panel_target"))
    tc["n_contexts"] = cov.sum(axis=1)
    tc["n_datasets"] = cov.apply(lambda r: meta.loc[r.index[r], "dataset"].nunique(), axis=1)
    tc["n_cell_lines"] = cov.apply(lambda r: meta.loc[r.index[r], "cell_line"].nunique(), axis=1)
    tc["n_unstimulated_contexts"] = (cov & unstim.reindex(configured).to_numpy()).sum(axis=1)
    tc["n_crispri_contexts"] = (cov & crispri.reindex(configured).to_numpy()).sum(axis=1)
    tc["total_cells"] = cells.sum(axis=1)
    tc["median_cells_per_context"] = cells.where(cov).median(axis=1)
    tc["contexts"] = cov.apply(lambda r: ";".join(r.index[r]), axis=1)
    tc["datasets"] = cov.apply(lambda r: ";".join(sorted(set(meta.loc[r.index[r], "dataset"]))), axis=1)
    tc["cell_lines"] = cov.apply(lambda r: ";".join(sorted(set(meta.loc[r.index[r], "cell_line"]))), axis=1)
    for col in gate_cols:
        tc["own_gene_above_gate_" + col.split("_")[-1]] = gated[col].reindex(panel_targets).to_numpy()
    tc = tc.sort_values(["n_cell_lines", "n_contexts", "total_cells"], kind="stable")
    tc.insert(0, "rank_least_covered", np.arange(1, len(tc) + 1))
    tc.reset_index().to_csv(out / "targets_coverage.csv", index=False, lineterminator="\n")
    cells[order].rename_axis("panel_target").reset_index().to_csv(out / "targets_by_context_cells.csv", index=False,
                                                                  lineterminator="\n")

    # genes: measured per context
    gc = pd.DataFrame(index=pd.Index(genes, name="gene"))
    gc["n_contexts_measured"] = measured.sum(axis=1)
    gc["n_datasets_measured"] = measured.apply(lambda r: meta.loc[r.index[r], "dataset"].nunique(), axis=1)
    gc["n_unstimulated_contexts_measured"] = (measured & unstim.reindex(configured).to_numpy()).sum(axis=1)
    gc["contexts_missing"] = (~measured).apply(lambda r: ";".join(r.index[r]), axis=1)
    gc = gc.join(gated).join(noprobe.rename("no_included_probe"))
    gc["gated_in_any"] = gated.any(axis=1)
    gc = gc.sort_values(["n_contexts_measured", "gated_in_any"], ascending=[True, False], kind="stable")
    gc.reset_index().to_csv(out / "genes_coverage.csv", index=False, lineterminator="\n")
    wide = gated.join(noprobe.rename("no_included_probe")).assign(n_contexts_measured=measured.sum(axis=1))
    wide = wide.join(measured[order].astype(int)).loc[gc.index]
    wide.rename_axis("gene").reset_index().to_csv(out / "genes_by_context_measured.csv", index=False,
                                                  lineterminator="\n")

    if a.compare:
        old = pd.read_csv(Path(a.compare) / "targets_coverage.csv")
        old_t, new_t = set(old.panel_target), set(panel_targets)
        diff = tc.reset_index()[["panel_target", "n_contexts", "n_datasets", "n_cell_lines"]]
        diff.insert(1, "status", np.where(diff.panel_target.isin(old_t), "kept", "new"))
        dropped = pd.DataFrame(dict(panel_target=sorted(old_t - new_t), status="dropped"))
        pd.concat([diff, dropped]).to_csv(out / "panel_diff.csv", index=False, lineterminator="\n")

    per_t = tc["n_contexts"]
    lines = [f"# Coverage of panel {pid}", "",
             f"{len(panel_targets)} panel targets, {len(genes)} panel genes; {len(configured)} contexts from "
             f"{meta['dataset'].nunique()} datasets and {meta['cell_line'].nunique()} cell lines "
             f"({int(unstim.sum())} unstimulated, {int(crispri.sum())} CRISPRi). Covered = the context has cells.", "",
             "## Panel targets", "",
             f"- perturbed in no context: {int((per_t == 0).sum())}",
             f"- contexts per target: min {per_t.min()}, median {per_t.median():.0f}, max {per_t.max()}",
             f"- cell lines per target: min {tc.n_cell_lines.min()}, median {tc.n_cell_lines.median():.0f}",
             "- least covered (cell lines / contexts / cells): " + ", ".join(
                 f"{t} ({r.n_cell_lines}/{r.n_contexts}/{r.total_cells})" for t, r in tc.head(15).iterrows()), "",
             "## Panel genes", "",
             f"- measured in no context: {int((gc.n_contexts_measured == 0).sum())}; in all {len(configured)}: "
             f"{int((gc.n_contexts_measured == len(configured)).sum())}; median contexts per gene "
             f"{gc.n_contexts_measured.median():.0f}",
             f"- above the 5 CPM gate in A/B/C and measured in fewer than 10 contexts: "
             f"{int((gc.gated_in_any & (gc.n_contexts_measured < 10)).sum())}", "",
             "## Per context", "",
             "| context | cell line | condition | panel targets | mean reliability, batch-matched | "
             "genes measured | missing genes above the gate in A/B/C |", "|---|---|---|---|---|---|---|"]
    for r in inv.itertuples():
        lines.append(f"| {r.context} | {r.cell_line} | {r.condition} | {r.n_panel_targets} | "
                     f"{r.mean_reliability_batchmatched:.2f} | {r.n_genes_measured} | {r.n_missing_genes_above_gate_ABC} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
