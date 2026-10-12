"""DLD1 original singlet QC and dense-count block reader."""
import h5py
import numpy as np
import pandas as pd
RAW = "data/raw/gse337988"
H5 = f"{RAW}/sublib2_processed_objects_Low_assays.h5"
COLDATA, ROWDATA = f"{RAW}/low_coldata.csv.gz", f"{RAW}/low_rowdata.csv.gz"
CTRL = "non-targeting"
BLOCK = 1000
_KEEP = None

def cell_table(min_cells, max_rows=0):
    """Kept rows (file order), their group code, and the group names (CTRL is code 0)."""
    cols = ["total.counts", "demux_type", "demux_type_v2", "demux_type.fishash", "guide_name",
            "assignment.fishash", "assignment_v2", "promoter", "MOI"]
    d = pd.read_csv(COLDATA, usecols=cols, dtype=str, keep_default_na=False)
    n_all = len(d)
    hs = d.demux_type == "singlet"
    if set(d.MOI[hs]) != {"Low"}:
        raise ValueError(f"hashtag singlets carry MOI {set(d.MOI[hs])}, expected only Low")
    gs = hs & (d.demux_type_v2 == "singlet")
    cons = gs & (d["demux_type.fishash"] == "singlet") & (d["assignment.fishash"] == d.guide_name)
    meng = d.guide_name.str.contains("MENG")
    keep = cons & ~meng
    if max_rows:
        print(f"  SMOKE TEST: only the first {max_rows:,} rows of the h5")
        keep &= np.arange(n_all) < max_rows
    print(f"cells: {n_all:,} in the Low object; hashtag singlets {hs.sum():,}; + guide singlet (v2) {gs.sum():,}; "
          f"+ second caller agrees {cons.sum():,}; - NTC_MENG {int((cons & meng).sum()):,} -> {keep.sum():,}")
    ntc = d.assignment_v2 == "NTC"
    prefix = np.where(ntc, "NTC", d.assignment_v2 + "_" + d.promoter + "_")
    agree = np.array([g.startswith(p) for g, p in zip(d.guide_name, prefix)])
    bad = keep.to_numpy() & ~agree
    if bad.any():
        raise ValueError(f"{int(bad.sum())} kept cells: guide_name disagrees with assignment_v2/promoter")
    lab = np.where(ntc, CTRL, d.assignment_v2 + "_" + d.promoter)
    gene_of = dict(zip(lab, d.assignment_v2))
    rows = np.flatnonzero(keep.to_numpy())
    lab = lab[rows]
    vc = pd.Series(lab).value_counts()
    small = vc[(vc < min_cells) & (vc.index != CTRL)]
    if len(small):
        print(f"  WARNING dropping {len(small)} TSS groups with < {min_cells} cells ({int(small.sum())} cells), "
              f"e.g. {small.head(5).to_dict()}")
        ok = ~np.isin(lab, small.index.to_numpy())
        rows, lab = rows[ok], lab[ok]
    names = np.array([CTRL] + sorted(set(lab) - {CTRL}))
    code = pd.Series(np.arange(len(names)), index=names)[lab].to_numpy().astype(np.int32)
    n = np.bincount(code, minlength=len(names))
    print(f"  {len(names) - 1:,} TSS groups, {n[0]:,} control cells; cells/TSS median {np.median(n[1:]):.0f} "
          f"(5-95%: {np.percentile(n[1:], 5):.0f}-{np.percentile(n[1:], 95):.0f}); largest "
          f"{dict(zip(names[1:][np.argsort(-n[1:])[:5]], np.sort(n[1:])[::-1][:5]))}")
    lib = d["total.counts"].to_numpy().astype(np.int64)[rows]
    return rows, code, names, np.array([gene_of[x] for x in names]), lib

def _init(keep):
    global _KEEP
    _KEEP = keep

def _read_block(lo):
    with h5py.File(H5, "r") as f:
        X = f["assay001"]
        hi = min(lo + BLOCK, X.shape[0])
        loc = np.flatnonzero(_KEEP[lo:hi])
        b = X[lo:hi][loc]
        assert X.shape[1] <= 65536, "column indices are stored as uint16"
    r, c = np.nonzero(b)
    v = b[r, c]
    vmax = int(v.max()) if len(v) else 0
    return (lo, np.bincount(r, minlength=len(loc)).astype(np.int32), c.astype(np.uint16),
            v.astype(np.uint16 if vmax <= 65535 else np.int32), b.sum(1, dtype=np.int64))
