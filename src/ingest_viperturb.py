"""VIPerturb original per-cell gene/control labels."""
import numpy as np
RAW = "data/raw/viperturb"
BINS = ("A", "B", "C")
CTRL = "non-targeting"
SRC_CTRL = "NO-TARGET"

def labels(meta):
    """Per-cell target label from meta.data$gene with the non-targeting cells mapped to CTRL; '' = cell dropped.
    The objects hold only assigned cells with ONE target gene each (bin A: 5,666 targets + NO-TARGET, per-target
    counts identical to the manifest), already filtered by the authors (UMI >= 1000). No Mixscape filter: the
    preprint applies it only to selected heatmaps, and no other donor screen here is escape-filtered either."""
    g = meta["gene"]
    if g.isna().any():
        print(f"  WARNING {int(g.isna().sum())} cells have no gene label -> dropped")
    lab = g.fillna("").astype(str).to_numpy().astype(object)
    if (lab == CTRL).any():
        raise SystemExit(f"a target is literally named {CTRL!r}")
    lab[lab == SRC_CTRL] = CTRL
    multi = np.char.find(lab.astype(str), "_") >= 0
    if multi.any():
        print(f"  WARNING {int(multi.sum())} labels contain '_' (multi-target?): {sorted(set(lab[multi]))[:5]}")
    print(f"  labels: {len(set(lab)) - 1:,} targets, {int((lab == CTRL).sum()):,} {SRC_CTRL} -> {CTRL!r}")
    return lab
