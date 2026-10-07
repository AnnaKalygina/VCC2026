"""Feng source parsing and original guide/control labeling."""
import json
import urllib.request
import numpy as np
import pandas as pd
RAW = "data/raw/feng"
ARTICLE_COUNTS = 27989294
SCREENS = {"fit": ("GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz", 51059483, "GW_Fitness_meta.tsv.gz"), "nonfit": ("GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz", 51059558, "GW_NonFitness_meta.tsv.gz")}
CTRL_TAGS = ("unassigned", "NonTarget")
CTRL = "non-targeting"

def api_sizes(article):
    d = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}", timeout=60))
    return {f["name"]: (f["id"], f["size"]) for f in d["files"]}

def parse_row(line, n_cells):
    """One dense text row `feature,c1,...,cn` -> (feature, nonzero column idx int32, counts int32).

    Pure-numpy byte parser (~5 ms per 322k-field row vs 50 ms for str.split): commas are located
    once, a field is zero iff it is the single byte '0', and only the nonzero fields are decoded,
    digit by digit. Every non-comma byte must be a digit -- a float or a sign raises."""
    i = line.index(b",")
    feat = line[:i].decode()
    buf = np.frombuffer(line, np.uint8)[i:]
    end = len(buf)
    while end and buf[end - 1] in (10, 13):
        end -= 1
    buf = buf[:end]
    comma = np.flatnonzero(buf == 44)
    if len(comma) != n_cells:
        raise ValueError(f"{feat}: {len(comma)} fields, header has {n_cells} cells")
    if ((buf != 44) & ((buf < 48) | (buf > 57))).any():
        raise ValueError(f"{feat}: non-integer bytes in the row -- these are not raw UMI counts")
    starts = comma + 1
    lens = np.diff(np.r_[comma, end]) - 1
    if (lens <= 0).any():
        raise ValueError(f"{feat}: {int((lens <= 0).sum())} empty fields")
    nz = np.flatnonzero((lens > 1) | (buf[starts] != 48))
    s, ln = starts[nz], lens[nz]
    v = np.zeros(len(nz), np.int64)
    for k in range(int(ln.max()) if len(ln) else 0):
        m = ln > k
        v[m] = v[m] * 10 + (buf[s[m] + k].astype(np.int64) - 48)
    if len(v) and v.max() >= 2 ** 31:
        raise ValueError(f"{feat}: count {v.max()} overflows int32")
    # uint16 halves the resident matrix; np.concatenate upcasts to int32 if any row needs it
    return feat, nz.astype(np.int32), v.astype(np.uint16 if not len(v) or v.max() < 65536 else np.int32)

def load_meta(meta_path, cells):
    """Metadata rows in count-file column order. Count ids = metadata ids without the `MP-` prefix."""
    m = pd.read_csv(meta_path, sep="\t")
    m.index = m["Cell_ID"].str.replace(r"^MP-", "", regex=True)
    if not m.index.is_unique:
        raise SystemExit(f"{meta_path}: duplicate cell ids after stripping the MP- prefix")
    ci = pd.Index(cells.astype(object))     # NOT np.setdiff1d: <U vs object arrays compare in Python (minutes)
    missing = ci[~ci.isin(m.index)]
    if len(missing):
        raise SystemExit(f"{len(missing)} count-file cells have no metadata row, e.g. {missing[:3]}")
    extra = len(m) - len(cells)
    if extra:
        print(f"  WARNING {extra} metadata rows have no count column (dropped)")
    m = m.loc[ci]
    g = m["Guide_Call"].astype(str)
    tgt = g.str.rsplit("_", n=1).str[0]
    is_ctrl = g.isin(CTRL_TAGS) | tgt.isin(CTRL_TAGS)
    m["label"] = np.where(is_ctrl, CTRL, tgt)
    m["pool"] = m["Batch"].str.replace(r"_Inlet\d+$", "", regex=True)
    n_na = int(m["Cell_Line"].isna().sum())
    if n_na:
        print(f"  WARNING {n_na} cells without Cell_Line -> stratum line 'NA'")
    m["line"] = m["Cell_Line"].fillna("NA").astype(str)
    m["stratum"] = m["pool"] + "|" + m["line"]
    return m
