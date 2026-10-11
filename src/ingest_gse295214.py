"""hESC public file discovery and original guide calling."""
import re
import urllib.request
import h5py
import numpy as np
import scipy.sparse as sp
SERIES = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE295nnn/GSE295214/suppl/filelist.txt"
SAMPLE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM{a}nnn/GSM{b}/suppl/{f}"
CONTROL_RE = re.compile(r"^(non[-_]?targeting|ntc|negative|neg)", re.I)

def sample_files():
    """(GSM id, filename) for every sample listed in the series file list."""
    txt = urllib.request.urlopen(SERIES).read().decode()
    out = []
    for line in txt.splitlines():
        p = line.split("\t")
        if len(p) > 1 and p[0] == "File" and p[1].endswith(".h5"):
            gsm = p[1].split("_")[0]
            out.append((gsm, p[1]))
    return out

def read_sample(path, min_umi, max_second):
    """-> (genes, gex csr [cells x genes] restricted to assigned cells, target label per cell, assignment rate).

    Guide UMIs are summed per target gene first (two guides for one gene are not a doublet). A cell is assigned
    to its top gene when that gene has >= min_umi UMIs and every other gene has <= max_second x as many (GEO
    states no rule; the old rule, top guide >= 50% of guide UMIs, admitted 50/50 two-gene cells and kept the
    ~10% of cells whose second gene carries > 20% of the top gene's UMIs)."""
    with h5py.File(path) as h:
        m = h["matrix"]
        ftype = np.array(m["features/feature_type"][:]).astype(str)
        names = np.array(m["features/name"][:]).astype(str)
        shape = m["shape"][:]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                          shape=(shape[0], shape[1])).T.tocsr()      # cells x features
    is_g = ftype == "Gene Expression"
    G = X[:, is_g]
    gene = np.array([("non-targeting" if CONTROL_RE.match(n) else n.split("_")[0]) for n in names[~is_g]])
    ug, inv = np.unique(gene, return_inverse=True)
    S = (X[:, ~is_g] @ sp.csr_matrix((np.ones(len(gene)), (np.arange(len(gene)), inv)), shape=(len(gene), len(ug)))).tocsr()
    S.eliminate_zeros()
    nnz = np.diff(S.indptr)
    o = np.lexsort((S.data, np.repeat(np.arange(S.shape[0]), nnz)))   # per cell, gene UMIs ascending
    top, second, best = np.zeros(S.shape[0]), np.zeros(S.shape[0]), np.zeros(S.shape[0], np.int64)
    has1, has2 = nnz >= 1, nnz >= 2
    last = S.indptr[1:] - 1
    top[has1], best[has1] = S.data[o[last[has1]]], S.indices[o[last[has1]]]
    second[has2] = S.data[o[last[has2] - 1]]
    ok = (top >= min_umi) & (second <= max_second * top)
    target = ug[best]
    return names[is_g], G[ok], target[ok], ok.mean()
