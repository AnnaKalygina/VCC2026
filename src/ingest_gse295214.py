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

def read_sample(path, min_umi, min_frac):
    """-> (genes, gex csr [cells x genes] restricted to assigned cells, target label per cell)."""
    with h5py.File(path) as h:
        m = h["matrix"]
        ftype = np.array(m["features/feature_type"][:]).astype(str)
        names = np.array(m["features/name"][:]).astype(str)
        shape = m["shape"][:]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]),
                          shape=(shape[0], shape[1])).T.tocsr()      # cells x features
    is_g = ftype == "Gene Expression"
    G = X[:, is_g]
    S = X[:, ~is_g]
    gnames = names[~is_g]
    tot = np.asarray(S.sum(1)).ravel()
    top = np.asarray(S.argmax(1)).ravel()
    topv = np.asarray(S.max(1).todense()).ravel()
    ok = (topv >= min_umi) & (topv >= min_frac * np.maximum(tot, 1))
    target = np.array([("non-targeting" if CONTROL_RE.match(n) else n.split("_")[0])
                       for n in gnames[top]])
    return names[is_g], G[ok], target[ok], ok.mean()
