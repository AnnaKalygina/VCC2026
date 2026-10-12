"""Read-only remote HDF5 and AnnData categorical helpers."""
import json
import urllib.request
import h5py
import numpy as np
from remote_h5 import HTTPRangeFile, figshare
KOLF_ART, KOLF_FILE = 27261219, "KOLF_Pan_Genome_QC_Filtered.h5ad"

def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])

def open_figshare(article, name, **kw):
    meta = json.load(urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}"))
    hit = [f for f in meta["files"] if f["name"] == name]
    if len(hit) != 1:
        raise SystemExit(f"figshare {article}: {len(hit)} files named {name}")
    fid, size = hit[0]["id"], int(hit[0]["size"])
    print(f"  figshare {article}/{fid} '{name}' {size / 1e9:.2f} GB", flush=True)
    return h5py.File(figshare(fid, size, **kw), "r"), f"figshare:{fid}#{name}"

def open_url(url, **kw):
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=120) as r:
        size = int(r.headers["Content-Length"])
    fh = HTTPRangeFile(url, size, **kw)
    fh._signed = url          # public S3 object: served directly, no presigned redirect to chase
    print(f"  {url} {size / 1e9:.2f} GB", flush=True)
    return h5py.File(fh, "r"), url

def cat(h, key):
    """An AnnData categorical obs/var column as strings, in either of the two on-disk layouts."""
    node = h[key]
    if isinstance(node, h5py.Group):                          # anndata >= 0.8: obs/<k>/{categories,codes}
        cats, codes = node["categories"][:], node["codes"][:]
    else:
        parent, leaf = key.rsplit("/", 1)
        legacy = f"{parent}/__categories/{leaf}"
        cats, codes = (h[legacy][:], node[:]) if legacy in h else (None, None)
    if cats is not None:
        if (codes < 0).any():
            raise SystemExit(f"{key}: {int((codes < 0).sum())} missing values")
        return _dec(cats)[codes]
    if node.dtype.kind in "SO":
        return _dec(node[:])
    raise SystemExit(f"{key} is neither categorical nor a string dataset (dtype {node.dtype})")
