"""Orion sparse-count extraction with the original guide-quality filter."""
import numpy as np
import pyarrow.parquet as pq
from downloads import download
def fetch(url, dest):
    download(url, dest)

def read_shard(path):
    """Extract sparse triplets + per-cell metadata from one parquet, dropping filtered cells."""
    t = pq.ParquetFile(path).read(
        columns=["gene_token_id", "gene_expression", "gene_target", "pass_guide_filter"])
    lc = t["gene_token_id"].combine_chunks()
    ec = t["gene_expression"].combine_chunks()
    off = np.asarray(lc.offsets).astype(np.int64)
    cols = np.asarray(lc.values).astype(np.int32)
    raw = np.asarray(ec.values).astype(np.float32)      # counts <= 3325; float32 is exact here
    tgt = np.asarray(t["gene_target"].combine_chunks()).astype(str)
    keep = np.asarray(t["pass_guide_filter"].combine_chunks()).astype(bool)
    del t, lc, ec

    n_drop = int((~keep).sum())
    if n_drop:
        # Not silent: Batch1 has zero of these, so any occurrence is new information.
        print(f"  WARNING dropped {n_drop:,}/{len(keep):,} cells with pass_guide_filter == 0",
              flush=True)
        nzc = np.diff(off)
        m = np.repeat(keep, nzc)
        off = np.concatenate([[0], np.cumsum(nzc[keep])])
        cols, raw, tgt = cols[m], raw[m], tgt[keep]
    return off, cols, raw, tgt
