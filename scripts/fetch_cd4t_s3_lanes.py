"""Recover the two CD4 T Stim8hr lanes whose GEO matrices are truncated on GEO itself (AUDIT N1) from the authors'
S3 h5ad files, which hold the same cells.

Each lane's cells are one contiguous row range of the S3 CSR matrix; only that range is read by HTTP range requests
(src/remote_h5.py, about 6 to 7 GB per lane) and written in the Cell Ranger h5 layout the CD4 T reader reads
(gene-expression features only), to data/raw/cd4t_marson2025/s3_lanes/CD4i_<lane>_<donor>_<cond>.h5, with a receipt
in data/provenance/.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import anndata as ad  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from add_dispersion import open_url  # noqa: E402
from downloads import digest  # noqa: E402

URL = 'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{}.assigned_guide.h5ad'
RAW = ROOT / 'data/raw/cd4t_marson2025'
LANES = [('D1', 'Stim8hr', 'R1L14'), ('D4', 'Stim8hr', 'R2L09')]     # truncated on GEO (AUDIT N1)
STEP = 5000                                                         # cells per range read


def fetch(donor, cond, lane):
    dest = RAW / 's3_lanes' / f'CD4i_{lane}_{donor}_{cond}.h5'
    receipt = ROOT / 'data/provenance' / (hashlib.sha256(str(dest.resolve()).encode()).hexdigest() + '.json')
    if dest.exists() and receipt.exists() and json.loads(receipt.read_text())['sha256'] == digest(dest):
        print(f'have {dest.name}', flush=True)
        return
    obs = pd.read_parquet(RAW / 'obs' / f'{donor}_{cond}.obs.parquet', columns=['lane_id'])
    rows = np.flatnonzero(obs.lane_id.astype(str).to_numpy() == f'CD4i_{lane}')
    lo, hi = int(rows[0]), int(rows[-1]) + 1
    if hi - lo != len(rows):
        raise SystemExit(f'{donor} {cond} {lane}: lane cells are not one contiguous row range')
    url = URL.format(f'{donor}_{cond}')
    h, _ = open_url(url)
    X = h['X']
    n_genes = int(X.attrs['shape'][1])
    v = ad.io.read_elem(h['var'])
    var = {k: v[k].astype(str).to_numpy() for k in ('gene_ids', 'gene_name', 'feature_types')}
    if set(var['feature_types']) != {'Gene Expression'}:
        raise SystemExit(f'{url}: features other than gene expression')
    indptr = X['indptr'][lo:hi + 1].astype(np.int64)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp')
    with h5py.File(tmp, 'w') as out:
        m = out.create_group('matrix')
        nnz = int(indptr[-1] - indptr[0])
        data = m.create_dataset('data', (nnz,), np.int32, chunks=(1 << 20,), compression='gzip')
        ind = m.create_dataset('indices', (nnz,), np.int64, chunks=(1 << 20,), compression='gzip')
        for a in range(0, hi - lo, STEP):
            b = min(a + STEP, hi - lo)
            s, e = int(indptr[a]), int(indptr[b])
            d = X['data'][s:e]
            if (d < 0).any() or (d != np.round(d)).any():
                raise SystemExit(f'{url}: rows {lo + a}-{lo + b} are not raw integer counts')
            data[s - indptr[0]:e - indptr[0]] = d.astype(np.int32)
            ind[s - indptr[0]:e - indptr[0]] = X['indices'][s:e]
            print(f'  {lane} {donor} {cond}: {b:,} of {hi - lo:,} cells', flush=True)
        m['indptr'] = indptr - indptr[0]
        m['shape'] = np.array([n_genes, hi - lo], np.int64)
        m['barcodes'] = obs.index[lo:hi].str.split('_').str[0].to_numpy().astype('S')
        f = m.create_group('features')
        f['id'] = var['gene_ids'].astype('S')
        f['name'] = var['gene_name'].astype('S')
        f['feature_type'] = var['feature_types'].astype('S')
    h.close()
    tmp.replace(dest)
    receipt.write_text(json.dumps(dict(url=url, path=str(dest.relative_to(ROOT)), rows=[lo, hi],
                                       bytes=dest.stat().st_size, sha256=digest(dest), upstream_md5=None),
                                  indent=2) + '\n')
    print(f'wrote {dest.name}: {hi - lo:,} cells', flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside Slurm')
    for lane in LANES:
        fetch(*lane)


if __name__ == '__main__':
    main()
