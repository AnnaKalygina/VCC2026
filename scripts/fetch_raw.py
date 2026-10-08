"""Download every raw source into data/raw/<source>/ with a download receipt."""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from downloads import download  # noqa: E402

ORION_REV = '53a5bc98d49247bcf967500292575c3d3602de31'   # X-Atlas-Orion dataset revision, pinned 2026-10-09


def _json(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.load(r)


def figshare(article, names, source):
    files = {f['name']: f for f in _json(f'https://api.figshare.com/v2/articles/{article}')['files']}
    return [dict(source=source, name=n, url=f'https://ndownloader.figshare.com/files/{files[n]["id"]}',
                 size=files[n]['size'], md5=files[n].get('computed_md5') or None) for n in names]


def zenodo(record, names, source):
    files = {f['key']: f for f in _json(f'https://zenodo.org/api/records/{record}')['files']}
    out = []
    for n in names:
        algo, md5 = files[n]['checksum'].split(':', 1)
        out.append(dict(source=source, name=n, url=f'https://zenodo.org/api/records/{record}/files/{n}/content',
                        size=files[n]['size'], md5=md5 if algo == 'md5' else None))
    return out


def orion():
    tree = _json(f'https://huggingface.co/api/datasets/Xaira-Therapeutics/X-Atlas-Orion/tree/{ORION_REV}/data')
    base = f'https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/{ORION_REV}/data/'
    return [dict(source='orion', name=f['path'].split('/')[-1], url=base + f['path'].split('/')[-1], size=f['size'],
                 md5=None) for f in tree if f['path'].endswith('.parquet')]


def geo_series(acc, names, source):
    d = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/'
    return [dict(source=source, name=n, url=d + n, size=None, md5=None) for n in names]


def geo_samples(acc, source, keep):
    """Every per-sample supplementary file of a GEO series whose name passes `keep`."""
    d = f'https://ftp.ncbi.nlm.nih.gov/geo/series/{acc[:-3]}nnn/{acc}/suppl/filelist.txt'
    with urllib.request.urlopen(d, timeout=120) as r:
        rows = [line.split('\t') for line in r.read().decode().splitlines()[1:]]
    out = []
    for row in rows:
        name = row[1]
        if not name.startswith('GSM') or not keep(name):
            continue
        gsm = name.split('_')[0]
        out.append(dict(source=source, name=name, size=int(row[3]), md5=None,
                        url=f'https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}'))
    return out


SOURCES = [
    lambda: figshare(20029387, ['K562_essential_raw_singlecell_01.h5ad', 'rpe1_raw_singlecell_01.h5ad'], 'replogle2022'),
    lambda: zenodo(7416068, ['FrangiehIzar2021_RNA.h5ad'], 'scperturb'),
    lambda: figshare(27989294, ['GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz',
                                'GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz'], 'feng'),
    lambda: zenodo(18460279, ['genome_wide_binA.RDS', 'genome_wide_binB.RDS', 'genome_wide_binC.RDS'], 'viperturb'),
    lambda: zenodo(14518762, [f'Seurat_object_{s}_Perturb_seq.rds' for s in ('IFNG', 'INS', 'TGFB', 'TNFA', 'IFNB')]
                   + ['A_readme.txt'], 'jiang2025'),
    lambda: figshare(20029387, ['K562_gwps_raw_singlecell_01.h5ad'], 'replogle2022'),
    orion,
    lambda: figshare(27261219, ['KOLF_Pan_Genome_QC_Filtered.h5ad'], 'kolf'),
    lambda: geo_series('GSE264667', ['GSE264667_jurkat_raw_singlecell_01.h5ad', 'GSE264667_hepg2_raw_singlecell_01.h5ad'],
                       'nadig2025'),
    lambda: geo_series('GSE337988', ['GSE337988_sublib2_processed_objects_Low_assays.h5'], 'gse337988'),
    lambda: geo_samples('GSE295214', 'gse295214', lambda n: n.endswith('_filtered_feature_bc_matrix.h5')),
    # CD4 T (Marson 2025): GEO's per-lane Cell Ranger matrices hold every cell of the authors' S3 h5ad files with
    # identical counts (checked 2026-10-09 on donor 1 resting, lane R1L01: 137,668 of 137,668 cells, 300 cells
    # count-identical), plus the guide-capture features, at a tenth of the size (171 GB vs about 1.7 TB).
    # The authors' per-cell guide calls and QC flags come from the S3 obs tables (scripts/fetch_cd4t_obs.py).
    lambda: geo_samples('GSE314342', 'cd4t_marson2025', lambda n: n.endswith('_sample_filtered_feature_bc_matrix.h5')),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--parallel-files', type=int, default=2, help='files downloaded at once (each uses aria2 splits)')
    ap.add_argument('--only', nargs='*', help='restrict to these sources')
    a = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside Slurm')
    items = [it for g in SOURCES for it in g() if not a.only or it['source'] in a.only]
    known = sum(it['size'] or 0 for it in items)
    print(f'{len(items)} files, {known / 1e9:.1f} GB with known size', flush=True)
    failed = []

    def one(it):
        download(it['url'], ROOT / 'data/raw' / it['source'] / it['name'], size=it['size'], md5=it['md5'])
        return it['name']

    with cf.ThreadPoolExecutor(a.parallel_files) as pool:
        futures = {pool.submit(one, it): it for it in items}
        for f in cf.as_completed(futures):
            try:
                print(f'done {f.result()}', flush=True)
            except Exception as e:                                  # keep going; report at the end
                failed.append(futures[f]['name'])
                print(f'FAILED {futures[f]["name"]}: {e}', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} downloads failed: {failed}')


if __name__ == '__main__':
    main()
