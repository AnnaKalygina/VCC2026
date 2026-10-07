"""Generate/download prerequisites. Full preparation runs inside Slurm."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src/clean'))
import h5py
import numpy as np
import pandas as pd
from downloads import download, figshare_file, json_url, zenodo_file
from readers import cat, index_of

BASE = 'https://storage.googleapis.com/arc-institute-virtual-cell-atlas/'
SPLITS = {'Training': ('train', 150), 'Validation': ('validation', 50), 'Test': ('test', 100)}


def validate_h1(root=ROOT):
    genes, controls, targets = None, None, set()
    report = {}
    for split, (_, expected) in SPLITS.items():
        with h5py.File(root / f'data/vcc2025/adata_{split}.h5ad') as h:
            axis = index_of(h, 'var')
            labels = cat(h, 'obs/target_gene')
            ids = index_of(h, 'obs')[labels == 'non-targeting']
            current = set(labels) - {'non-targeting'}
            if len(current) != expected or targets & current:
                raise ValueError(f'{split}: expected {expected} distinct, disjoint targets')
            if genes is not None:
                np.testing.assert_array_equal(axis, genes)
                np.testing.assert_array_equal(ids, controls)
            if len(ids) == 0 or len(set(ids)) != len(ids):
                raise ValueError(f'{split}: empty or duplicate controls')
            genes, controls = axis, ids
            targets |= current
            if 'X' not in h:
                raise ValueError(f'{split}: missing count matrix')
            report[split] = dict(targets=len(current), controls=len(ids), genes=len(axis))
    if len(targets) != 300:
        raise ValueError('H1 must contain all 300 targets')
    return report


def h1(split):
    directory, _ = SPLITS[split]
    object_name = f'virtual-cell-challenge/2025/{directory}/adata_{split}.h5ad'
    meta = json_url('https://storage.googleapis.com/storage/v1/b/arc-institute-virtual-cell-atlas/o/'
                    + urllib.parse.quote(object_name, safe=''))
    checksum = base64.b64decode(meta['md5Hash']).hex() if 'md5Hash' in meta else None
    download(BASE + object_name, ROOT / f'data/vcc2025/adata_{split}.h5ad',
             size=int(meta['size']), md5=checksum)


def orion():
    download('https://huggingface.co/datasets/Xaira-Therapeutics/X-Atlas-Orion/resolve/main/metadata/gene_metadata.parquet',
             ROOT / 'data/orion_probe/gene_metadata.parquet')


def feng():
    for screen, target in [('FitnessGenes', 'GW_Fitness_meta.tsv.gz'), ('NonFitnessGenes', 'GW_NonFitness_meta.tsv.gz')]:
        figshare_file(27989294, f'GenomeWideScreen_{screen}_Cell-Metadata.tsv.gz', ROOT / 'data/raw/feng' / target)
    figshare_file(26819743, 'GenomeWideScreen_LFC_byGene.tsv.gz', ROOT / 'data/raw/feng/GenomeWideScreen_LFC_byGene.tsv.gz')


def dld1():
    prefix = ROOT / 'data/raw/gse337988/low'
    rds = download('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/GSE337988_sublib2_processed_objects_Low_se.rds',
                   prefix.parent / 'Low_se.rds')
    subprocess.run(['Rscript', str(ROOT / 'scripts/gse337988_coldata.R'), str(rds), str(prefix)], check=True)
    rd = pd.read_csv(str(prefix) + '_rowdata.csv.gz', nrows=1)
    cd = pd.read_csv(str(prefix) + '_coldata.csv.gz', nrows=1)
    if not {'feature', 'Symbol'} <= set(rd) or not {'guide_name', 'assignment_v2', 'MOI'} <= set(cd):
        raise ValueError('DLD1 metadata extraction is incomplete')
    rds.unlink()


def viperturb():
    zenodo_file(18460279, 'genome_wide_manifest.txt', ROOT / 'data/raw/viperturb/genome_wide_manifest.txt')


def ensembl():
    from genes import load_vocab
    pairs = []
    with h5py.File(ROOT / 'data/vcc2025/adata_Training.h5ad') as h:
        pairs.append(pd.DataFrame(dict(symbol=index_of(h, 'var'), ensembl=cat(h, 'var/gene_id'), source='vcc2025_H1')))
    rd = pd.read_csv(ROOT / 'data/raw/gse337988/low_rowdata.csv.gz', dtype=str)
    pairs.append(pd.DataFrame(dict(symbol=rd.Symbol, ensembl=rd.feature, source='gse337988__dld1')))
    om = pd.read_parquet(ROOT / 'data/orion_probe/gene_metadata.parquet')
    pairs.append(pd.DataFrame(dict(symbol=om.gene_name, ensembl=om.ensembl_id, source='orion')))
    # Read only gene identifiers, not the authors' effect estimates. Deduplicate each chunk.
    for chunk in pd.read_csv(ROOT / 'data/raw/feng/GenomeWideScreen_LFC_byGene.tsv.gz', sep='\t',
                             usecols=['Expressed_Gene_Symbol', 'Expressed_Gene_Ens_ID'], chunksize=500_000):
        pairs.append(chunk.drop_duplicates().rename(columns={'Expressed_Gene_Symbol':'symbol',
                                                            'Expressed_Gene_Ens_ID':'ensembl'}).assign(source='feng__hpsc'))
    df = pd.concat(pairs).drop_duplicates()
    df['ensembl'] = df.ensembl.str.split('.').str[0]
    df = df[df.symbol.isin(set(load_vocab())) & df.ensembl.str.startswith('ENSG')]
    df = df.groupby(['ensembl', 'symbol']).source.agg(lambda s: ','.join(sorted(set(s)))).reset_index()
    ambiguous = df.groupby('ensembl').symbol.nunique()
    ambiguous = set(ambiguous[ambiguous > 1].index)
    if ambiguous:
        print(f'WARNING dropping {len(ambiguous)} ambiguous Ensembl IDs', flush=True)
    df = df[~df.ensembl.isin(ambiguous)]
    out = ROOT / 'data/vocab_ensembl.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix('.tmp')
    df.to_csv(part, sep='\t', index=False)
    part.replace(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('step', choices=['h1', 'validate-h1', 'vocabulary', 'orion', 'feng', 'dld1', 'viperturb', 'ensembl'])
    ap.add_argument('--split', choices=SPLITS)
    a = ap.parse_args()
    os.chdir(ROOT)
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Data preparation must run inside Slurm; use the workflow profile')
    if a.step == 'h1':
        if a.split is None:
            ap.error('h1 requires --split')
        h1(a.split)
    elif a.step == 'validate-h1':
        report = validate_h1()
        (ROOT / 'data/vcc2025/validated.json').write_text(json.dumps(report, indent=2) + '\n')
    elif a.step == 'vocabulary':
        from vocabulary import write_vocabulary
        write_vocabulary(ROOT / 'data/vcc2026/gene_names.csv')
    else:
        globals()[a.step]()


if __name__ == '__main__':
    main()
