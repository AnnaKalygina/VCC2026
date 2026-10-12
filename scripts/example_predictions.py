"""Zero-LFC example from H1 gene names and target labels, or the H1 summary's own LFCs as an oracle."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/clean'))
import h5py
import numpy as np
from readers import cat, index_of
from prepare_data import validate_h1

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', default='results/example_predictions.npz')
ap.add_argument('--oracle-from', help='summary h5 whose pert/lfc becomes the prediction')
args = ap.parse_args()
validate_h1()
targets = set()
for split in ['Training', 'Validation', 'Test']:
    with h5py.File(ROOT / f'data/vcc2025/adata_{split}.h5ad') as h:
        genes = index_of(h, 'var')
        targets |= set(cat(h, 'obs/target_gene')) - {'non-targeting'}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
perts = np.array(sorted(targets))
lfc, source = np.zeros((300, len(genes)), np.float32), 'zero-LFC example; no perturbation measurements'
if args.oracle_from:
    with h5py.File(args.oracle_from) as h:
        sg = {x.decode(): i for i, x in enumerate(h['genes'][:])}
        st = {x.decode(): i for i, x in enumerate(h['pert/targets'][:])}
        rows, cols = [st[t] for t in perts], np.array([sg.get(g, -1) for g in genes])
        lfc[:, cols >= 0] = np.nan_to_num(h['pert/lfc'][:][rows][:, cols[cols >= 0]])
    print(f'{(cols < 0).sum()} H1 genes not on the summary axis keep zero LFC: {list(genes[cols < 0])}')
    source = f'in-sample oracle: measured LFCs of the H1 cells from {Path(args.oracle_from).name}'
np.savez_compressed(out, genes=genes, perts=perts, lfc=lfc, source=source)
print(out)
