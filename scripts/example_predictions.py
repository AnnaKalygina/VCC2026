"""Zero-LFC example using only H1 gene names and target labels."""
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
args = ap.parse_args()
validate_h1()
targets = set()
for split in ['Training', 'Validation', 'Test']:
    with h5py.File(ROOT / f'data/vcc2025/adata_{split}.h5ad') as h:
        genes = index_of(h, 'var')
        targets |= set(cat(h, 'obs/target_gene')) - {'non-targeting'}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, genes=genes, perts=np.array(sorted(targets)),
                    lfc=np.zeros((300, len(genes)), np.float32), source='zero-LFC example; no perturbation measurements')
print(out)
