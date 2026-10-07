"""Small synthetic checks: 24 genes, 32 controls, 40 perturbed cells; no downloads."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('NUMBA_NUM_THREADS', '1')
os.environ.setdefault('POLARS_MAX_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/clean'), str(ROOT / 'scripts')]

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from stats import Engine
from encoder import fit_control, encode_decoupled
from evaluate_h1 import check_run, load_predictions, merge_moments, raw_counts

# Retained reference cells keep every original UMI.
_count_fixture = sp.csr_matrix([[1, 3, 0], [0, 7, 2]])
np.testing.assert_array_equal(raw_counts(_count_fixture).toarray(), _count_fixture.toarray())
from cell_eval2.moments import GroupMoments
from prepare_data import validate_h1
from vocabulary import write_vocabulary
from readers import READERS
import yaml


def rejects(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected invalid input to fail')


start = time.monotonic()
rng = np.random.default_rng(7)
ctrl = rng.poisson(80, (32, 24))
pert = rng.poisson(85, (40, 24))
# Distinct prime library totals avoid pert-only nonzero CPM ties, which the
# inherited streaming MWU approximates (documented in src/clean/stats.py).
primes = [i for i in range(3000, 4000) if all(i % d for d in range(2, int(i**.5) + 1))][:72]
ctrl[:, -1] = np.array(primes[:32]) - ctrl[:, :-1].sum(1)
pert[:, -1] = np.array(primes[32:]) - pert[:, :-1].sum(1)
groups = np.repeat([0, 1], 20)
engine = Engine(24, 2, seed=0)
lc, lp = ctrl.sum(1), pert.sum(1)
engine.set_controls(sp.csr_matrix(ctrl), lc)
engine.add(sp.csr_matrix(pert), lp, groups)
res = engine.result(np.array([-1, -1]))
cpm = ctrl / lc[:, None] * 1e6
for group in (0, 1):
    x = pert[groups == group] / lp[groups == group, None] * 1e6
    np.testing.assert_allclose(res['mean_cpm'][group], x.mean(0), rtol=1e-5)
    np.testing.assert_allclose(res['lfc'][group], np.log2((x.mean(0) + 1e-9) / (cpm.mean(0) + 1e-9)), atol=1e-5)
    p = mannwhitneyu(x, cpm, axis=0, method='asymptotic', use_continuity=True).pvalue
    np.testing.assert_allclose(res['nlog10p'][group], -np.log10(p), atol=1e-5, rtol=1e-5)
p, alpha, libs = fit_control(ctrl)
y = encode_decoupled(p, p, alpha, libs, np.zeros(24), np.zeros(24), 8, rng)
assert y.shape == (8, 24) and (y >= 0).all() and np.equal(y, np.rint(y)).all()

with tempfile.TemporaryDirectory() as name:
    tmp = Path(name)
    write_vocabulary(tmp / 'gene_names.csv')
    assert len((tmp / 'gene_names.csv').read_text().splitlines()) == 18534
    genes, targets = np.array(['G0', 'G1']), np.array([f'T{i}' for i in range(300)])
    pred = tmp / 'pred.npz'
    values = np.tile([20., -40.], (300, 1))
    np.savez(pred, genes=genes, perts=targets[::-1], lfc=values, lfc_bulk=values * 4, source='test')
    matrices, info = load_predictions(pred, genes, targets)
    np.testing.assert_array_equal(matrices['lfc'], values)
    np.testing.assert_array_equal(matrices['lfc_bulk'], values)
    np.savez(pred, genes=genes, perts=np.repeat('T0', 300), lfc=np.zeros((300, 2)), source='test')
    rejects(lambda: load_predictions(pred, genes, targets))
    np.savez(pred, genes=genes, perts=targets, lfc=np.full((300, 2), np.nan), source='test')
    rejects(lambda: load_predictions(pred, genes, targets))
    out = tmp / 'evaluation'; out.mkdir()
    meta = {'seed': 0}
    (out / 'metadata.json').write_text(json.dumps(meta))
    rejects(lambda: check_run(out, {'seed': 1}))
    (out / 'DONE').touch()
    rejects(lambda: check_run(out, meta))
    # All H1 labels and shared controls, represented by tiny metadata-only HDF5s.
    data = tmp / 'data/vcc2025'; data.mkdir(parents=True)
    for split, begin, count in [('Training', 0, 150), ('Validation', 150, 50), ('Test', 200, 100)]:
        with h5py.File(data / f'adata_{split}.h5ad', 'w') as h:
            h.create_dataset('var/_index', data=np.array(['G0', 'G1'], dtype='S'))
            h.create_dataset('obs/_index', data=np.array(['C0'] + [f'C{i}' for i in range(begin+1, begin+count+1)], dtype='S'))
            h.create_dataset('obs/target_gene', data=np.array(['non-targeting'] + [f'T{i}' for i in range(begin, begin+count)], dtype='S'))
            h.create_dataset('X', data=np.ones((count + 1, 2), dtype=np.int32))
    assert sum(v['targets'] for v in validate_h1(tmp).values()) == 300
    with h5py.File(data / 'adata_Test.h5ad', 'r+') as h:
        h['obs/_index'][0] = b'BAD'
    rejects(lambda: validate_h1(tmp))

parts = [GroupMoments(np.array(['T0', 'non-targeting']), np.array([2., 3.]), np.array([4., 5.]), np.array([6., 7.])),
         GroupMoments(np.array(['T1', 'non-targeting']), np.array([8., 3.]), np.array([9., 5.]), np.array([10., 7.]))]
merged = merge_moments(parts, ['T0', 'T1', 'non-targeting'])
assert merged.counts.tolist() == [2., 8., 3.]
contexts = yaml.safe_load((ROOT / 'workflow/config.yaml').read_text())['contexts']
assert len(contexts) == 21 and all(c['reader'] in READERS for c in contexts.values())
print(f'PASS: small synthetic statistics, encoder, prediction, cache, H1, and aggregation checks ({time.monotonic()-start:.1f}s)')
