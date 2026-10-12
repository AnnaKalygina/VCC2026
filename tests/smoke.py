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
# Batch-matched reference, per-gene SE and half deltas against brute force: 3 batches, the third
# without controls; batch sums are kept (B <= MAX_BATCH_SUMS).
from stats import MIN_CTRL_BATCH
cb = np.repeat([0, 1], 16)
ctrl_b = np.vstack([ctrl, rng.poisson(80, (2 * MIN_CTRL_BATCH - 32, 24))])
cb = np.r_[cb, np.repeat([0, 1], MIN_CTRL_BATCH - 16)]
lcb = ctrl_b.sum(1)
pb = rng.choice([0, 1, 2], 40, p=[.6, .3, .1])
eb = Engine(24, 2, seed=0, batches=['b0', 'b1'])
eb.set_controls(sp.csr_matrix(ctrl_b), lcb, batch=cb)
eb.add(sp.csr_matrix(pert), lp, groups, batch=pb)
rb = eb.result(np.array([-1, -1]))
cm_b = np.array([(ctrl_b[cb == k] / lcb[cb == k, None]).mean(0) * 1e6 for k in (0, 1)])
for group in (0, 1):
    m = groups == group
    x = pert[m] / lp[m, None] * 1e6
    w = np.array([(pb[m] == k).sum() for k in (0, 1)], float)
    ref = w @ cm_b / w.sum()
    np.testing.assert_allclose(rb['lfc_batchmatched'][group], np.log2((x.mean(0) + 1e-9) / (ref + 1e-9)), rtol=1e-5)
    np.testing.assert_allclose(rb['se_cpm'][group], x.std(0, ddof=1) / np.sqrt(m.sum()), rtol=1e-5)
    assert rb['n_cells_by_batch'][group].tolist() == [int(w[0]), int(w[1]), int((pb[m] == 2).sum())]
    np.testing.assert_allclose(rb['frac_no_ctrl'][group], (pb[m] == 2).mean())
    for k in (0, 1, 2):
        if (pb[m] == k).any():
            np.testing.assert_allclose(rb['mean_cpm_by_batch'][group, k], x[pb[m] == k].mean(0), rtol=1e-5)
    np.testing.assert_allclose(rb['lfc'][group], np.log2((x.mean(0) + 1e-9) / ((ctrl_b / lcb[:, None]).mean(0) * 1e6 + 1e-9)), atol=1e-5)
    assert np.isfinite(rb['half_delta_A'][group]).all() and np.isfinite(rb['half_delta_B'][group]).all()
np.testing.assert_allclose(rb['control']['mean_cpm_by_batch'], cm_b, rtol=1e-5)
np.testing.assert_allclose(rb['control']['se_cpm'], (ctrl_b / lcb[:, None] * 1e6).std(0, ddof=1) / np.sqrt(len(cb)), rtol=1e-5)
# Half deltas reproduce the split-0 reliability correlation over all genes.
r0 = np.corrcoef(rb['half_delta_A'][0], rb['half_delta_B'][0])[0, 1]
np.testing.assert_allclose(r0, rb['reliability_r'][0, 0], rtol=1e-4)
assert 'lfc_batchmatched' not in res and np.isfinite(res['se_cpm']).all()
# Batch-matched half deltas against brute force: each half against the control halves of each batch, weighted
# by that half's own cells per batch (AUDIT N8).
from stats import _half_codes
rank = np.zeros(40, np.int64)
for group in (0, 1):
    rank[groups == group] = np.arange((groups == group).sum())
codes = _half_codes(groups, rank, 0, 8)
lg_p, lg_c = np.log1p(pert / lp[:, None] * 1e4), np.log1p(ctrl_b / lcb[:, None] * 1e4)
for group in (0, 1):
    for h, key in ((0, 'half_delta_bm_A'), (1, 'half_delta_bm_B')):
        cells = (groups == group) & (codes[:, 0] == h)
        ok = np.array([((cb == k) & (eb.c_half[0] == h)).sum() >= MIN_CTRL_BATCH // 2 for k in (0, 1)])
        w = np.array([(pb[cells] == k).sum() for k in (0, 1)]) * ok
        cm = np.array([lg_c[(cb == k) & (eb.c_half[0] == h)].mean(0) if ok[k] else np.zeros(24) for k in (0, 1)])
        np.testing.assert_allclose(rb[key][group], lg_p[cells].mean(0) - w @ cm / w.sum(), atol=1e-5)
r0 = np.corrcoef(rb['half_delta_bm_A'][0], rb['half_delta_bm_B'][0])[0, 1]
np.testing.assert_allclose(r0, rb['reliability_r_batchmatched'][0, 0], rtol=1e-4)
# Gene-major block mode (KOLF, Feng) must equal row mode on the same cells, batches included.
from readers import run_gene_major
allx = sp.vstack([sp.csr_matrix(ctrl_b), sp.csr_matrix(pert)]).tocsc()
is_c = np.r_[np.ones(len(cb), bool), np.zeros(40, bool)]
blocks = [(np.arange(0, 12), allx[:, :12]), (np.arange(12, 24), allx[:, 12:])]
eg = Engine(24, 2, seed=0, batches=['b0', 'b1'])
labels = np.r_[np.array(['b0', 'b1'])[cb], np.array(['b0', 'b1', 'b2'])[pb]]
run_gene_major(iter(blocks), eg, np.r_[lcb, lp].astype(float), np.r_[np.full(len(cb), -1), groups], is_c,
               dict(ctrl_seen=0, ctrl_low_umi=0, pert_seen=0, pert_unlabelled=0, pert_low_umi=0), lambda m: None, 1000, labels)
rg = eg.result(np.array([-1, -1]))
for k in ('mean_cpm', 'lfc', 'nlog10p', 'se_cpm', 'lfc_batchmatched', 'half_delta_A', 'reliability_r', 'mean_cpm_by_batch',
          'half_delta_bm_A', 'reliability_r_batchmatched'):
    np.testing.assert_allclose(rg[k], rb[k], rtol=1e-5, atol=1e-6, err_msg=k)

# One target-label rule for every reader (AUDIT S10).
from labels import TargetLabels
from types import SimpleNamespace
vocab = np.array(['HLA-B', 'HSPE1', 'MOB4', 'AARS1', 'G1'])
tl = TargetLabels(vocab, SimpleNamespace(vocab_of=np.array([3, -1])), np.array(['AARS', 'XYZ']))
for raw, want in [('*', ('', 'sentinel')), ('nan', ('', 'sentinel')), ('HLA-B', ('HLA-B', 'exact')),
                  ('hla-b', ('HLA-B', 'case')), ('HLA_B', ('HLA-B', 'underscore')), ('AARS', ('AARS1', 'deposit_gene_axis')),
                  ('HSPE1_MOB4', ('', 'multi_gene')), ('XYZ', ('XYZ', 'not_in_panel'))]:
    assert tl(raw) == want, (raw, tl(raw))
tl.hgnc = {'MOB4A': ('MOB4', 'previous')}
assert tl('MOB4A') == ('MOB4', 'hgnc_previous') and tl('MOB4') == ('MOB4', 'exact')
rep = tl.report({'XYZ': 7})
assert rep['actions']['sentinel'] == 2 and {r['raw'] for r in rep['changed']} == {'*', 'nan', 'hla-b', 'HLA_B', 'AARS', 'HSPE1_MOB4', 'XYZ', 'MOB4A'}
assert [r['merged_with'] for r in rep['changed'] if r['raw'] == 'MOB4A'] == [['MOB4']]

# Frangieh guides (AUDIT S2): one gene (any of its guides, control guides allowed) or only control guides.
from readers import frangieh_labels
c, lab = frangieh_labels(np.array(['A_1', 'A_1;A_3', 'A_1;B_2', 'NO_SITE_4', 'NO_SITE_4;ONE_NON-GENE_SITE_9',
                                   'A_1;NO_SITE_2', 'A_1;NO_S', np.nan], dtype=object))
assert c.tolist() == [0, 0, 0, 1, 1, 0, 0, 0] and lab.tolist() == ['A', 'A', '', '', '', 'A', '', '']

# hESC guide calling (AUDIT S9): guides summed per gene, top >= 5 UMI, second gene <= 0.2 x top.
from ingest_gse295214 import read_sample
with tempfile.TemporaryDirectory() as name:
    f = Path(name) / 'lane.h5'
    names = ['G1', 'G2', 'A_1', 'A_2', 'B_1', 'non-targeting_1']
    guides = np.array([[5, 3, 0, 0], [10, 0, 3, 0], [10, 0, 2, 0], [4, 0, 0, 0], [0, 0, 0, 6]])
    M = sp.csc_matrix(np.c_[np.ones((5, 2), int), guides].T)
    with h5py.File(f, 'w') as h:
        for k, v in dict(data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape).items():
            h[f'matrix/{k}'] = v
        h['matrix/features/name'] = np.array(names, dtype='S')
        h['matrix/features/feature_type'] = np.array(['Gene Expression'] * 2 + ['CRISPR Guide Capture'] * 4, dtype='S')
    _, X, tgt, rate = read_sample(str(f), 5, 0.2)
    assert tgt.tolist() == ['A', 'A', 'non-targeting'] and X.shape == (3, 2) and rate == 0.6

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
assert len(contexts) == 56 and all(c['reader'] in READERS for c in contexts.values())
assert all('batch_key' in vars(r) for r in READERS.values()), 'every reader states its batch key (or None)'
print(f'PASS: small synthetic statistics, encoder, prediction, cache, H1, and aggregation checks ({time.monotonic()-start:.1f}s)')
