"""Internal all-300 H1 evaluation with block DE and global PDS/MSE."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'scripts'))
import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp
from cell_eval2 import EvalConfig, aggregate_metrics_wide, compute_metrics
from cell_eval2.catalog import CATALOG, PROFILES
from cell_eval2.moments import GroupMoments
from cell_eval2.prep import pseudobulk_bulk_lognorm_with_moments
from cell_eval2.run import dispatch_anndata_metrics
from downloads import digest
from encoder import encode_decoupled, fit_control, fit_depth
from prepare_data import validate_h1
from score import ANCHORS_2026, SCORED, scale

CTRL, PERT_COL = 'non-targeting', 'target'


def load_predictions(path, genes, targets, bulk_gain=.25):
    with np.load(path, allow_pickle=False) as z:
        if not {'genes', 'perts', 'lfc', 'source'} <= set(z.files):
            raise ValueError('Prediction requires genes, perts, lfc, and source')
        g, t = z['genes'].astype(str), z['perts'].astype(str)
        if g.ndim != 1 or t.ndim != 1 or len(set(g)) != len(g) or len(set(t)) != len(t):
            raise ValueError('Prediction axes must be unique one-dimensional arrays')
        np.testing.assert_array_equal(g, genes)
        if set(t) != set(targets) or len(t) != 300:
            raise ValueError('Prediction must contain exactly the 300 H1 targets')
        position = {name: i for i, name in enumerate(t)}
        order = [position[name] for name in targets]
        matrices = {}
        for name in ['lfc'] + (['lfc_bulk'] if 'lfc_bulk' in z.files else []):
            value = z[name]
            if value.shape != (300, len(genes)) or not np.isfinite(value).all():
                raise ValueError(f'{name}: expected finite [300, {len(genes)}] matrix')
            value = value[order].astype(np.float64)
            # The bulk-channel gain applies to a given lfc_bulk too; neither LFC channel is bounded.
            if name == 'lfc_bulk':
                value *= bulk_gain
            matrices[name] = value
        if 'lfc_bulk' not in matrices:
            matrices['lfc_bulk'] = bulk_gain * matrices['lfc']
        if z['source'].ndim != 0:
            raise ValueError('source must be a scalar string')
        return matrices, dict(source=str(z['source']))


def check_run(out, metadata):
    """Reject stale or partial completion instead of silently reusing old scores."""
    path = out / 'metadata.json'
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError(f'{out}: inputs/settings changed; choose a new output folder')
    if (out / 'DONE').exists():
        required = ['metadata.json', 'per_pert.csv', 'agg.csv', 'score.csv', 'summary.json']
        if not all((out / name).is_file() for name in required):
            raise ValueError(f'{out}: incomplete result despite DONE marker')
        return True
    return False


def raw_counts(counts):
    """Preserve original UMI counts; only convert storage to float32 CSR."""
    x = sp.csr_matrix(counts, dtype=np.float32)
    if not np.isfinite(x.data).all() or (x.data < 0).any() or not np.equal(x.data, np.rint(x.data)).all():
        raise ValueError('Expected raw nonnegative integer UMI counts')
    if (np.asarray(x.sum(1)).ravel() <= 0).any():
        raise ValueError('Reference contains a zero-count cell')
    return x


def merge_moments(parts, keys):
    rows = {}
    for m in parts:
        for i, key in enumerate(m.perts):
            value = (m.counts[i], m.sumsq[i], m.jk[i])
            if str(key) in rows:
                np.testing.assert_allclose(rows[str(key)], value, atol=1e-10)
            rows[str(key)] = value
    values = np.array([rows[key] for key in keys])
    return GroupMoments(np.array(keys), values[:, 0], values[:, 1], values[:, 2])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--bulk-gain', type=float, default=.25,
                    help='share of the LFC realized in the pooled pseudobulk (pds, mse); 1 realizes all of it')
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        ap.error('H1 evaluation must run inside Slurm')
    if args.seed < 0:
        ap.error('Require seed >= 0')
    pred_path, out = Path(args.predictions).resolve(), Path(args.out).resolve()
    validate_h1()
    real_paths = [ROOT / f'data/vcc2025/adata_{s}.h5ad' for s in ('Training', 'Validation', 'Test')]
    sources = [ad.read_h5ad(path, backed='r') for path in real_paths]
    try:
        genes = sources[0].var_names.to_numpy().astype(str)
        where = {}
        for index, a in enumerate(sources):
            labels = a.obs['target_gene'].astype(str).to_numpy()
            for target in sorted(set(labels) - {CTRL}):
                where[target] = (index, np.flatnonzero(labels == target))
        targets = sorted(where)
        matrices, pred_info = load_predictions(pred_path, genes, targets, args.bulk_gain)
        cfg = EvalConfig.from_preset('vcc2026')
        cfg.de = replace(cfg.de, backend='pdex', clip_value=None)
        cfg.device = 'cpu'
        cfg.num_threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
        # No implicit reference cache: completed blocks are reused only within this signed run.
        cfg.cache_real = None
        code = [Path(__file__), ROOT / 'src/encoder.py', ROOT / 'src/score.py']
        metadata = dict(predictions_sha256=digest(pred_path), seed=args.seed, count_protocol='original_umi_v1',
                        n_perts=300, n_cells=400, block_size=25, cell_weight_scale=1,
                        bulk_gain=args.bulk_gain, prediction=pred_info, preset=repr(cfg),
                        code_sha256={p.name:digest(p) for p in code},
                        reference=[dict(path=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                   for p in real_paths], cell_eval2_revision='5e64833518a6603a0301cbe28185d49c30f4a986')
        if check_run(out, metadata):
            print(f'Already complete: {out}', flush=True)
            return
        out.mkdir(parents=True, exist_ok=True)
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        labels = sources[0].obs['target_gene'].astype(str).to_numpy()
        ctrl = raw_counts(sources[0].X[np.flatnonzero(labels == CTRL)])
        p, alpha, libs = fit_control(ctrl)
        _, p_pool, alpha = fit_depth(ctrl, p, libs)
        sampling_rng = np.random.default_rng(args.seed)
        selections = {}
        for target in targets:
            _, rows = where[target]
            if len(rows) < 400:
                print(f'WARNING {target}: only {len(rows)} reference cells; keeping all', flush=True)
            selections[target] = rows if len(rows) < 400 else np.sort(sampling_rng.choice(rows, 400, replace=False))
        frames, pm_parts, rm_parts, pbulks, rbulks = [], [], [], {}, {}
        for lo in range(0, 300, 25):
            subset = targets[lo:lo + 25]
            block = out / 'blocks' / str(lo // 25)
            block.mkdir(parents=True, exist_ok=True)
            if (block / 'DONE').exists():
                if not (block / 'de.csv').is_file() or not (block / 'moments.npz').is_file():
                    raise ValueError(f'{block}: incomplete cached block')
                frames.append(pl.read_csv(block / 'de.csv'))
                with np.load(block / 'moments.npz', allow_pickle=False) as z:
                    for prefix, bulks, parts in [('p', pbulks, pm_parts), ('r', rbulks, rm_parts)]:
                        keys = z[prefix + 'keys'].astype(str)
                        if set(keys) != set(subset + [CTRL]):
                            raise ValueError(f'{block}: cached target mismatch')
                        parts.append(GroupMoments(keys, z[prefix+'counts'], z[prefix+'sumsq'], z[prefix+'jk']))
                        for key, value in zip(keys, z[prefix+'bulk']):
                            if key in bulks:
                                np.testing.assert_allclose(bulks[key], value, atol=1e-12)
                            bulks[key] = value.copy()
                continue
            rb, pb, labels = [], [], []
            for i, target in enumerate(subset, lo):
                source_index, _ = where[target]
                x = raw_counts(sources[source_index].X[selections[target]])
                y = sp.csr_matrix(encode_decoupled(p, p_pool, alpha, libs, matrices['lfc'][i],
                                 matrices['lfc_bulk'][i], 400,
                                 np.random.default_rng(np.random.SeedSequence([args.seed, i, 930]))), dtype=np.float32)
                rb.append(x); pb.append(y); labels.extend([target] * x.shape[0])
            real = ad.AnnData(sp.vstack(rb + [ctrl]), obs={PERT_COL:labels + [CTRL] * ctrl.shape[0]})
            pred = ad.AnnData(sp.vstack(pb + [ctrl[:1]]), obs={PERT_COL:np.repeat(subset, 400).tolist() + [CTRL]})
            real.var_names = pred.var_names = genes
            print(f'Scoring {lo + 1}-{lo + 25}/300', flush=True)
            df = compute_metrics(pred, real, config=cfg, metrics=[m for m in PROFILES['vcc2026'] if CATALOG[m].kind == 'de'])
            df.write_csv(block / 'de.csv')
            frames.append(df)
            stored = {}
            for prefix, a, bulks, parts in [('p', pred, pbulks, pm_parts), ('r', real, rbulks, rm_parts)]:
                keys, mean, moments = pseudobulk_bulk_lognorm_with_moments(a, PERT_COL, bulk_target_sum=cfg.bulk_target_sum)
                parts.append(moments)
                stored.update({prefix+'keys':keys, prefix+'bulk':mean, prefix+'counts':moments.counts,
                               prefix+'sumsq':moments.sumsq, prefix+'jk':moments.jk})
                for key, value in zip(keys, mean):
                    bulks[str(key)] = value
            np.savez_compressed(block / 'moments.npz', **stored)
            (block / 'DONE').write_text('DE and global sufficient statistics complete\n')
            del pred, real, rb, pb, x, y
            gc.collect()
        keys = targets + [CTRL]
        rows = dispatch_anndata_metrics([m for m in PROFILES['vcc2026'] if CATALOG[m].kind != 'de'],
            {'bulk_lognorm':(np.array(keys), np.vstack([pbulks[k] for k in keys]))},
            {'bulk_lognorm':(np.array(keys), np.vstack([rbulks[k] for k in keys]))}, genes, cfg,
            comparator='bulk_lognorm', pred_moments={'bulk_lognorm':merge_moments(pm_parts, keys)},
            real_moments={'bulk_lognorm':merge_moments(rm_parts, keys)}, driver='in-memory')
        df = pl.concat(frames + [pl.DataFrame(rows)])
        wide = aggregate_metrics_wide(df)
        mean = wide.filter(pl.col('statistic') == 'mean').row(0, named=True)
        raw = {key:float(mean[key]) for key in SCORED}
        if not all(np.isfinite(v) for v in raw.values()):
            raise ValueError(f'Undefined aggregate metric: {raw}')
        scaled = scale(raw, ANCHORS_2026['b'], ANCHORS_2026['r'])
        df.write_csv(out / 'per_pert.csv'); wide.write_csv(out / 'agg.csv'); scaled.write_csv(out / 'score.csv')
        summary = dict(aggregate=float(scaled['s'].mean()), raw=raw, scaled=dict(zip(scaled['metric'], scaled['s'])))
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (out / 'DONE').write_text('All 300 targets scored; global PDS/MSE; internal midpoint-anchor scaling\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        for a in sources:
            a.file.close()


if __name__ == '__main__':
    main()
