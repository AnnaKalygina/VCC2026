"""Save the per-cell metadata of the 12 CD4 T S3 h5ad files without their counts."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import anndata as ad  # noqa: E402
from add_dispersion import open_url  # noqa: E402

URL = 'https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/{}.assigned_guide.h5ad'
OUT = ROOT / 'data/raw/cd4t_marson2025/obs'


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside Slurm')
    OUT.mkdir(parents=True, exist_ok=True)
    for cond in ('Rest', 'Stim8hr', 'Stim48hr'):
        for donor in (1, 2, 3, 4):
            name = f'D{donor}_{cond}'
            dest = OUT / f'{name}.obs.parquet'
            if dest.exists():
                print(f'have {dest.name}', flush=True)
                continue
            url = URL.format(name)
            h, _ = open_url(url)
            obs = ad.io.read_elem(h['obs']) if hasattr(ad, 'io') else ad.experimental.read_elem(h['obs'])
            shape = [int(x) for x in h['X'].attrs['shape']]
            h.close()
            obs.index.name = 'cell'
            tmp = dest.with_suffix('.tmp')
            obs.to_parquet(tmp)
            tmp.replace(dest)
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
            receipt = dict(url=url, group='obs', n_cells=len(obs), x_shape=shape, columns=list(obs.columns),
                           path=str(dest.relative_to(ROOT)), sha256=sha)
            dest.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(f'wrote {dest.name}: {len(obs):,} cells', flush=True)


if __name__ == '__main__':
    main()
