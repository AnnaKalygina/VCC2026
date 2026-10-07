"""scPerturb raw-count selection, symbols, and stimulation-context helpers."""
import re
import numpy as np
import scipy.sparse as sp
from downloads import download as fetch
STUDIES = {"frangieh": ("FrangiehIzar2021_RNA.h5ad", "dc438f53c476a3dd562898654d48ed15", ["frangieh__control", "frangieh__ifng", "frangieh__coculture"]), "adamson": ("AdamsonWeissman2016_GSM2406681_10X010.h5ad", "2fa44ea61a8dd35742af618638ec65fc", ["adamson__k562"])}
def download(study, raw_dir):
    name, checksum, _ = STUDIES[study]
    return fetch(f"https://zenodo.org/records/7416068/files/{name}?download=1", raw_dir / name, md5=checksum)

def log(message):
    print(message, flush=True)

def choose_count_matrix(a):
    if "counts" in a.layers:
        return a.layers["counts"], "layers/counts"
    if a.raw is not None:
        raw = a.raw.X
        values = raw.data if sp.issparse(raw) else np.asarray(raw)
        if np.isfinite(values).all() and np.all(values >= 0) and np.all(values == np.floor(values)):
            log("WARNING selecting raw/X because layers/counts is absent")
            return raw, "raw/X"
    x = a.X
    values = x.data if sp.issparse(x) else np.asarray(x)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("No identifiable raw integer count matrix: layers/counts and raw/X are absent or X is noninteger")
    log("WARNING selecting X as raw counts; no layers/counts or integer raw/X is present")
    return x, "X"

def validate_counts(x, source):
    values = x.data if sp.issparse(x) else np.asarray(x)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError(f"Selected matrix {source} is not finite, nonnegative integer counts")

def select_symbols(a):
    if "gene_name" in a.var:
        return a.var["gene_name"].astype(str).to_numpy(), "var/gene_name"
    if "gene_symbol" in a.var:
        return a.var["gene_symbol"].astype(str).to_numpy(), "var/gene_symbol"
    return a.var_names.astype(str).to_numpy(), "var_names"

def find_perturbations(obs):
    for col in ("perturbation", "target_gene", "gene"):
        if col in obs:
            return col
    raise ValueError(f"No perturbation label column among perturbation/target_gene/gene; available: {list(obs.columns)}")

def map_contexts(study, obs, expected):
    if study != "frangieh":
        return np.full(len(obs), expected[0], dtype=object), {"field": None, "values": {expected[0]: "all cells"}}
    condition_col = "perturbation_2"
    if condition_col not in obs:
        raise ValueError(f"Frangieh requires obs[{condition_col!r}] to preserve stimulation contexts; have {list(obs.columns)}")
    raw = obs[condition_col].astype("string")
    if raw.isna().any():
        raise ValueError(f"Frangieh obs[{condition_col!r}] has {int(raw.isna().sum())} missing values")
    mapped = []
    source_values = {key: set() for key in expected}
    for value in raw.astype(str):
        norm = re.sub(r"[^a-z0-9]", "", value.lower())
        if "ifn" in norm:
            context = "frangieh__ifng"
        elif "coculture" in norm or "cocult" in norm:
            context = "frangieh__coculture"
        elif any(token in norm for token in ("control", "unstimulated", "baseline", "vehicle")):
            context = "frangieh__control"
        else:
            raise ValueError(f"Unrecognized Frangieh condition {value!r}; refusing to pool it")
        if context not in source_values:
            raise ValueError(f"Condition {value!r} maps to unexpected context {context}")
        source_values[context].add(value)
        mapped.append(context)
    absent = [c for c, vals in source_values.items() if not vals]
    if absent:
        raise ValueError(f"Frangieh conditions missing required contexts {absent}; observed {sorted(set(raw.astype(str)))}")
    return np.asarray(mapped, dtype=object), {"field": condition_col, "values": {k: sorted(v) for k, v in source_values.items()}}
