"""One reader per source family. Every reader exposes

    .genes, .ensembl   native feature axis (ensembl may be None)
    .controls()        -> iterator of (native count chunk [n, n_native] holding ONLY control cells, batches)
    .cells()           -> iterator of (native count chunk, labels, batches) for the PERTURBED cells
                          (labels: target name per row; "" = drop the row)
    .targets           all target labels (available once .controls() is exhausted)
    .provenance        dict merged into the output json
    .batch_key         what the batch labels are (None: the deposit has no batch structure we use,
                       and every chunk's batches is None)

or, for gene-major sources (CSC / gene-row text), `.run_blocks(engine, gene_map, label_index)` plus
`.is_ctrl` and `.batch_labels` (per cell, or None).
Cell-level QC that is specific to a deposit (guide singlets, hashing, ...) lives here and reuses the
original ingest code; the generic vocab-UMI filter (per-context min_umi) is applied by run.py.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CHUNK = 20_000


def _dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def cat(h, key):
    from add_dispersion import cat as _cat           # handles both anndata categorical layouts
    return _cat(h, key)


def index_of(h, grp):
    """obs/var index as strings; legacy anndata names the index column in attrs['_index']."""
    k = h[grp].attrs.get("_index", "_index")
    return cat(h, f"{grp}/{k.decode() if isinstance(k, bytes) else k}")


def x_rows(h, lo, hi, key="X"):
    """Rows [lo, hi) of an AnnData matrix stored as CSR group or dense dataset -> csr float64."""
    node = h[key]
    if isinstance(node, h5py.Dataset):
        return sp.csr_matrix(node[lo:hi].astype(np.float64))
    enc = node.attrs.get("encoding-type", b"csr_matrix")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    assert enc == "csr_matrix", f"{key} is {enc}; rows need CSR or dense"
    ip = node["indptr"][lo:hi + 1].astype(np.int64)
    d = node["data"][ip[0]:ip[-1]]
    ix = node["indices"][ip[0]:ip[-1]]
    n_cols = int(node.attrs["shape"][1])
    return sp.csr_matrix((d.astype(np.float64), ix, ip - ip[0]), shape=(hi - lo, n_cols))


def obs_strings(h, key):
    """An obs column as strings, whether categorical, string or numeric."""
    node = h[key]
    if isinstance(node, h5py.Dataset) and node.dtype.kind in "iuf":
        return node[:].astype(np.int64).astype(str) if node.dtype.kind in "iu" else node[:].astype(str)
    return cat(h, key)


def check_counts(X, where):
    """Refuse non-integer matrices: everything downstream assumes raw UMI counts. run.py calls it on every chunk
    and run_gene_major on every gene block."""
    d = X.data[:2_000_000]
    bad = int((d != np.round(d)).sum())
    if bad or (d < 0).any():
        raise SystemExit(f"{where}: {bad} non-integer / negative values in a sample -> not raw counts")


# ----------------------------------------------------------------------------------- VCC2025 H1
class H1:
    """Train + val + test as ONE context. The three splits share the identical 38,176 NT controls
    (verified: same obs_names), so controls are read once, from Training."""
    SPLITS = ("Training", "Validation", "Test")
    CTRL = "non-targeting"
    batch_key = "obs/batch (48 Flex batches)"

    def __init__(self, tmp=None):
        self.paths = [ROOT / f"data/vcc2025/adata_{s}.h5ad" for s in self.SPLITS]
        with h5py.File(self.paths[0], "r") as f:
            self.genes = _dec(f["var/_index"][:])
            self.ensembl = _dec(f["var/gene_id"][:])
        ids, tg = [], set()
        for p in self.paths:
            with h5py.File(p, "r") as f:
                assert list(_dec(f["var/_index"][:])) == list(self.genes), f"{p}: gene axis differs"
                lab = cat(f, "obs/target_gene")
                ids.append(set(_dec(f["obs/_index"][:])[lab == self.CTRL]))
                tg |= set(lab) - {self.CTRL}
        if not (ids[0] == ids[1] == ids[2]):
            raise SystemExit("H1 splits do not share one control pool -- merging would double count")
        self.targets = sorted(tg)
        self.provenance = dict(source="data/vcc2025/adata_{Training,Validation,Test}.h5ad",
                               note="controls shared by all splits, read once from Training")

    def controls(self):
        with h5py.File(self.paths[0], "r") as f:
            lab, bat = cat(f, "obs/target_gene"), cat(f, "obs/batch")
            for lo in range(0, len(lab), CHUNK):
                X = x_rows(f, lo, min(lo + CHUNK, len(lab)))
                k = np.flatnonzero(lab[lo:lo + CHUNK] == self.CTRL)
                yield X[k], bat[lo:lo + CHUNK][k]

    def cells(self):
        for p in self.paths:
            with h5py.File(p, "r") as f:
                lab, bat = cat(f, "obs/target_gene"), cat(f, "obs/batch")
                for lo in range(0, len(lab), CHUNK):
                    X = x_rows(f, lo, min(lo + CHUNK, len(lab)))
                    lb = lab[lo:lo + CHUNK]
                    k = np.flatnonzero(lb != self.CTRL)
                    yield X[k], lb[k], bat[lo:lo + CHUNK][k]




# ----------------------------------------------------------------------------------- shared helpers
def raw_file(source, name):
    """data/raw/<source>/<name> as fetched by scripts/fetch_raw.py, with its download receipt. Summaries never
    download (LeoMed compute nodes are offline) and never delete raw files."""
    path = ROOT / "data/raw" / source / name
    receipt = ROOT / "data/provenance" / (hashlib.sha256(str(path.resolve()).encode()).hexdigest() + ".json")
    if not path.exists() or not receipt.exists():
        raise SystemExit(f"{path}: not fetched; run scripts/fetch_raw.py first")
    r = json.loads(receipt.read_text())
    if path.stat().st_size != r["bytes"]:
        raise SystemExit(f"{path}: size differs from its download receipt")
    return path, r


def prefetched(items, fetch):
    """Yield fetch(item) results in order while the NEXT item downloads in a background thread."""
    nxt = None
    for k, it in enumerate(items):
        if nxt is None:
            cur = fetch(it)
        else:
            th, box = nxt
            th.join()
            if "err" in box:
                raise box["err"]
            cur = box["out"]
        if k + 1 < len(items):
            box = {}

            def run(it2=items[k + 1], box=box):
                try:
                    box["out"] = fetch(it2)
                except BaseException as e:      # re-raised in the main thread
                    box["err"] = e
            th = threading.Thread(target=run)
            th.start()
            nxt = (th, box)
        yield cur


class GeneMajor:
    """Gene-major sources (CSC / gene-row text): collect native gene columns until every native
    column of a measured gene has arrived (duplicates are summed), then hand out blocks of complete
    measured genes as (cols, csc [n_cells x b])."""

    def __init__(self, gm, n_cells, block=512):
        self.gm, self.n, self.block = gm, n_cells, block
        M = gm.M.tocsr()
        self.col_of = np.full(M.shape[0], -1)
        r, c = M.nonzero()
        self.col_of[r] = c
        self.need = np.bincount(c, minlength=M.shape[1])
        self.have = np.zeros_like(self.need)
        self.parts = {}
        self.ready = []

    def add(self, native_j, rows, vals):
        """rows/vals: non-zero cell indices and counts of native column native_j. -> list of blocks."""
        c = self.col_of[native_j]
        if c < 0:
            return []
        self.parts.setdefault(c, []).append((np.asarray(rows, np.int64), np.asarray(vals, np.float64)))
        self.have[c] += 1
        if self.have[c] == self.need[c]:
            self.ready.append(c)
        return self.flush() if len(self.ready) >= self.block else []

    def flush(self, final=False):
        if not self.ready or (not final and len(self.ready) < self.block):
            return []
        cols = np.array(self.ready, np.int64)
        ptr, ri, vv = [0], [], []
        for c in cols:
            p = self.parts.pop(c)
            r = np.concatenate([x[0] for x in p])
            v = np.concatenate([x[1] for x in p])
            if len(p) > 1:                                 # duplicates -> sum
                u, inv = np.unique(r, return_inverse=True)
                r, v = u, np.bincount(inv, v)
            ri.append(r)
            vv.append(v)
            ptr.append(ptr[-1] + len(r))
        self.ready = []
        X = sp.csc_matrix((np.concatenate(vv), np.concatenate(ri), np.array(ptr)), shape=(self.n, len(cols)))
        X.sort_indices()
        return [(cols, X)]

    def check_done(self):
        assert not self.parts and not self.ready and (self.have == self.need).all(), "gene-major stream incomplete"


def run_gene_major(blocks, eng, L, grp, is_ctrl, qc, log, min_umi, batch_labels=None):
    """Feed complete gene blocks to the engine. L: vocab depth per cell (from a first pass),
    grp: target index per cell (-1 = drop), is_ctrl: control mask, batch_labels: per cell (None without
    batches). Applies the UMI filter."""
    from stats import batch_codes
    bat = None if batch_labels is None else batch_codes(batch_labels, eng.batches)
    ok = L >= min_umi
    qc["ctrl_seen"] += int(is_ctrl.sum())
    qc["ctrl_low_umi"] += int((is_ctrl & ~ok).sum())
    qc["pert_seen"] += int((~is_ctrl).sum())
    qc["pert_unlabelled"] += int((~is_ctrl & (grp < 0)).sum())
    qc["pert_low_umi"] += int((~is_ctrl & (grp >= 0) & ~ok).sum())
    crow = np.flatnonzero(is_ctrl & ok)
    prow = np.flatnonzero(~is_ctrl & (grp >= 0) & ok)
    log(f"gene-major: {len(crow):,} controls, {len(prow):,} perturbed cells after QC")
    first = True
    for k, (cols, X) in enumerate(blocks):
        X = X.tocsr()
        check_counts(X, f"gene block {k}")
        eng.set_controls(X[crow], L[crow], cols, batch=None if bat is None else bat[crow])
        eng.reset_ranks()
        eng.add(X[prow], L[prow], grp[prow], cols, count_cells=first, batch=None if bat is None else bat[prow])
        first = False
        if k % 10 == 0:
            log(f"  gene block {k}: {len(cols)} genes")
    assert not first, "no gene block was produced"


# ----------------------------------------------------------------------------------- scPerturb
class ScPerturb:
    """Zenodo 7416068 h5ad (md5-pinned by scripts/summarize_scperturb.py). Single-gene perturbations only."""
    batch_key = None

    def __init__(self, study, context, tmp):
        sys.path.insert(0, str(ROOT / "scripts"))
        import anndata as ad
        import summarize_scperturb as S
        path, receipt = raw_file("scperturb", S.STUDIES[study][0])
        if receipt["upstream_md5"] != S.STUDIES[study][1]:
            raise SystemExit(f"{path}: receipt md5 {receipt['upstream_md5']} is not the pinned {S.STUDIES[study][1]}")
        a = ad.read_h5ad(path)
        x, self.matrix_field = S.choose_count_matrix(a)
        S.validate_counts(x, self.matrix_field)
        self.X = sp.csr_matrix(x)
        self.genes, _ = S.select_symbols(a)
        self.ensembl = None
        col = S.find_perturbations(a.obs)
        ctx, cinfo = S.map_contexts(study, a.obs, S.STUDIES[study][2])
        raw = a.obs[col].astype("string")
        if study == "adamson":
            is_ctrl = raw.str.contains(r"\(", regex=True, na=False).to_numpy()   # parenthesised control constructs
            tg = raw.fillna("").str.split("_").str[0].to_numpy()                   # strip plasmid id
        else:
            is_ctrl = raw.str.lower().str.strip().isin(S.CONTROL_LABELS).to_numpy()
            tg = raw.fillna("").str.strip().to_numpy()
        compound = pd.Series(tg).str.contains(r"[+;,]").to_numpy()
        lab = np.where(is_ctrl, "", np.where(compound, "", tg)).astype(object)
        inctx = ctx == context
        self.ctrl_rows = np.flatnonzero(inctx & is_ctrl)
        self.pert_rows = np.flatnonzero(inctx & ~is_ctrl)
        self.lab = lab
        self.targets = sorted(set(lab[self.pert_rows]) - {""})
        self.provenance = dict(source=S.RECORD.format(S.STUDIES[study][0]), md5=S.STUDIES[study][1],
                               matrix=self.matrix_field, perturbation_col=col, condition=cinfo,
                               n_compound_dropped=int((inctx & ~is_ctrl & compound).sum()), raw=receipt)

    def controls(self):
        for lo in range(0, len(self.ctrl_rows), CHUNK):
            yield self.X[self.ctrl_rows[lo:lo + CHUNK]], None

    def cells(self):
        for lo in range(0, len(self.pert_rows), CHUNK):
            r = self.pert_rows[lo:lo + CHUNK]
            yield self.X[r], self.lab[r], None


# ----------------------------------------------------------------------------------- Feng hPSC
class Feng:
    """figshare 27989294 dense gene x cell csv.gz, one screen per context. Controls = `unassigned` +
    `NonTarget` (the authors' definition: 499,998 control cells; only ~48 are NonTarget, guide
    assignment succeeded for ~30% of cells, so `unassigned` is mostly cells with an undetected guide
    -- an average over all knockdowns; the user-confirmed choice)."""
    batch_key = "pool (screen, pool, day) x cell line: the authors' cell-line and inlet adjustment"

    def __init__(self, screen, tmp):
        sys.path.insert(0, str(ROOT / "src"))
        import ingest_feng as m
        self.m = m
        name, fid, meta = m.SCREENS[screen]
        self.path, receipt = raw_file("feng", name)
        self.meta_path = ROOT / m.RAW / meta
        feats = []
        for ft, _ in self._rows(parse=False):
            feats.append(ft)
        self.feats = feats
        parts = [f.split(":") for f in feats]
        self.genes = np.array([p[1] for p in parts])
        self.ensembl = np.array([p[0] for p in parts])
        meta = m.load_meta(str(self.meta_path), self.cells_hdr)
        self.lab = meta["label"].to_numpy().astype(str)
        self.is_ctrl = self.lab == m.CTRL
        self.batch_labels = meta["stratum"].to_numpy().astype(str)
        self.targets = sorted(set(self.lab[~self.is_ctrl]))
        self.provenance = dict(source=f"figshare file {fid} ({name})", raw=receipt, meta=str(self.meta_path),
                               control="Guide_Call in {unassigned, NonTarget} (authors' definition)",
                               n_nontarget=int(meta["Guide_Call"].astype(str).str.startswith("NonTarget").sum()))

    def _rows(self, parse=True):
        """Stream Gene-Expression rows -> (feature, (idx, vals) or None)."""
        proc = subprocess.Popen(["pigz", "-dc", "-p", "3", str(self.path)], stdout=subprocess.PIPE, bufsize=1 << 24)
        f = proc.stdout
        self.cells_hdr = np.array(f.readline().rstrip(b"\r\n").decode().split(",")[1:])
        for line in f:
            if not line.strip():
                continue
            if not parse:
                ft = line[:line.index(b",")].decode()
                if ft.endswith(":Gene-Expression"):
                    yield ft, None
                continue
            ft, ix, v = self.m.parse_row(line, len(self.cells_hdr))
            if ft.endswith(":Gene-Expression"):
                yield ft, (ix, v)
        if proc.wait() != 0:
            raise SystemExit(f"pigz failed on {self.path}")

    def run_blocks(self, eng, gm, t_of, qc, log, min_umi):
        n = len(self.cells_hdr)
        L = np.zeros(n)
        mapped = np.zeros(len(self.feats), bool)
        mapped[np.unique(gm.M.nonzero()[0])] = True
        for r, (ft, (ix, v)) in enumerate(self._rows()):
            assert ft == self.feats[r], "row order changed between passes"
            if mapped[r]:
                L[ix] += v
        log(f"feng pass L done: median vocab depth {np.median(L):,.0f}")
        ti = {t: k for t, k in zip(self.targets, t_of)}
        grp = np.array([ti.get(t, -1) for t in self.lab], np.int64)
        gmj = GeneMajor(gm, n)

        def blocks():
            for r, (ft, (ix, v)) in enumerate(self._rows()):
                yield from gmj.add(r, ix, v)
            yield from gmj.flush(final=True)
            gmj.check_done()
        run_gene_major(blocks(), eng, L, grp, self.is_ctrl, qc, log, min_umi, self.batch_labels)


# ----------------------------------------------------------------------------------- GSE337988 DLD1
class DLD1:
    """GEO GSE337988 Low-MOI dense h5 + colData; cell QC as ingest_gse337988.cell_table (hashtag + guide
    singlets, second caller agrees, no NTC_MENG). Groups are TSS groups; post_result keeps one TSS per
    gene: the strongest own-gene knockdown among TSS groups with >= 20 cells (else the largest group)."""
    URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE337nnn/GSE337988/suppl/GSE337988_sublib2_processed_objects_Low_assays.h5"
    batch_key = "colData Sample (10x channel)"

    def __init__(self, tmp, procs=4):
        import ingest_gse337988 as m
        self.m, self.procs = m, procs
        self.path, receipt = raw_file("gse337988", self.URL.rsplit("/", 1)[1])
        m.H5 = str(self.path)                               # _read_block (forked workers) reads m.H5
        self.rows, code, names, self.gene_of, _ = m.cell_table(min_cells=1)
        self.lab = names[code]
        self.bat = pd.read_csv(ROOT / m.COLDATA, usecols=["Sample"], dtype=str)["Sample"].to_numpy()[self.rows]
        rd = pd.read_csv(ROOT / m.ROWDATA, dtype=str, keep_default_na=False)
        self.genes, self.ensembl = rd["Symbol"].to_numpy(), rd["feature"].to_numpy()
        with h5py.File(self.path) as f:
            assert f["assay001"].shape[1] == len(rd), "rowData does not match the h5 gene axis"
        self.targets = sorted(set(self.lab) - {m.CTRL})
        self.tss_gene = dict(zip(names, self.gene_of))
        self.provenance = dict(source=self.URL, raw=receipt, coldata=m.COLDATA,
                               qc="ingest_gse337988.cell_table(min_cells=1)")

    def _stream(self, sel):
        import multiprocessing as mp
        m = self.m
        with h5py.File(self.path) as f:
            n_rows, G = f["assay001"].shape
        keep = np.zeros(n_rows, bool)
        keep[self.rows[sel]] = True
        lab, bat = self.lab[sel], self.bat[sel]
        starts = [lo for lo in range(0, n_rows, m.BLOCK) if keep[lo:lo + m.BLOCK].any()]
        off = 0
        with mp.get_context("fork").Pool(self.procs, initializer=m._init, initargs=(keep,)) as pool:
            for lo, lens, c, v, _ in pool.imap(m._read_block, starts, chunksize=4):
                X = sp.csr_matrix((v.astype(np.float64), c.astype(np.int32), np.r_[0, np.cumsum(lens)]), shape=(len(lens), G))
                yield X, lab[off:off + len(lens)], bat[off:off + len(lens)]
                off += len(lens)
        assert off == int(sel.sum()), "DLD1 stream lost rows"

    def controls(self):
        for X, _, bat in self._stream(self.lab == self.m.CTRL):
            yield X, bat

    def cells(self):
        yield from self._stream(self.lab != self.m.CTRL)

    def target_symbol(self, t):
        return self.tss_gene[t]

    def post_result(self, utg, res, own_idx):
        """utg: TSS group names; own_idx: measured index of each group's own gene. -> one row per gene."""
        genes = np.array([self.tss_gene[t] for t in utg])
        keep = []
        for g in np.unique(genes):
            k = np.flatnonzero(genes == g)
            big = k[res["n_cells"][k] >= 20]
            if len(big) and own_idx[big[0]] >= 0:
                keep.append(big[np.argmin(res["lfc"][big, own_idx[big[0]]])])
            else:
                keep.append(k[np.argmax(res["n_cells"][k])])
        keep = np.array(sorted(keep))
        self.provenance["tss_chosen"] = {genes[i]: utg[i] for i in keep}
        return genes[keep], keep


# ----------------------------------------------------------------------------------- GSE295214 hESC
class HESC:
    """GEO GSE295214: 85 Cell Ranger h5 from data/raw, guide calling as
    ingest_gse295214.read_sample (top construct >= 5 UMI and >= 50% of guide UMIs)."""
    batch_key = "GEO sample file (10x lane)"

    N_SAMPLES = 85

    def __init__(self, tmp=None):
        import ingest_gse295214 as m
        self.m = m
        names = sorted(p.name for p in (ROOT / "data/raw/gse295214").glob("GSM*_filtered_feature_bc_matrix.h5"))
        if len(names) != self.N_SAMPLES:
            raise SystemExit(f"GSE295214: {len(names)} sample matrices on disk, expected {self.N_SAMPLES}")
        self.files = [raw_file("gse295214", n)[0] for n in names]
        with h5py.File(self.files[0]) as h:
            ft = h["matrix/features/feature_type"][:].astype(str)
            self.genes = h["matrix/features/name"][:].astype(str)[ft == "Gene Expression"]
            self.ensembl = h["matrix/features/id"][:].astype(str)[ft == "Gene Expression"]
        self.CTRL = "non-targeting"
        self.targets = None
        self.provenance = dict(source="GEO GSE295214 (85 filtered_feature_bc_matrix.h5)", guide_calling="min_umi 5, min_frac 0.5",
                               n_samples=len(self.files))

    def _iter(self):
        for p in self.files:
            g, X, tgt, rate = self.m.read_sample(str(p), 5, 0.5)
            assert list(g) == list(self.genes), f"{p}: gene axis differs"
            yield X, tgt, Path(p).name

    def controls(self):
        tg = set()
        for X, tgt, lane in self._iter():
            tg |= set(tgt) - {self.CTRL}
            k = np.flatnonzero(tgt == self.CTRL)
            yield X[k], np.full(len(k), lane)
        self.targets = sorted(tg)

    def cells(self):
        for X, tgt, lane in self._iter():
            k = np.flatnonzero(tgt != self.CTRL)
            yield X[k], tgt[k], np.full(len(k), lane)


# ----------------------------------------------------------------------------------- VIPerturb K562
class VIPerturb:
    """Zenodo 18460279 genome_wide_bin{A,B,C}.RDS -> scripts/viperturb_extract.R triplets (one bin on disk at a
    time). All three bins carry the SAME non-targeting cells, so controls come from bin A only."""
    batch_key = None
    ZEN = "https://zenodo.org/api/records/18460279"

    def __init__(self, tmp):
        import ingest_viperturb as m
        self.m, self.tmp = m, Path(tmp)
        self._extract("A")
        self.genes = np.array(open(self.tmp / "binA/genes.txt").read().splitlines())
        self.ensembl = None
        man = pd.read_csv(ROOT / m.RAW / "genome_wide_manifest.txt", sep=r"\s+")   # targets of all 3 bins
        self.targets = sorted(set(man["gene"].astype(str)) - {m.SRC_CTRL})
        self.provenance = dict(source=self.ZEN, files=[f"genome_wide_bin{b}.RDS" for b in m.BINS],
                               note="controls (NO-TARGET) taken once from bin A")

    def _extract(self, b):
        d = self.tmp / f"bin{b}"
        if (d / "info.txt").exists():
            return d
        rds, _ = raw_file("viperturb", f"genome_wide_bin{b}.RDS")
        subprocess.run(["Rscript", "scripts/viperturb_extract.R", str(rds), str(d)], check=True, cwd=ROOT)
        return d

    def _bin(self, b, want_ctrl):
        """Row chunks of one bin from memory-mapped triplets (a bin is ~20 GB of int32 on disk)."""
        d = self._extract(b)
        info = dict(line.split(" ", 1) for line in open(d / "info.txt").read().splitlines())
        G, N = int(info["n_genes"]), int(info["n_cells"])
        genes = np.array(open(d / "genes.txt").read().splitlines())
        assert list(genes) == list(self.genes), f"bin {b}: gene axis differs"
        p = np.fromfile(d / "p.bin", np.int32).astype(np.int64)
        i = np.memmap(d / "i.bin", np.int32, "r")
        x = np.memmap(d / "x.bin", np.int32, "r")
        cells = np.array(open(d / "cells.txt").read().splitlines())
        meta = pd.read_csv(d / "meta.csv", low_memory=False)
        meta["cell"] = meta["cell"].astype(str)
        lab = self.m.labels(meta.set_index("cell").loc[cells]).astype(str)
        sel = (lab == self.m.CTRL) if want_ctrl else (lab != self.m.CTRL)
        for lo in range(0, N, CHUNK):
            hi = min(lo + CHUNK, N)
            k = np.flatnonzero(sel[lo:hi])
            if not len(k):
                continue
            X = sp.csr_matrix((np.asarray(x[p[lo]:p[hi]], np.float64), np.asarray(i[p[lo]:p[hi]]), p[lo:hi + 1] - p[lo]),
                              shape=(hi - lo, G))[k]
            yield X, lab[lo:hi][k]

    def controls(self):
        for X, _ in self._bin("A", True):
            yield X, None

    def cells(self):
        for b in self.m.BINS:
            for X, lab in self._bin(b, False):
                yield X, lab, None
            shutil.rmtree(self.tmp / f"bin{b}")

    def cleanup(self):
        shutil.rmtree(self.tmp)                            # extracted triplets only; the RDS files stay


# ----------------------------------------------------------------------------------- X-Atlas/Orion
class Orion:
    """HF Xaira-Therapeutics/X-Atlas-Orion parquet shards (revision pinned in scripts/fetch_raw.py), read shard by
    shard twice (controls first; every shard holds ~5% controls), next shard prefetched. pass_guide_filter == 1 only."""
    CTRL = "Non-Targeting"
    batch_key = "parquet shard (one GEM sample each)"

    def __init__(self, line, n_shards, tmp):
        self.line, self.n = line, n_shards
        on_disk = sorted((ROOT / "data/raw/orion").glob(f"{line}_Batch*.parquet"))
        expected = {f"{line}_Batch{i}.parquet" for i in range(1, n_shards + 1)}
        if {p.name for p in on_disk} != expected:
            raise SystemExit(f"Orion {line}: shards on disk differ from Batch1..{n_shards}")
        self.receipts = {n: raw_file("orion", n)[1] for n in sorted(expected)}
        gm = pd.read_parquet(ROOT / "data/orion_probe/gene_metadata.parquet").sort_values("gene_token_id")
        assert (gm.gene_token_id.to_numpy() == np.arange(len(gm))).all()
        self.genes, self.ensembl = gm.gene_name.astype(str).to_numpy(), gm.ensembl_id.astype(str).to_numpy()
        self.targets = None
        self.provenance = dict(source=self.receipts[f"{line}_Batch1.parquet"]["url"].replace("Batch1.", f"Batch{{1..{n_shards}}}."),
                               line=line, sha256={n: r["sha256"] for n, r in self.receipts.items()})

    def _fetch(self, i):
        import ingest_orion_lfc as o
        p = ROOT / "data/raw/orion" / f"{self.line}_Batch{i}.parquet"
        off, cols, raw, tgt = o.read_shard(str(p))
        import pyarrow as pa
        pa.default_memory_pool().release_unused()          # arrow's pool otherwise keeps every shard's peak
        return sp.csr_matrix((raw.astype(np.float64), cols, off), shape=(len(tgt), len(self.genes))), tgt, p.stem

    def _iter(self):
        yield from prefetched(list(range(1, self.n + 1)), self._fetch)

    def controls(self):
        tg = set()
        for X, tgt, shard in self._iter():
            tg |= set(tgt) - {self.CTRL}
            k = np.flatnonzero(tgt == self.CTRL)
            yield X[k], np.full(len(k), shard)
        self.targets = sorted(tg)

    def cells(self):
        for X, tgt, shard in self._iter():
            k = np.flatnonzero(tgt != self.CTRL)
            yield X[k], tgt[k], np.full(len(k), shard)


# ----------------------------------------------------------------------------------- Replogle / Nadig
class ReplogleRows:
    """Replogle 2022 (figshare 20029387) and Nadig 2025 (GEO GSE264667) *_raw_singlecell_01.h5ad, read twice
    in row chunks from data/raw. Deposits keep only genes > 0.01 UMI/cell; the rest are NA.
    Controls: obs/gene == 'non-targeting'. GWPS promoter variants are pooled by gene symbol (obs/gene)."""
    CTRL = "non-targeting"
    batch_key = "obs/gem_group"

    def __init__(self, source, name, tmp=None):
        self.path, receipt = raw_file(source, name)
        self._open = lambda: h5py.File(self.path, "r")
        h = self._open()
        self.lab = cat(h, "obs/gene")
        self.bat = obs_strings(h, "obs/gem_group")
        idx = index_of(h, "var")
        self.genes = cat(h, "var/gene_name") if "gene_name" in h["var"] else idx
        self.ensembl = idx if str(idx[0]).startswith("ENSG") else None
        self.targets = sorted(set(self.lab) - {self.CTRL})
        assert self.CTRL in set(self.lab), "no non-targeting cells"
        self.n = len(self.lab)
        self.provenance = dict(source=receipt["url"], raw=receipt, control="obs/gene == non-targeting")
        h.close()

    def _iter(self):
        h = self._open()
        for lo in range(0, self.n, CHUNK):
            hi = min(lo + CHUNK, self.n)
            X = x_rows(h, lo, hi)
            yield X, self.lab[lo:hi], self.bat[lo:hi]
        h.close()

    def controls(self):
        for X, lb, bat in self._iter():
            k = np.flatnonzero(lb == self.CTRL)
            yield X[k], bat[k]

    def cells(self):
        for X, lb, bat in self._iter():
            k = np.flatnonzero(lb != self.CTRL)
            yield X[k], lb[k], bat[k]


# ----------------------------------------------------------------------------------- KOLF2.1J
class Kolf:
    """figshare+ 27261219 KOLF_Pan_Genome_QC_Filtered.h5ad (189 GB), `layers/counts` CSC, read from data/raw
    in column blocks: pass 1 = vocab depth per cell, pass 2 = the statistics, gene block by gene block.
    Controls: obs/perturbed == 'False' (the deposit's own non-targeting flag); targets obs/gene_target.
    Batch column to be chosen once the deposit is on disk."""
    batch_key = None
    batch_labels = None

    def __init__(self, tmp=None, max_nnz=24_000_000):
        from add_dispersion import KOLF_ART, KOLF_FILE
        self.path, receipt = raw_file("kolf", KOLF_FILE)
        self._open = lambda: h5py.File(self.path, "r")
        self.max_nnz = max_nnz
        h = self._open()
        self.is_ctrl = cat(h, "obs/perturbed") == "False"
        self.lab = cat(h, "obs/gene_target")
        print(f"kolf: {len(self.lab):,} cells, {int(self.is_ctrl.sum()):,} controls; gene_target of controls: "
              f"{pd.Series(self.lab[self.is_ctrl]).value_counts().head(3).to_dict()}", flush=True)
        self.genes = index_of(h, "var")
        self.ensembl = cat(h, "var/gene_ids") if "gene_ids" in h["var"] else None
        self.indptr = h["layers/counts/indptr"][:].astype(np.int64)
        assert len(self.indptr) == len(self.genes) + 1, "layers/counts is not CSC over var"
        self.targets = sorted(set(self.lab[~self.is_ctrl]))
        self.provenance = dict(source=f"figshare+ {KOLF_ART}/{KOLF_FILE}", raw=receipt, control="obs/perturbed == 'False'")
        h.close()

    def _columns(self, wanted):
        """Yield (native column j, row idx, values) for wanted columns, reading contiguous nnz blocks."""
        h = self._open()
        data, idx = h["layers/counts/data"], h["layers/counts/indices"]
        ip, G = self.indptr, len(self.genes)
        g0 = 0
        while g0 < G:
            g1 = g0 + 1
            while g1 < G and ip[g1 + 1] - ip[g0] <= self.max_nnz:
                g1 += 1
            if wanted[g0:g1].any():
                lo, hi = int(ip[g0]), int(ip[g1])
                d, ix = data[lo:hi].astype(np.float64), idx[lo:hi].astype(np.int64)
                for j in range(g0, g1):
                    if wanted[j]:
                        a, b = int(ip[j]) - lo, int(ip[j + 1]) - lo
                        yield j, ix[a:b], d[a:b]
            g0 = g1
        h.close()

    def run_blocks(self, eng, gm, t_of, qc, log, min_umi):
        n = len(self.lab)
        wanted = np.zeros(len(self.genes), bool)
        wanted[np.unique(gm.M.nonzero()[0])] = True
        L = np.zeros(n)
        for k, (j, r, v) in enumerate(self._columns(wanted)):
            L += np.bincount(r, v, minlength=n)
            if k % 2000 == 0:
                log(f"  kolf pass L: column {j:,}/{len(self.genes):,}")
        log(f"kolf pass L done: median vocab depth {np.median(L):,.0f}")
        ti = dict(zip(self.targets, t_of))
        grp = np.array([-1 if c else ti.get(t, -1) for t, c in zip(self.lab, self.is_ctrl)], np.int64)
        gmj = GeneMajor(gm, n)

        def blocks():
            for j, r, v in self._columns(wanted):
                yield from gmj.add(j, r, v)
            yield from gmj.flush(final=True)
            gmj.check_done()
        run_gene_major(blocks(), eng, L, grp, self.is_ctrl, qc, log, min_umi)


# ----------------------------------------------------------------------------------- CD4+ T (Marson 2025)
class CD4T:
    """Genome-scale CRISPRi in primary CD4+ T cells, 4 donors pooled into one context per condition. Counts: GEO
    GSE314342 per-lane Cell Ranger matrices (one file per lane x donor x condition; the same cells and counts as
    the authors' S3 h5ad files). Labels: the authors' S3 obs tables (scripts/fetch_cd4t_obs.py). Cells:
    guide_group == 'targeting single sgRNA' and not low_quality; controls guide_type == 'non-targeting';
    targets perturbed_gene_name. Batch: donor."""
    DONORS = ("D1", "D2", "D3", "D4")
    RAW = ROOT / "data/raw/cd4t_marson2025"
    CTRL = "\0ctrl"
    batch_key = "donor"

    def __init__(self, cond, tmp=None):
        self.files, self.meta, receipts = [], [], {}
        for d in self.DONORS:
            obs_path = self.RAW / "obs" / f"{d}_{cond}.obs.parquet"
            r = json.loads(obs_path.with_suffix(".json").read_text())
            if hashlib.sha256(obs_path.read_bytes()).hexdigest() != r["sha256"]:
                raise SystemExit(f"{obs_path}: differs from its receipt")
            receipts[obs_path.name] = r
            o = pd.read_parquet(obs_path, columns=["lane_id", "guide_group", "guide_type", "low_quality",
                                                   "perturbed_gene_name"])
            ok = (o.guide_group == "targeting single sgRNA").to_numpy() & ~o.low_quality.astype(bool).to_numpy()
            lab = np.where(ok & (o.guide_type == "non-targeting").to_numpy(), self.CTRL,
                           np.where(ok, o.perturbed_gene_name.astype(str).to_numpy(), ""))
            barcode = o.index.str.split("_").str[0].to_numpy()
            lane = o.lane_id.astype(str).str.replace("CD4i_", "", regex=False).to_numpy()
            for ln in sorted(set(lane)):
                hit = sorted(self.RAW.glob(f"GSM*_CD4i_{ln}_{d}_{cond}_sample_filtered_feature_bc_matrix.h5"))
                if len(hit) != 1:
                    raise SystemExit(f"CD4 T {d} {cond} lane {ln}: {len(hit)} GEO matrices on disk")
                path, receipts[hit[0].name] = raw_file("cd4t_marson2025", hit[0].name)
                m = lane == ln
                self.files.append((d, path))
                self.meta.append((barcode[m], lab[m]))
        # Sequencing runs list the same genes in different orders (by id or by position): every lane is mapped
        # onto the first lane's gene order by Ensembl id.
        self.cols = []
        for _, path in self.files:
            with h5py.File(path) as h:
                f = h["matrix/features"]
                ids, names = _dec(f["id"][:]), _dec(f["name"][:])
                gex = _dec(f["feature_type"][:]) == "Gene Expression"
            if not self.cols:
                self.ensembl, self.genes = ids[gex], names[gex]
            pos = pd.Index(ids).get_indexer(self.ensembl)
            if set(ids[gex]) != set(self.ensembl) or (pos < 0).any():
                raise SystemExit(f"{path}: its genes differ from the other lanes")
            self.cols.append(pos)
        labs = np.concatenate([lb for _, lb in self.meta])
        self.targets = sorted(set(labs) - {"", self.CTRL})
        self.provenance = dict(source="GEO GSE314342 per-lane matrices + S3 marson2025_data obs tables", cond=cond,
                               n_lane_files=len(self.files), raw=receipts,
                               control="guide_type non-targeting, single sgRNA, not low_quality")

    def _iter(self, want_ctrl):
        for (donor, path), (bc, lab), cols in zip(self.files, self.meta, self.cols):
            with h5py.File(path) as h:
                m = h["matrix"]
                n_feat, n_bc = (int(x) for x in m["shape"][:])
                X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=(n_feat, n_bc)).T.tocsr()
                file_bc = _dec(m["barcodes"][:])
            pos = pd.Index(file_bc).get_indexer(bc)
            if (pos < 0).any():
                raise SystemExit(f"{path}: {int((pos < 0).sum())} annotated cells missing from the GEO matrix")
            k = np.flatnonzero(lab == self.CTRL) if want_ctrl else np.flatnonzero((lab != "") & (lab != self.CTRL))
            yield X[pos[k]][:, cols].astype(np.float64), lab[k], np.full(len(k), donor)

    def controls(self):
        for X, _, bat in self._iter(True):
            yield X, bat

    def cells(self):
        yield from self._iter(False)


READERS = {"h1": H1, "scperturb": ScPerturb, "feng": Feng, "dld1": DLD1, "hesc": HESC, "viperturb": VIPerturb,
           "orion": Orion, "replogle_rows": ReplogleRows, "kolf": Kolf, "cd4t": CD4T}
