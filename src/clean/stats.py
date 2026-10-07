"""Streaming per-group statistics, defined exactly as cell-eval2 (vcc2026 preset) defines them.

Groups = the control pool + each perturbation. Every quantity below is either an additive per-group
sum (so chunks of cells can be added and thrown away) or, for the Mann-Whitney test, an additive sum
against a per-gene SORTED control reference (so the controls must be loaded first):

  L_i       = sum of counts over the vocab genes the dataset measures (cell-eval2 normalizes over
              the AnnData's own gene axis, which for VCC2026 is the 18,533-gene vocab)
  CPM_ij    = c_ij / L_i * 1e6                          (cells with L_i < 1000 are dropped)
  mean_cpm  = arithmetic mean_i CPM_ij                  (cell_eval2 prep._grouped_means)
  lfc       = log2((mean_pert + 1e-9) / (mean_ctrl + 1e-9))      (de_compute.py:294), ungated
  nlog10p   = -log10 two-sided MWU p, pdex/numba_mwu asymptotic formula with tie correction and
              continuity (numba_mwu/_core.py:_mwu_stats_from_rank_sum), vs the FULL control pool
  disp      = NB method of moments, Var = mu + a mu^2, mu_ij = L_i p_j, p_j = mean_i c_ij/L_i
              (same model as src/encoder.py:fit_control), stored UNCLAMPED
  reliability = 8 random half-splits; delta_h = mean log1p(CP10k)[pert half h] - mean[ctrl half h];
              Pearson(delta_A, delta_B) over measured genes minus the target; mean r -> Spearman-Brown
              (the src/percell_splithalf.py recipe)

MWU streaming identity: U1 = sum_{pert cells i} (#ctrl < x_i + 0.5 #ctrl == x_i) exactly, so U1 is a
sum over perturbed cells. The tie term sum(t^3 - t) is exact for the zero tie and for control/pert
ties; it assumes no two PERTURBED cells share one identical non-zero CPM (they would need equal c/L),
an O(1e-6) effect on sigma. test_stats.py: equals exact float64 MWU (scipy) to <1e-5 rel; pdex itself
differs by <=3e-4 rel in -log10 p because it ranks float32 log1p(CPM) (rounding merges/splits ties).
"""
import math

import h5py
import numba as nb
import numpy as np
import scipy.sparse as sp

EPS = 1e-9
MIN_UMI = 1000
N_SPLITS = 8
MIN_HALF = 8          # src/percell_splithalf.py MIN_CELLS_HALF
LN10 = math.log(10.0)


@nb.njit(inline="always")
def _coin(seed, split, grp, pair):
    """splitmix64 hash -> one fair bit per (seed, split, group, pair); stateless so that the same cell
    sequence always gets the same halves, whichever way it is chunked."""
    z = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(split) * np.uint64(0xBF58476D1CE4E5B9) \
        + np.uint64(grp) * np.uint64(0x94D049BB133111EB) + np.uint64(pair)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return np.int64(z & np.uint64(1))


@nb.njit
def _half_codes(grp, rank, seed, n_splits):
    """Pairing: consecutive cells (by rank within group) form a pair; a per-split coin sends the first
    to half 0 or 1 and the second to the other half -> exact halves, odd last cell lands on the coin."""
    out = np.empty((grp.size, n_splits), np.uint8)
    for i in range(grp.size):
        for s in range(n_splits):
            out[i, s] = _coin(seed, s, grp[i], rank[i] // 2) ^ (rank[i] & 1)
    return out


@nb.njit(parallel=True)
def _add_kernel(indptr, rows, data, L, grp, codes, cols,
                s_frac, s_c2, s_Lc, s_log, s_logA, nnz, s_u, s_tie,
                c_ptr, c_val, c_zero):
    """One CSC chunk; local column j is measured gene cols[j]. Gene-parallel, so no races."""
    n_splits = codes.shape[1]
    for j in nb.prange(indptr.size - 1):
        jj = cols[j]
        ref = c_val[c_ptr[j]:c_ptr[j + 1]]
        cz = c_zero[j]
        for k in range(indptr[j], indptr[j + 1]):
            i = rows[k]
            g = grp[i]
            c = data[k]
            Li = L[i]
            f = c / Li
            s_frac[g, jj] += f
            s_c2[g, jj] += c * c
            s_Lc[g, jj] += Li * c
            lg = math.log1p(f * 1e4)
            s_log[g, jj] += lg
            for s in range(n_splits):
                if codes[i, s] == 0:
                    s_logA[g, s, jj] += lg
            nnz[g, jj] += 1
            x = c / Li * 1e6
            lo = np.searchsorted(ref, x, side="left")
            hi = np.searchsorted(ref, x, side="right")
            eq = hi - lo
            s_u[g, jj] += cz + lo + 0.5 * eq
            if eq > 0:
                s_tie[g, jj] += 3.0 * eq * eq + 3.0 * eq


@nb.njit
def _sorted_ref(indptr, vals):
    """Sort each control column in place; return the non-zero tie term sum(t^3 - t) per gene."""
    G = indptr.size - 1
    tie = np.zeros(G)
    for j in range(G):
        v = vals[indptr[j]:indptr[j + 1]]
        v.sort()
        t = 1
        for k in range(1, v.size + 1):
            if k < v.size and v[k] == v[k - 1]:
                t += 1
            else:
                tie[j] += t * t * t - t
                t = 1
    return tie


@nb.njit
def _control_pass(indptr, indices, data, L, half, b):
    """CSR control pool -> per-gene sums, per-split half sums of log1p(CP10k), and the CSC-ordered CPM
    values (counting sort; CPM = c / L * 1e6 exactly as in _add_kernel, so ties compare bit-identically)."""
    S = half.shape[0]
    s_frac, s_c2, s_Lc = np.zeros(b), np.zeros(b), np.zeros(b)
    hsum = np.zeros((S, 2, b))
    cnt = np.zeros(b + 1, np.int64)
    for k in range(indices.size):
        cnt[indices[k] + 1] += 1
    ptr = np.cumsum(cnt)
    fill = ptr[:-1].copy()
    val = np.empty(indices.size)
    for i in range(indptr.size - 1):
        Li = L[i]
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            c = np.float64(data[k])
            f = c / Li
            s_frac[j] += f
            s_c2[j] += c * c
            s_Lc[j] += Li * c
            lg = math.log1p(f * 1e4)
            for s in range(S):
                h = half[s, i]
                if h >= 0:
                    hsum[s, h, j] += lg
            val[fill[j]] = c / Li * 1e6
            fill[j] += 1
    return s_frac, s_c2, s_Lc, hsum, ptr, val


def nb_mom(s_frac, s_c2, s_Lc, n, sL, sL2):
    """NB method-of-moments alpha from sufficient statistics (broadcast over leading dims)."""
    p = s_frac / n
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (s_c2 - 2 * p * s_Lc + p ** 2 * sL2 - p * sL) / (p ** 2 * sL2)
    return np.where((p > 0) & (n >= 2), a, np.nan)


def mwu_nlog10p(U1, n1, n2, tie):
    """pdex two-sided asymptotic MWU (continuity + tie correction) -> -log10 p, computed through
    log_ndtr so that p below float64's range stays finite (pdex itself would report p = 0 there)."""
    from scipy.special import log_ndtr
    n = n1 + n2
    U = np.maximum(U1, n1 * n2 - U1)
    s2 = n1 * n2 / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (U - n1 * n2 / 2.0 - 0.5) / np.sqrt(s2)
    out = np.maximum(0.0, -(log_ndtr(-z) + math.log(2.0)) / LN10)   # p = min(1, 2 Phi(-z))
    return np.where(s2 > 0, out, 0.0)


def _pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


class Engine:
    """Accumulator over G measured genes for T perturbations (+ the control pool).

    Row mode:   set_controls(ctrl_csr, L_ctrl) once, then add(csr, L, grp) per chunk.
    Block mode: (gene-major sources: kolf CSC, Feng gene rows) per gene block `cols` (measured-gene
                indices, any order, each gene complete within one block): set_controls(.., cols) then
                add(.., cols, count_cells=(first block)); call reset_ranks() before each block so every
                block sees the same cell order and hence the same half assignment.
    """

    def __init__(self, n_genes, n_targets, seed=0, n_splits=N_SPLITS):
        G, T, S = n_genes, n_targets, n_splits
        self.G, self.T, self.S, self.seed = G, T, S, seed
        self.s_frac, self.s_c2, self.s_Lc, self.s_u = (np.zeros((T, G)) for _ in range(4))
        self.s_log = np.zeros((T, G), np.float32)
        self.s_tie = np.zeros((T, G), np.float32)
        self.s_logA = np.zeros((T, S, G), np.float32)
        self.nnz = np.zeros((T, G), np.int32)
        self.n = np.zeros(T, np.int64)
        self.sL = np.zeros(T)
        self.sL2 = np.zeros(T)
        self.L_parts, self.g_parts = [], []          # per-cell depth, for medians
        self.rank = np.zeros(T, np.int64)
        self.n_low = 0                               # cells dropped by the UMI filter
        # control, full gene axis
        self.c_mean, self.c_disp = np.full(G, np.nan), np.full(G, np.nan)
        self.c_halfA, self.c_halfB = np.zeros((S, G)), np.zeros((S, G))
        self.c_tie = np.zeros(G)
        self.c_zero_full = np.zeros(G)
        self.n_c = None

    def set_controls(self, X, L, cols=None):
        """X: [n_c, b] counts (cells already UMI-filtered), L: [n_c] vocab depth. One numba pass over the
        CSR arrays (no scipy copies: control pools reach 1e9 non-zeros) -> control moments, half means
        and the per-gene sorted CPM reference for the MWU."""
        X = sp.csr_matrix(X)
        n_c, b = X.shape
        if self.n_c is None:
            self.n_c, self.c_L = n_c, np.asarray(L, np.float64)
            perms = [np.random.default_rng([self.seed, 1000 + s]).permutation(n_c) for s in range(self.S)]
            h = n_c // 2
            self.c_idx = [(p[:h], p[h:2 * h]) for p in perms]
            self.c_half = np.full((self.S, n_c), -1, np.int8)      # 0 = half A, 1 = half B, -1 = odd cell out
            for s, (a, bb) in enumerate(self.c_idx):
                self.c_half[s, a], self.c_half[s, bb] = 0, 1
        assert n_c == self.n_c, "control pool changed between gene blocks"
        L = self.c_L
        sl = np.arange(b, dtype=np.int64) if cols is None else np.asarray(cols, np.int64)
        assert len(sl) == b
        s_frac, s_c2, s_Lc, hsum, ptr, val = _control_pass(X.indptr.astype(np.int64), X.indices.astype(np.int64),
                                                           X.data, L, self.c_half, b)
        p = s_frac / n_c
        self.c_mean[sl] = p * 1e6
        self.c_disp[sl] = nb_mom(s_frac, s_c2, s_Lc, n_c, L.sum(), (L ** 2).sum())
        h = n_c // 2
        self.c_halfA[:, sl] = hsum[:, 0] / h
        self.c_halfB[:, sl] = hsum[:, 1] / h
        self.ref_ptr, self.ref_val = ptr, val
        self.c_tie[sl] = _sorted_ref(self.ref_ptr, self.ref_val)
        self.ref_zero = (n_c - np.diff(self.ref_ptr)).astype(np.float64)
        self.c_zero_full[sl] = self.ref_zero
        self.ref_cols = sl

    def reset_ranks(self):
        self.rank[:] = 0

    def add(self, X, L, grp, cols=None, count_cells=True):
        """X: [n, b] counts of PERTURBED cells (UMI-filtered), grp: [n] target index 0..T-1."""
        if cols is not None:
            assert np.array_equal(cols, self.ref_cols), "add() must use the gene block of the last set_controls()"
        grp = np.asarray(grp, np.int64)
        L = np.asarray(L, np.float64)
        # rank of each cell within its group, in arrival order
        order = np.argsort(grp, kind="stable")
        gs = grp[order]
        starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        within = np.arange(gs.size) - np.repeat(starts, np.diff(np.r_[starts, gs.size]))
        rank = np.empty_like(grp)
        rank[order] = self.rank[gs] + within
        np.add.at(self.rank, grp, 1)
        codes = _half_codes(grp, rank, self.seed, self.S)
        Xc = sp.csc_matrix(X, dtype=np.float64)
        Xc.sort_indices()
        _add_kernel(Xc.indptr.astype(np.int64), Xc.indices.astype(np.int64), Xc.data, L, grp, codes, self.ref_cols,
                    self.s_frac, self.s_c2, self.s_Lc, self.s_log, self.s_logA, self.nnz, self.s_u,
                    self.s_tie, self.ref_ptr, self.ref_val, self.ref_zero)
        if count_cells:
            self.n += np.bincount(grp, minlength=self.T)
            self.sL += np.bincount(grp, L, minlength=self.T)
            self.sL2 += np.bincount(grp, L ** 2, minlength=self.T)
            self.L_parts.append(L.astype(np.float32))
            self.g_parts.append(grp.astype(np.int32))

    def result(self, target_gene_idx):
        """-> dict of [T, G] float32 arrays + per-target and control vectors.
        target_gene_idx: [T] measured-gene index of each target's own gene (-1 if not measured)."""
        T, G, S = self.T, self.G, self.S
        n2 = float(self.n_c)
        out = {k: np.empty((T, G), np.float32) for k in ("mean_cpm", "lfc", "disp", "nlog10p")}
        for lo in range(0, T, 1024):                          # target blocks: float64 temporaries stay ~1 GB
            b = slice(lo, min(lo + 1024, T))
            n = self.n[b].astype(np.float64)[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = self.s_frac[b] / n * 1e6
            out["mean_cpm"][b] = mean
            out["lfc"][b] = np.log2((mean + EPS) / (self.c_mean + EPS))
            out["disp"][b] = nb_mom(self.s_frac[b], self.s_c2[b], self.s_Lc[b], n, self.sL[b, None], self.sL2[b, None])
            zp = n - self.nnz[b]                              # zero-count pert cells per gene
            U1 = self.s_u[b] + 0.5 * self.c_zero_full * zp
            t0 = self.c_zero_full + zp
            tie = self.c_tie + (t0 ** 3 - t0) + self.s_tie[b]
            out["nlog10p"][b] = mwu_nlog10p(U1, n, n2, tie)
        empty = self.n == 0
        for k in ("mean_cpm", "lfc", "disp", "nlog10p"):
            out[k][empty] = np.nan
        del self.s_frac, self.s_c2, self.s_Lc, self.s_u, self.s_tie, self.nnz   # reliability needs only s_log/s_logA
        # reliability
        r = np.full((T, S), np.nan)
        keep = np.ones(G, bool)
        pairs = self.n // 2
        for s in range(S):
            odd = (self.n % 2 == 1)
            last_coin = np.array([_coin(self.seed, s, t, (self.n[t] - 1) // 2) for t in range(T)]) if T else np.zeros(0)
            nA = pairs + (odd & (last_coin == 0))
            nB = self.n - nA
            ok = (nA >= MIN_HALF) & (nB >= MIN_HALF) & (self.n_c // 2 >= MIN_HALF)
            for t in np.flatnonzero(ok):
                A = self.s_logA[t, s].astype(np.float64)
                B = self.s_log[t].astype(np.float64) - A
                dA = A / nA[t] - self.c_halfA[s]
                dB = B / nB[t] - self.c_halfB[s]
                keep[:] = True
                if target_gene_idx[t] >= 0:
                    keep[target_gene_idx[t]] = False
                r[t, s] = _pearson_rows(dA[keep][None], dB[keep][None])[0]
        out["reliability_r"] = r.astype(np.float32)
        cnt = np.isfinite(r).sum(1)
        rm = np.where(cnt > 0, np.nansum(r, 1) / np.maximum(cnt, 1), np.nan)   # NaN = too few cells
        out["reliability_mean"] = rm.astype(np.float32)
        out["reliability_sb"] = np.where(np.isfinite(rm), np.maximum(2 * rm / (1 + rm), 0), np.nan).astype(np.float32)
        out["n_cells"] = self.n.astype(np.int64)
        out["depth_mean"] = np.where(self.n > 0, self.sL / np.maximum(self.n, 1), np.nan)
        Ls = np.concatenate(self.L_parts) if self.L_parts else np.zeros(0)
        gs = np.concatenate(self.g_parts) if self.g_parts else np.zeros(0, np.int32)
        o = np.lexsort((Ls, gs))
        Ls, gs = Ls[o], gs[o]
        med = np.full(T, np.nan)
        b = np.searchsorted(gs, np.arange(T + 1))
        for t in np.flatnonzero(self.n > 0):
            med[t] = np.median(Ls[b[t]:b[t + 1]])
        out["depth_median"] = med
        out["control"] = dict(mean_cpm=self.c_mean.astype(np.float32), disp=self.c_disp.astype(np.float32),
                              n_cells=int(self.n_c), depth_mean=float(self.c_L.mean()),
                              depth_median=float(np.median(self.c_L)))
        return out


def write_h5(path, res, vocab, measured_idx, targets, attrs):
    """Expand the measured-gene axis to the full vocab (NaN where not measured) and write."""
    V = len(vocab)
    measured = np.zeros(V, bool)
    measured[measured_idx] = True

    def expand(a):
        full = np.full(a.shape[:-1] + (V,), np.nan, np.float32)
        full[..., measured_idx] = a
        return full

    s = h5py.string_dtype()
    with h5py.File(path, "w") as f:
        f.attrs.update(attrs)
        f.create_dataset("genes", data=np.asarray(vocab, object), dtype=s)
        f.create_dataset("measured", data=measured)
        c = res["control"]
        f.create_dataset("control/mean_cpm", data=expand(c["mean_cpm"]))
        f.create_dataset("control/disp", data=expand(c["disp"]))
        for k in ("n_cells", "depth_mean", "depth_median"):
            f.create_dataset(f"control/{k}", data=c[k])
        f.create_dataset("pert/targets", data=np.asarray(targets, object), dtype=s)
        for k in ("n_cells", "depth_mean", "depth_median", "reliability_mean", "reliability_sb", "reliability_r"):
            f.create_dataset(f"pert/{k}", data=res[k])
        T = len(targets)
        for k in ("mean_cpm", "disp", "lfc", "nlog10p"):
            ds = f.create_dataset(f"pert/{k}", shape=(T, V), dtype=np.float32, chunks=(min(64, max(T, 1)), V),
                                  compression="gzip", compression_opts=4, shuffle=True, fillvalue=np.nan)
            for lo in range(0, T, 1024):
                ds[lo:lo + 1024] = expand(res[k][lo:lo + 1024])
