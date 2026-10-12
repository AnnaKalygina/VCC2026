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
              (the src/percell_splithalf.py recipe); split 0's two half deltas are kept as vectors
  reliability_*_batchmatched, half_delta_bm_A/B = the same, each half against controls mixed by its cells per batch
  se_cpm    = standard error of mean_cpm, sqrt(var(CPM) / n)
  lfc_batchmatched = lfc against controls mixed by the target's cells per batch; the MWU test stays pooled

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
MIN_CTRL_BATCH = 20
MAX_BATCH_SUMS = 10
GATE_CPM = 5.0
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
                c_ptr, c_val, c_zero, s_frac2, bat, s_fracB):
    """One CSC chunk; local column j is measured gene cols[j]. Gene-parallel, so no races."""
    n_splits = codes.shape[1]
    by_batch = s_fracB.shape[1] > 0
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
            s_frac2[g, jj] += f * f
            if by_batch:
                s_fracB[g, bat[i], jj] += f
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
def _control_pass(indptr, indices, data, L, half, b, bat, n_batches):
    """CSR control pool -> per-gene sums (also per batch), per-split half sums of log1p(CP10k), and the CSC-ordered CPM
    values (counting sort; CPM = c / L * 1e6 exactly as in _add_kernel, so ties compare bit-identically)."""
    S = half.shape[0]
    s_frac, s_frac2, s_c2, s_Lc = np.zeros(b), np.zeros(b), np.zeros(b), np.zeros(b)
    bsum = np.zeros((n_batches, b))
    hsum = np.zeros((S, 2, n_batches, b))
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
            s_frac2[j] += f * f
            bsum[bat[i], j] += f
            s_c2[j] += c * c
            s_Lc[j] += Li * c
            lg = math.log1p(f * 1e4)
            for s in range(S):
                h = half[s, i]
                if h >= 0:
                    hsum[s, h, bat[i], j] += lg
            val[fill[j]] = c / Li * 1e6
            fill[j] += 1
    return s_frac, s_frac2, s_c2, s_Lc, bsum, hsum, ptr, val


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


def batch_codes(labels, batches):
    """Batch label per cell -> code: position in `batches`, or len(batches) for a batch without controls."""
    import pandas as pd
    code = pd.Index(list(batches)).get_indexer(np.asarray(labels).astype(str))
    return np.where(code < 0, len(batches), code).astype(np.int64)


def se_cpm(s_frac, s_frac2, n):
    """Standard error of the arithmetic mean CPM from sum(c/L) and sum((c/L)^2) over n cells."""
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s_frac / n
        var = np.maximum(s_frac2 / n - m * m, 0) * n / (n - 1)
        return np.where(n >= 2, np.sqrt(var / n) * 1e6, np.nan)


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
    Batches:    `batches` names the batches with controls (codes 0..B-1); cells of other batches get code B.
    """

    def __init__(self, n_genes, n_targets, seed=0, n_splits=N_SPLITS, batches=None):
        G, T, S = n_genes, n_targets, n_splits
        self.G, self.T, self.S, self.seed = G, T, S, seed
        self.batches = None if batches is None else list(batches)
        B = 1 if batches is None else len(self.batches)
        self.B = B
        self.n_tb = np.zeros((T, B + 1), np.int64)
        by_batch = batches is not None and B <= MAX_BATCH_SUMS
        self.s_fracB = np.zeros((T, B + 1, G) if by_batch else (1, 0, 1), np.float32)
        self.c_bsum, self.c_nb = np.zeros((B, G)), None
        if batches is not None:
            self.c_hsumB = np.zeros((S, 2, B, G))
            self.n_tbA = np.zeros((T, S, B + 1), np.int32)
        self.c_se = np.full(G, np.nan)
        self.s_frac, self.s_frac2, self.s_c2, self.s_Lc, self.s_u = (np.zeros((T, G)) for _ in range(5))
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

    def set_controls(self, X, L, cols=None, batch=None):
        """X: [n_c, b] counts (cells already UMI-filtered), L: [n_c] vocab depth, batch: [n_c] codes or None. One numba
        pass over the CSR arrays (no scipy copies: control pools reach 1e9 non-zeros) -> control moments, half means
        and the per-gene sorted CPM reference for the MWU."""
        X = sp.csr_matrix(X)
        n_c, b = X.shape
        if self.n_c is None:
            self.n_c, self.c_L = n_c, np.asarray(L, np.float64)
            self.c_bat = np.zeros(n_c, np.int64) if batch is None else np.asarray(batch, np.int64)
            assert len(self.c_bat) == n_c and (self.c_bat >= 0).all() and (self.c_bat < self.B).all()
            self.c_nb = np.bincount(self.c_bat, minlength=self.B)
            perms = [np.random.default_rng([self.seed, 1000 + s]).permutation(n_c) for s in range(self.S)]
            h = n_c // 2
            self.c_idx = [(p[:h], p[h:2 * h]) for p in perms]
            self.c_half = np.full((self.S, n_c), -1, np.int8)      # 0 = half A, 1 = half B, -1 = odd cell out
            for s, (a, bb) in enumerate(self.c_idx):
                self.c_half[s, a], self.c_half[s, bb] = 0, 1
            self.c_hnB = np.stack([[np.bincount(self.c_bat[self.c_half[s] == h], minlength=self.B) for h in (0, 1)]
                                   for s in range(self.S)])
        assert n_c == self.n_c, "control pool changed between gene blocks"
        L = self.c_L
        sl = np.arange(b, dtype=np.int64) if cols is None else np.asarray(cols, np.int64)
        assert len(sl) == b
        s_frac, s_frac2, s_c2, s_Lc, bsum, hsum, ptr, val = _control_pass(
            X.indptr.astype(np.int64), X.indices.astype(np.int64), X.data, L, self.c_half, b, self.c_bat, self.B)
        p = s_frac / n_c
        self.c_mean[sl] = p * 1e6
        self.c_se[sl] = se_cpm(s_frac, s_frac2, n_c)
        self.c_bsum[:, sl] = bsum
        self.c_disp[sl] = nb_mom(s_frac, s_c2, s_Lc, n_c, L.sum(), (L ** 2).sum())
        h = n_c // 2
        self.c_halfA[:, sl] = hsum[:, 0].sum(1) / h
        self.c_halfB[:, sl] = hsum[:, 1].sum(1) / h
        if self.batches is not None:
            self.c_hsumB[..., sl] = hsum
        self.ref_ptr, self.ref_val = ptr, val
        self.c_tie[sl] = _sorted_ref(self.ref_ptr, self.ref_val)
        self.ref_zero = (n_c - np.diff(self.ref_ptr)).astype(np.float64)
        self.c_zero_full[sl] = self.ref_zero
        self.ref_cols = sl

    def reset_ranks(self):
        self.rank[:] = 0

    def add(self, X, L, grp, cols=None, count_cells=True, batch=None):
        """X: [n, b] counts of PERTURBED cells (UMI-filtered), grp: [n] target index, batch: [n] codes or None."""
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
        bat = np.zeros(grp.size, np.int64) if batch is None else np.asarray(batch, np.int64)
        assert len(bat) == grp.size and (bat >= 0).all() and (bat <= self.B).all()
        Xc = sp.csc_matrix(X, dtype=np.float64)
        Xc.sort_indices()
        _add_kernel(Xc.indptr.astype(np.int64), Xc.indices.astype(np.int64), Xc.data, L, grp, codes, self.ref_cols,
                    self.s_frac, self.s_c2, self.s_Lc, self.s_log, self.s_logA, self.nnz, self.s_u,
                    self.s_tie, self.ref_ptr, self.ref_val, self.ref_zero, self.s_frac2, bat, self.s_fracB)
        if count_cells:
            np.add.at(self.n_tb, (grp, bat), 1)
            if self.batches is not None:
                for s in range(self.S):
                    a = codes[:, s] == 0
                    np.add.at(self.n_tbA, (grp[a], s, bat[a]), 1)
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
        keys = ["mean_cpm", "lfc", "disp", "nlog10p", "se_cpm"]
        batched = self.batches is not None
        if batched:
            keys.append("lfc_batchmatched")
            ok_b = self.c_nb >= MIN_CTRL_BATCH
            c_mean_b = np.where(ok_b[:, None], self.c_bsum / np.maximum(self.c_nb, 1)[:, None] * 1e6, 0.0)
            gated = self.c_mean >= GATE_CPM
            shift = np.full(T, np.nan)
        out = {k: np.empty((T, G), np.float32) for k in keys}
        for lo in range(0, T, 1024):                          # target blocks: float64 temporaries stay ~1 GB
            b = slice(lo, min(lo + 1024, T))
            n = self.n[b].astype(np.float64)[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = self.s_frac[b] / n * 1e6
            out["mean_cpm"][b] = mean
            out["lfc"][b] = np.log2((mean + EPS) / (self.c_mean + EPS))
            out["disp"][b] = nb_mom(self.s_frac[b], self.s_c2[b], self.s_Lc[b], n, self.sL[b, None], self.sL2[b, None])
            out["se_cpm"][b] = se_cpm(self.s_frac[b], self.s_frac2[b], n)
            if batched:
                w = self.n_tb[b, :self.B] * ok_b
                with np.errstate(divide="ignore", invalid="ignore"):
                    ref = (w @ c_mean_b) / w.sum(1, keepdims=True)
                lbm = np.log2((mean + EPS) / (ref + EPS))
                out["lfc_batchmatched"][b] = lbm
                d = (out["lfc"][b] - lbm)[:, gated]
                with np.errstate(invalid="ignore"):
                    shift[b] = np.sqrt(np.nanmean(d * d, 1)) if gated.any() else np.nan
            zp = n - self.nnz[b]                              # zero-count pert cells per gene
            U1 = self.s_u[b] + 0.5 * self.c_zero_full * zp
            t0 = self.c_zero_full + zp
            tie = self.c_tie + (t0 ** 3 - t0) + self.s_tie[b]
            out["nlog10p"][b] = mwu_nlog10p(U1, n, n2, tie)
        empty = self.n == 0
        for k in keys:
            out[k][empty] = np.nan
        del self.s_frac, self.s_frac2, self.s_c2, self.s_Lc, self.s_u, self.s_tie, self.nnz
        if batched:
            out["n_cells_by_batch"] = self.n_tb
            with np.errstate(divide="ignore", invalid="ignore"):
                out["frac_no_ctrl"] = 1 - (self.n_tb[:, :self.B] * ok_b).sum(1) / self.n
            out["batch_shift_rms"] = shift.astype(np.float32)
            if self.s_fracB.shape[1]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    out["mean_cpm_by_batch"] = (self.s_fracB / self.n_tb[:, :, None] * 1e6).astype(np.float32)
            del self.s_fracB
        out["half_delta_A"] = np.full((T, G), np.nan, np.float32)
        out["half_delta_B"] = np.full((T, G), np.nan, np.float32)
        if batched:
            out["half_delta_bm_A"] = np.full((T, G), np.nan, np.float32)
            out["half_delta_bm_B"] = np.full((T, G), np.nan, np.float32)
            r_bm = np.full((T, S), np.nan)
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
            if batched:
                okh = self.c_hnB[s] >= MIN_CTRL_BATCH // 2
                with np.errstate(divide="ignore", invalid="ignore"):
                    cmh = np.where(okh[:, :, None], self.c_hsumB[s] / self.c_hnB[s][:, :, None], 0.0)
                wA = self.n_tbA[:, s, :self.B] * okh[0]
                wB = (self.n_tb[:, :self.B] - self.n_tbA[:, s, :self.B]) * okh[1]
            for t in np.flatnonzero(ok):
                A = self.s_logA[t, s].astype(np.float64)
                B = self.s_log[t].astype(np.float64) - A
                dA = A / nA[t] - self.c_halfA[s]
                dB = B / nB[t] - self.c_halfB[s]
                if s == 0:
                    out["half_delta_A"][t], out["half_delta_B"][t] = dA, dB
                keep[:] = True
                if target_gene_idx[t] >= 0:
                    keep[target_gene_idx[t]] = False
                r[t, s] = _pearson_rows(dA[keep][None], dB[keep][None])[0]
                if batched and wA[t].sum() > 0 and wB[t].sum() > 0:
                    eA = A / nA[t] - wA[t] @ cmh[0] / wA[t].sum()
                    eB = B / nB[t] - wB[t] @ cmh[1] / wB[t].sum()
                    if s == 0:
                        out["half_delta_bm_A"][t], out["half_delta_bm_B"][t] = eA, eB
                    r_bm[t, s] = _pearson_rows(eA[keep][None], eB[keep][None])[0]
        out["reliability_r"] = r.astype(np.float32)
        cnt = np.isfinite(r).sum(1)
        rm = np.where(cnt > 0, np.nansum(r, 1) / np.maximum(cnt, 1), np.nan)   # NaN = too few cells
        out["reliability_mean"] = rm.astype(np.float32)
        out["reliability_sb"] = np.where(np.isfinite(rm), np.maximum(2 * rm / (1 + rm), 0), np.nan).astype(np.float32)
        if batched:
            out["reliability_r_batchmatched"] = r_bm.astype(np.float32)
            cnt = np.isfinite(r_bm).sum(1)
            rm = np.where(cnt > 0, np.nansum(r_bm, 1) / np.maximum(cnt, 1), np.nan)
            out["reliability_mean_batchmatched"] = rm.astype(np.float32)
            out["reliability_sb_batchmatched"] = np.where(np.isfinite(rm), np.maximum(2 * rm / (1 + rm), 0),
                                                          np.nan).astype(np.float32)
            del self.c_hsumB, self.n_tbA
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
                              se_cpm=self.c_se.astype(np.float32), n_cells=int(self.n_c),
                              depth_mean=float(self.c_L.mean()), depth_median=float(np.median(self.c_L)))
        if batched:
            out["control"]["n_cells_by_batch"] = self.c_nb
            with np.errstate(divide="ignore", invalid="ignore"):
                out["control"]["mean_cpm_by_batch"] = (self.c_bsum / self.c_nb[:, None] * 1e6).astype(np.float32)
            out["batches"] = self.batches
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
        for k in ("mean_cpm", "disp", "se_cpm", "mean_cpm_by_batch"):
            if k in c:
                f.create_dataset(f"control/{k}", data=expand(c[k]))
        for k in ("n_cells", "depth_mean", "depth_median", "n_cells_by_batch"):
            if k in c:
                f.create_dataset(f"control/{k}", data=c[k])
        if "batches" in res:
            f.create_dataset("batches", data=np.asarray(res["batches"], object), dtype=s)
        f.create_dataset("pert/targets", data=np.asarray(targets, object), dtype=s)
        for k in ("n_cells", "depth_mean", "depth_median", "reliability_mean", "reliability_sb", "reliability_r",
                  "n_cells_by_batch", "frac_no_ctrl", "batch_shift_rms", "reliability_mean_batchmatched",
                  "reliability_sb_batchmatched", "reliability_r_batchmatched"):
            if k in res:
                f.create_dataset(f"pert/{k}", data=res[k])
        T = len(targets)
        for k in ("mean_cpm", "disp", "lfc", "nlog10p", "se_cpm", "lfc_batchmatched", "half_delta_A", "half_delta_B",
                  "half_delta_bm_A", "half_delta_bm_B", "mean_cpm_by_batch"):
            if k not in res:
                continue
            shape = res[k].shape[1:-1] + (V,)
            ds = f.create_dataset(f"pert/{k}", shape=(T,) + shape, dtype=np.float32,
                                  chunks=(max(1, min(64, T) // int(np.prod(shape[:-1]))),) + shape,
                                  compression="gzip", compression_opts=4,
                                  shuffle=True, fillvalue=np.nan)
            for lo in range(0, T, 1024):
                ds[lo:lo + 1024] = expand(res[k][lo:lo + 1024])
