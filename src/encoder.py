"""Original control/depth fits and count encoder; no cell reweighting."""
import numpy as np

def fit_control(counts):
    """(composition, NB dispersion, library sizes) from a control count matrix [n, G].

    Method-of-moments on Var = mu + alpha*mu^2 with mu[i,g] = L_i * p_g, expanded into O(G)
    accumulators and swept in row blocks: a 30k x 18,080 float64 view of the whole block is
    4.3 GB and every `counts ** 2` style expression would hold another copy of it.

    ⚠️ `p` is the MEAN OF PER-CELL FRACTIONS, not the pooled `sum_i c_ig / sum_i L_i`. The
    pooled form weights each cell by its library size; cell-eval2 scores lfc off the ARITHMETIC
    mean of per-cell CPM (mean_calc="arithmetic"), which is the unweighted one. The two differ
    exactly for genes whose expression fraction correlates with depth -- mito/ribo, always -- and
    that difference lands as a per-gene lfc bias we would have no way to see.

    ponytail: alpha is still fitted against mu_ig = L_i * p_g with this unweighted p. That is a
    slightly wrong mean model for the depth-correlated genes; the resulting dispersion error is
    second order next to the first-order lfc bias fixed above. Revisit only if the encoder's
    realised p-values come out miscalibrated on those genes specifically.
    """
    n, G = counts.shape
    BLOCK = max(1, 200_000_000 // (8 * G))   # ~1.6 GB of float64 per block
    L = np.empty(n)
    s_frac = np.zeros(G)   # sum_i c_ig / L_i        (unweighted per-cell fractions)
    s_c2 = np.zeros(G)     # sum_i c_ig^2
    s_Lc = np.zeros(G)     # sum_i L_i c_ig
    n_keep = 0
    for lo in range(0, n, BLOCK):
        blk = counts[lo:lo + BLOCK]
        b = np.asarray(blk.toarray() if hasattr(blk, "toarray") else blk, dtype=np.float64)
        Lb = b.sum(1)
        L[lo:lo + b.shape[0]] = Lb
        keep = Lb > 0
        n_keep += int(keep.sum())
        s_frac += (b[keep] / Lb[keep, None]).sum(0)
        s_c2 += np.einsum("ij,ij->j", b, b)
        s_Lc += Lb @ b
    tot = L.sum()
    if tot <= 0:
        raise ValueError("control matrix has no counts")
    if n_keep < n:
        print(f"  WARNING fit_control: {n - n_keep}/{n} control cells have zero counts -> "
              f"dropped from the composition (they carry no fraction to average)")
    p = s_frac / s_frac.sum()

    # sum_i (c_ig - L_i p_g)^2  =  sum c^2 - 2 p_g sum(L_i c_ig) + p_g^2 sum L_i^2
    ss = s_c2 - 2 * p * s_Lc + p ** 2 * (L ** 2).sum()
    sum_mu = p * tot                 # sum_i mu_ig
    sum_mu2 = p ** 2 * (L ** 2).sum()  # sum_i mu_ig^2
    alpha = np.divide(ss - sum_mu, sum_mu2, out=np.zeros_like(p), where=sum_mu2 > 0)

    n_neg = int((alpha < 0).sum())
    if n_neg:
        print(f"  WARNING fit_control: {n_neg}/{len(p)} genes are UNDER-dispersed vs Poisson "
              f"(alpha < 0) -> clamped to 0")
    return p, np.maximum(alpha, 0.0), L

def fit_depth(counts, p, L, n_q=200, bmax=3.0):
    """Depth covariance of the controls -> (beta [G], p_pool [G], alpha [G] refitted), for `encode(beta=...)`.

    WHY (logs/brainstorm/cellgen_S.md, 2026-09-22): real cells' composition depends on depth -- a gene's per-cell
    fraction scales like r^beta_g, r = L / mean L. The pooled composition sum c / sum L (what the pseudobulk and
    thus pds / expr_mse read) therefore differs from the unweighted mean of per-cell fractions p (what the lfc reads)
    by a gene-specific ratio. `encode` without beta gives every cell the SAME composition, so its pseudobulk equals
    p * 2^lfc and every emitted row carries delta0 = log1p(TS p) - log1p(TS p_pool) vs the controls (norm 1.7-2.5,
    21-30 % of a predicted delta at gain 1.5), which the truth does not carry.
    beta_g is solved so that, over the controls' own depth distribution, pooled / unweighted = p_pool / p exactly
    (mean(r^(b+1)) / (mean(r) mean(r^b)) is increasing in b; bisection on n_q depth quantiles). No shrinkage: the
    ratio being matched IS the controls' observed one, which is what the scored delta is taken against.
    alpha is refitted by moments against mu_ig = L_i p_g r_i^beta_g / mean_i(r_i^beta_g); the constant-composition
    alpha would count the depth-driven spread twice."""
    r = L / L.mean()
    tot = np.asarray(counts.sum(0), dtype=np.float64).ravel()
    p_pool = tot / tot.sum()
    rho = np.divide(p_pool, p, out=np.ones_like(p), where=p > 0)
    rq = np.quantile(r, (np.arange(n_q) + 0.5) / n_q)
    lr = np.log(rq)
    lo, hi = np.full(len(p), -bmax), np.full(len(p), bmax)
    for _ in range(40):
        b = (lo + hi) / 2
        e = np.exp(np.outer(b, lr))
        f = (e * rq).mean(1) / (rq.mean() * e.mean(1))
        lo, hi = np.where(f < rho, b, lo), np.where(f >= rho, b, hi)
    beta = np.where(p > 0, (lo + hi) / 2, 0.0)
    n_clip = int((np.abs(beta) > bmax - 1e-3).sum())
    if n_clip:
        print(f"  WARNING fit_depth: {n_clip} genes hit |beta| = {bmax} (pooled/unweighted ratio outside the reachable range)")
    # alpha refit, same block sweep as fit_control
    n, G = counts.shape
    BLOCK = max(1, 100_000_000 // (8 * G))
    mbeta = np.exp(np.outer(np.log(r), beta)).mean(0) if n * G <= 5e7 else None
    if mbeta is None:                                  # mean_i r_i^beta_g without an n x G matrix
        mbeta = np.zeros(G)
        for lo_ in range(0, n, BLOCK):
            mbeta += np.exp(np.outer(np.log(r[lo_:lo_ + BLOCK]), beta)).sum(0)
        mbeta /= n
    ss = np.zeros(G); smu = np.zeros(G); smu2 = np.zeros(G)
    for lo_ in range(0, n, BLOCK):
        blk = counts[lo_:lo_ + BLOCK]
        c = np.asarray(blk.toarray() if hasattr(blk, "toarray") else blk, dtype=np.float64)
        mu = L[lo_:lo_ + BLOCK, None] * p[None, :] * np.exp(np.outer(np.log(r[lo_:lo_ + BLOCK]), beta)) / mbeta[None, :]
        ss += ((c - mu) ** 2).sum(0); smu += mu.sum(0); smu2 += (mu ** 2).sum(0)
    alpha = np.divide(ss - smu, smu2, out=np.zeros(G), where=smu2 > 0)
    print(f"  fit_depth: |beta| median {np.median(np.abs(beta[p > 0])):.3f}, q99 {np.quantile(np.abs(beta[p > 0]), 0.99):.3f}; "
          f"alpha refit: {int((alpha < 0).sum())} genes < 0 -> 0")
    return beta, p_pool, np.maximum(alpha, 0.0)

def _solve_tilt(r, ratio, bmax=4.0, n_iter=40):
    """Per-gene depth elasticity b with mean(r^(b+1)) / (mean(r) mean(r^b)) = ratio on the GIVEN cells' relative depths r
    (the ratio is increasing in b). Bisection, vectorised over genes. Returns (b [G], n_unreachable)."""
    lr = np.log(r)
    lo, hi = np.full(ratio.shape, -bmax), np.full(ratio.shape, bmax)
    for _ in range(n_iter):
        b = (lo + hi) / 2
        e = np.exp(np.outer(b, lr))                                  # [G, n]
        f = (e * r).mean(1) / (r.mean() * e.mean(1))
        lo, hi = np.where(f < ratio, b, lo), np.where(f >= ratio, b, hi)
    b = (lo + hi) / 2
    return b, int((np.abs(b) > bmax - 1e-3).sum())

def encode_decoupled(p_ctrl, p_pool, alpha, lib_sizes, lfc_m, lfc_b, n_cells=400, rng=None, gate_cpm=5.0, bmax=4.0, tilt_solver=None, L=None):
    """⚠️ EXPERIMENT (user decision 2026-09-28; brainstorm A classifies it as metric gaming): cells whose MEAN per-cell composition
    realises `lfc_m` (what cell-eval2's lfc and hence nmae / fid / reach signs read) while their POOLED pseudobulk realises `lfc_b`
    (what pds / expr_mse read), both against the control. Mechanism: a per-(target, gene) depth tilt b_g solved on the drawn library
    sizes so that pooled / unweighted = q_B / q_M, then the per-cell renormalised composition is calibrated to the q_M mean, counts
    are NB with the control dispersion, and exact_bulk pins each gene's total to its expectation. lfc_b = lfc_m reduces to the
    depth-covariant encoder (b = the controls' own elasticity). Genes whose ratio is out of reach at |b| = bmax are clipped (counted)."""
    rng = np.random.default_rng() if rng is None else rng
    qm = p_ctrl * np.exp2(lfc_m); qm /= qm.sum()
    qb = p_pool * np.exp2(lfc_b); qb /= qb.sum()
    # L may be pre-drawn by the caller (so a post-process pass can see the exact depths used here).
    L = rng.choice(np.asarray(lib_sizes, dtype=np.float64), size=n_cells) if L is None else np.asarray(L, dtype=np.float64)
    r = L / L.mean()
    ratio = np.divide(qb, qm, out=np.ones_like(qm), where=qm > 0)
    # Explicit opt-in leaves historical callers and running builds unchanged.
    solver = _solve_tilt if tilt_solver is None else tilt_solver
    b, n_clip = solver(r, ratio, bmax)
    E = np.exp(np.outer(np.log(r), b))                               # [n, G]
    a_ = qm.copy()
    for _ in range(10):
        Q = a_ * E
        Q /= Q.sum(1, keepdims=True)
        m_ = Q.mean(0)
        a_ = a_ * np.divide(qm, m_, out=np.ones_like(qm), where=m_ > 0)
    mu = L[:, None] * Q
    alpha = np.asarray(alpha, dtype=np.float64)
    pois = alpha <= 0
    shape = np.where(pois, 1.0, 1.0 / np.where(pois, 1.0, alpha))
    lam = np.where(pois[None, :], mu, rng.gamma(shape=shape, scale=mu * alpha))
    counts = rng.poisson(lam).astype(np.int32)
    d = np.rint(mu.sum(0)).astype(np.int64) - counts.sum(0, dtype=np.int64)
    for g in np.flatnonzero(d > 0):
        counts[:, g] += rng.multinomial(d[g], mu[:, g] / mu[:, g].sum()).astype(np.int32)
    for g in np.flatnonzero(d < 0):
        counts[:, g] -= rng.multivariate_hypergeometric(counts[:, g], -d[g]).astype(np.int32)
    gated = p_ctrl * 1e6 > gate_cpm
    dead = gated & (counts.sum(0) == 0)
    if dead.any():
        counts[rng.integers(n_cells, size=int(dead.sum())), np.flatnonzero(dead)] = 1
    if n_clip:
        print(f"  encode_decoupled: {n_clip} genes hit |tilt| = {bmax} (their pooled/mean split is only partly delivered)")
    return counts
