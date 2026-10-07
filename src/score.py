"""Six-member internal H1 scaling using published VCC2026 midpoint anchors.
This is an internal proxy aggregate, not an official leaderboard score.
"""
import polars as pl
SCORED = {
    "pds_cosine": None,
    "expr_mse_unbiased_capped_norm": (0.0, 1.0),
    "de_wilcoxon_direction_fidelity_yield_raw": None,
    "de_wilcoxon_direction_reach_raw": None,
    "de_wilcoxon_sig_jaccard": None,
    "de_wilcoxon_lfc_nmae": (-6.0, None),
}
ANCHORS_2026 = {
    "b": {"pds_cosine": 0.500, "expr_mse_unbiased_capped_norm": 0.989,
          "de_wilcoxon_direction_fidelity_yield_raw": 0.5135,
          "de_wilcoxon_direction_reach_raw": 0.072,
          "de_wilcoxon_sig_jaccard": 0.029, "de_wilcoxon_lfc_nmae": 1.0013},
    "r": {"pds_cosine": 0.9555, "expr_mse_unbiased_capped_norm": 0.0365,
          "de_wilcoxon_direction_fidelity_yield_raw": 0.8135,
          "de_wilcoxon_direction_reach_raw": 0.968,
          "de_wilcoxon_sig_jaccard": 0.399, "de_wilcoxon_lfc_nmae": 0.400},
}

def scale(u, b, r):
    """The six raw aggregates -> the six scaled members. Pure; see _demo() for the sign check.

    The clamps act on `s`, NOT on the raw metric (refs/vcc2026-metrics.md:87-89, which sits inside
    the paragraph defining s). That distinction is load-bearing in one direction: the expression
    error is clamped to [0, 1], so beating the replicate on it earns NOTHING, while lfc_nmae is
    only FLOORED at -6 and pays without limit above 1.
    """
    rows = []
    for m, clamp in SCORED.items():
        missing = [n for n, d in (("u", u), ("b", b), ("r", r)) if m not in d]
        if missing:
            raise KeyError(f"{m} absent from {missing} -- the three runs scored different "
                           "metric sets, so their scales are not comparable")
        span = r[m] - b[m]
        if span == 0:
            raise ValueError(f"{m}: r == b == {b[m]}, the scale is degenerate -- the replicate "
                             "is no better than the baseline, so this metric carries no signal "
                             "on this data and must not be averaged in silently")
        raw = (u[m] - b[m]) / span
        s = raw
        if clamp:
            lo, hi = clamp
            s = max(s, lo) if lo is not None else s
            s = min(s, hi) if hi is not None else s
        rows.append({"metric": m, "u": u[m], "b": b[m], "r": r[m], "span": span,
                     "s_raw": raw, "s": s, "clamped": s != raw})
    return pl.DataFrame(rows)
