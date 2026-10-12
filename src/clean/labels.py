"""Target label to panel symbol, with one rule for every reader."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HGNC = ROOT / "data/raw/hgnc/hgnc_complete_set_2026-10-06.txt"
SENTINELS = {"", "*", "nan", "none", "na", "null"}
SEP = re.compile(r"[_+;,]")


def hgnc_suggestions(vocab):
    """HGNC symbol -> (the one panel gene of its HGNC record, whether the symbol is approved, previous or alias)."""
    if not HGNC.exists():
        return {}
    t = pd.read_csv(HGNC, sep="\t", usecols=["symbol", "prev_symbol", "alias_symbol"], dtype=str).fillna("")
    t = t[["symbol", "prev_symbol", "alias_symbol"]]
    vi, approved = set(vocab), set(t["symbol"])
    tiers = ({}, {}, {})
    for s, prev, alias in t.itertuples(index=False):
        prev, alias = [x for x in prev.split("|") if x], [x for x in alias.split("|") if x]
        hit = {n for n in [s] + prev + alias if n in vi and (n == s or n not in approved)}
        if not hit:
            continue
        for tier, group in zip(tiers, ([s], prev, [a for a in alias if a not in prev])):
            for n in group:
                if n == s or n not in approved:
                    tier.setdefault(n, set()).update(hit)
    out = {}
    for kind, tier in zip(("approved", "previous", "alias"), tiers):
        for n, hit in tier.items():
            if n not in out and n not in vi:
                out[n] = (next(iter(hit)), kind) if len(hit) == 1 else None
    return {n: s for n, s in out.items() if s is not None}


class TargetLabels:
    """Callable raw label -> (panel symbol, or "" to drop the cells; action). Remembers every label it saw."""

    def __init__(self, vocab, gm=None, native_symbols=None):
        self.vocab = set(vocab)
        up = {}
        for g in vocab:
            up.setdefault(g.upper(), []).append(g)
        self.up = {u: g[0] for u, g in up.items() if len(g) == 1}
        self.native = {}
        if gm is not None:
            hits = {}
            for s, v in zip(np.asarray(native_symbols).astype(str), gm.vocab_of):
                if v >= 0:
                    hits.setdefault(s, set()).add(vocab[v])
            self.native = {s: next(iter(h)) for s, h in hits.items() if len(h) == 1}
        self.hgnc = hgnc_suggestions(vocab)
        self.seen = {}

    def _panel(self, s):
        if s in self.vocab:
            return s, "exact"
        if s.upper() in self.up:
            return self.up[s.upper()], "case"
        return None, None

    def _map(self, raw):
        s = str(raw).strip()
        if s.lower() in SENTINELS:
            return "", "sentinel"
        name, how = self._panel(s)
        if name:
            return name, how
        if "_" in s:
            name, _ = self._panel(s.replace("_", "-"))
            if name:
                return name, "underscore"
        if s in self.native:
            return self.native[s], "deposit_gene_axis"
        parts = [p for p in SEP.split(s) if p]
        if len(parts) > 1 and sum(self._panel(p)[0] is not None for p in parts) > 1:
            return "", "multi_gene"
        if s in self.hgnc:
            name, kind = self.hgnc[s]
            return name, f"hgnc_{kind}"
        return s, "not_in_panel"

    def __call__(self, raw):
        if raw not in self.seen:
            self.seen[raw] = self._map(raw)
        return self.seen[raw]

    def report(self, cells):
        """Count of labels per action, and every changed label with the raw labels that became the same gene."""
        actions = pd.Series([a for _, a in self.seen.values()]).value_counts().to_dict()
        raws = {}
        for r, (n, _) in self.seen.items():
            if n:
                raws.setdefault(n, []).append(r)
        changed = [dict(raw=r, action=a, name=n, cells=int(cells.get(r, 0)),
                        merged_with=sorted(x for x in raws.get(n, []) if x != r))
                   for r, (n, a) in sorted(self.seen.items()) if a != "exact"]
        return dict(actions={k: int(v) for k, v in actions.items()}, changed=changed)
