"""Word knowledge that is DATA, not code:
  * synonym groups written in the knowledge base file ("synonyms": [["register","sign up","enrol"], ...])
  * WordNet (a dictionary, optional: pip install nltk, then  py -m nltk.downloader wordnet)
  * contrast sets LEARNED from the knowledge base: if two KB questions are identical except for one word each
    ("swim cutoff" / "bike cutoff"), those two words are alternatives -> a question using one must not match the other.
"""
from __future__ import annotations

from .text import stem
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set


class Lexicon:
    def __init__(self, synonym_groups: Iterable[Iterable[str]] = (), wordnet: str = "auto", path_threshold: float = 2.0) -> None:
        self.path_threshold = path_threshold
        self.max_senses = 1
        self._syn: Dict[str, int] = {}
        for i, group in enumerate(synonym_groups):
            for w in group:
                self._syn[w.lower().replace(" ", "_")] = i
                self._syn[self._norm(w)] = i
        self._wn = None
        self.wordnet_status = "off"
        if wordnet in ("auto", "on"):
            try:
                from nltk.corpus import wordnet as wn
                wn.synsets("test")            # raises LookupError when the data is not downloaded
                self._wn, self.wordnet_status = wn, "on"
            except Exception as exc:          # ImportError / LookupError
                if wordnet == "on":
                    raise RuntimeError("WordNet requested but not available: pip install nltk  &&  py -m nltk.downloader wordnet") from exc
        self._contrast: Dict[str, int] = {}
        self._cache: Dict[tuple, bool] = {}

    @staticmethod
    def _norm(w: str) -> str:
        return stem(w.lower().replace(" ", "").replace("_", "").replace("-", ""))

    def _gid(self, w: str):
        g = self._syn.get(w)
        return g if g is not None else self._syn.get(self._norm(w))

    # ---- synonyms
    def related(self, a: str, b: str) -> bool:
        if a == b:
            return True
        ga, gb = self._gid(a), self._gid(b)
        if ga is not None and ga == gb:
            return True
        if self._wn is None:
            return False
        key = (a, b) if a < b else (b, a)
        if key not in self._cache:
            self._cache[key] = self._wn_related(a, b)
        return self._cache[key]

    def _forms(self, w: str) -> Set[str]:
        """The word, its WordNet synonyms and derivationally related forms (register -> registration, enter -> entry)."""
        out = {w}
        for s in self._wn.synsets(w.replace("_", " "))[:self.max_senses]:      # only the most frequent senses (polysemy guard)
            for l in s.lemmas():
                out.add(l.name().lower())
                out.update(d.name().lower() for d in l.derivationally_related_forms())
        return out

    def _wn_related(self, a: str, b: str) -> bool:
        fa, fb = self._forms(a), self._forms(b)
        if fa & fb:
            return True
        best = 0.0
        for x in self._wn.synsets(a.replace("_", " ")):
            for y in self._wn.synsets(b.replace("_", " ")):
                if x.pos() == y.pos():
                    best = max(best, x.path_similarity(y) or 0.0)
        return best >= self.path_threshold

    def _same_group(self, a: str, b: str) -> bool:
        ga, gb = self._gid(a), self._gid(b)
        return ga is not None and ga == gb

    # ---- contrasts learned from the knowledge base
    def learn_contrasts(self, keyword_sets: Sequence[FrozenSet[str]], min_shared: int = 2) -> None:
        """Two KB questions that share >= min_shared keywords and differ by exactly one word each declare those words as
        alternatives (swim / bike). Stored as direct pairs, never chained, so one noisy pair cannot spread."""
        edges: Set[FrozenSet[str]] = set()
        for i, a in enumerate(keyword_sets):
            for b in keyword_sets[i + 1:]:
                ao, bo = a - b, b - a
                if len(ao) == 1 and len(bo) == 1 and len(a & b) >= min_shared:
                    (x,), (y,) = tuple(ao), tuple(bo)
                    if not self._same_group(x, y):
                        edges.add(frozenset((x, y)))
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in edges:
            x, y = tuple(e)
            parent[find(x)] = find(y)
        self._edges = edges
        self._comp = {w: find(w) for w in list(parent)}

    _edges: Set[FrozenSet[str]] = frozenset()
    _comp: Dict[str, str] = {}

    def contrast(self, a: str, b: str) -> bool:
        """Alternatives of one set (swim / bike / run), unless the KB's synonym groups say the two words mean the same."""
        ca, cb = self._comp.get(a), self._comp.get(b)
        return a != b and ca is not None and ca == cb and not self._same_group(a, b)

    @property
    def contrast_sets(self) -> List[List[str]]:
        """The learned sets, for display."""
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in self._edges:
            x, y = tuple(e)
            parent[find(x)] = find(y)
        out: Dict[str, List[str]] = {}
        for w in list(parent):
            out.setdefault(find(w), []).append(w)
        return [sorted(v) for v in out.values() if len(v) > 1]
