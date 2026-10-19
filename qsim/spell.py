"""Tiny typo corrector: fixes a misspelt word only when it is very close to a word we know ("iroman" -> "ironman")."""
from __future__ import annotations

import difflib
import re
from typing import Iterable, List, Set

from .text import STOPWORDS

_WORD = re.compile(r"[A-Za-z][A-Za-z']+")


class SpellFixer:
    def __init__(self, vocabulary: Iterable[str], cutoff: float = 0.84, min_len: int = 5) -> None:
        self.vocab: Set[str] = {w.lower() for w in vocabulary if len(w) >= min_len}
        self._sorted: List[str] = sorted(self.vocab)
        self.cutoff, self.min_len = cutoff, min_len

    @classmethod
    def from_texts(cls, texts: Iterable[str], extra: Iterable[str] = (), **kw) -> "SpellFixer":
        words = {w.lower() for t in texts for w in _WORD.findall(t)} | {w.lower() for w in extra}
        return cls(words - STOPWORDS, **kw)

    def fix(self, text: str) -> str:
        def repl(m: re.Match) -> str:
            w = m.group(0)
            lw = w.lower()
            if len(lw) < self.min_len or lw in self.vocab or lw in STOPWORDS:
                return w
            close = difflib.get_close_matches(lw, self._sorted, n=1, cutoff=self.cutoff)
            return close[0] if close else w
        return _WORD.sub(repl, text)
