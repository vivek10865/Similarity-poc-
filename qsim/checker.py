"""Deterministic question matcher: frozen spaCy analysis + keyword alignment + slot checks. No training, no LLM, no encoder.

For a pair of questions:
 1. extract content keywords (lemmas of nouns / verbs / adjectives), numbers, names, question type, negation;
 2. align keywords: same stem, same synonym group (KB data / WordNet), or static word-vector cosine >= threshold;
 3. score = harmonic mean of the (IDF-weighted) share of each question's keywords that found a partner;
 4. hard conflicts (cap the score, block SAME): different numbers, different names, different question type,
    negation differs, KB-learned contrast words (swim vs bike), a single unmatched noun on each side that is clearly different;
 5. label: SAME / RELATED / UNRELATED.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .config import Config
from .lexicon import Lexicon
from .nlp import Analysis, Analyzer, Keyword
from .spell import SpellFixer

log = logging.getLogger("qsim")
SAME, RELATED, UNRELATED = "same", "related", "unrelated"


class InvalidQuestionError(ValueError):
    pass


@dataclass(frozen=True)
class SimilarityResult:
    label: str                      # same | related | unrelated
    similarity_pct: float           # 0-100, after penalties / caps
    confidence: float
    hard_conflicts: Tuple[str, ...] = ()
    soft_differences: Tuple[str, ...] = ()
    reasons: Tuple[str, ...] = ()
    raw_pct: float = 0.0            # score before the conflict cap (ranks the "closest" question)
    matched: Tuple[Tuple[str, str], ...] = ()     # aligned keyword pairs (question A word, question B word)
    only_a: Tuple[str, ...] = ()    # keywords of A without a partner
    only_b: Tuple[str, ...] = ()
    differing_words: FrozenSet[str] = frozenset()

    @property
    def is_same(self) -> bool:
        return self.label == SAME

    def summary(self) -> str:
        return f"{self.label.upper()}  similarity={self.similarity_pct:.0f}%  confidence={self.confidence:.0%}"


class SimilarityChecker:
    def __init__(self, analyzer: Optional[Analyzer] = None, config: Optional[Config] = None, spell_fixer: Optional[SpellFixer] = None,
                 lexicon: Optional[Lexicon] = None, kb_questions: Sequence[str] = (), extra_names: Sequence[str] = ()) -> None:
        self.config = config or Config()
        self.analyzer = analyzer or Analyzer()
        self.spell_fixer = spell_fixer
        self.lexicon = lexicon or Lexicon()
        self._df: Dict[str, int] = {}
        self._kb_names: FrozenSet[str] = frozenset()
        self._n = 0
        if kb_questions:
            self.fit(kb_questions, extra_names)

    # ------------------------------------------------------------------ learning from the KB (deterministic counting)
    def fit(self, kb_questions: Sequence[str], extra_names: Sequence[str] = ()) -> None:
        analyses = [self.analyzer.analyze(self._clean(q)) for q in kb_questions]
        self._n = len(analyses)
        self._df = {}
        self._kb_names = frozenset(n for a in analyses for n in a.names)
        self.analyzer.set_known_names(self._kb_names | {w.lower() for n in extra_names for w in n.split()})
        analyses = [self.analyzer.analyze(self._clean(q)) for q in kb_questions]       # again, now with the KB names known
        self._kb_names = frozenset(n for a in analyses for n in a.names) | {w.lower() for n in extra_names for w in n.split()}
        for a in analyses:
            for k in a.keywords:
                self._df[k.key] = self._df.get(k.key, 0) + 1
        self.lexicon.learn_contrasts([frozenset(k.key for k in a.keywords) for a in analyses])

    def _weight(self, key: str) -> float:
        return math.log(1 + (self._n + 1) / (1 + self._df.get(key, 0))) if self._n else 1.0

    _POS_CLASS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "a"}
    _POS_WEIGHT = {"NOUN": 1.0, "PROPN": 1.0, "VERB": 0.8, "ADJ": 0.8, "ADV": 0.4}

    # ------------------------------------------------------------------ public API
    def compare(self, a: str, b: str) -> SimilarityResult:
        ca, cb = self._clean(a), self._clean(b)
        if ca.lower().rstrip("?!. ") == cb.lower().rstrip("?!. "):        # identical text (also keyword-less ones like "T1")
            return SimilarityResult(SAME, 100.0, 0.99, reasons=("identical text",))
        return self._compare(self.analyzer.analyze(ca), self.analyzer.analyze(cb))

    def is_same(self, a: str, b: str) -> bool:
        return self.compare(a, b).is_same

    def compare_many(self, pairs: Sequence[Tuple[str, str]], **_ignored) -> List[SimilarityResult]:
        return [self.compare(a, b) for a, b in pairs]

    def rank(self, question: str, candidates: Sequence[str]) -> List[Tuple[str, SimilarityResult]]:
        if not candidates:
            return []
        results = self._distrust_after_rejection([self.compare(question, c) for c in candidates])
        order = sorted(range(len(candidates)),
                       key=lambda i: (results[i].is_same, results[i].similarity_pct if results[i].is_same else results[i].raw_pct),
                       reverse=True)
        return [(candidates[i], results[i]) for i in order]

    def best_matches(self, question: str, candidates: Sequence[str], top_k: int = 3) -> List[Tuple[str, SimilarityResult]]:
        return self.rank(question, candidates)[:top_k]

    def word_similarity(self, a: str, b: str) -> Optional[float]:
        return self.analyzer.word_vector_similarity(a, b)

    # ------------------------------------------------------------------ internals
    def _clean(self, q: str) -> str:
        if not isinstance(q, str) or not q.strip():
            raise InvalidQuestionError("question must be a non-empty string")
        if len(q) > self.config.max_chars:
            raise InvalidQuestionError(f"question longer than {self.config.max_chars} characters")
        q = " ".join(q.split())
        if q.isupper():            # SHOUTING would otherwise be read as names
            q = q.lower()
        return self.spell_fixer.fix(q) if self.spell_fixer else q

    def _word_sim(self, x: Keyword, y: Keyword) -> float:
        if x.key == y.key:
            return 1.0
        if self.lexicon.contrast(x.key, y.key):      # KB says these are alternatives (swim / bike): never the same word
            return 0.0
        if self.lexicon.related(x.lemma, y.lemma) or self.lexicon.related(x.key, y.key):
            return 1.0
        if self._POS_CLASS.get(x.pos) != self._POS_CLASS.get(y.pos):      # a noun is never "close" to an adverb (time vs far)
            return 0.0
        s = self.analyzer.word_vector_similarity(x.lemma, y.lemma)
        return s if s is not None and s >= self.config.vector_threshold else 0.0

    def _align(self, a: Sequence[Keyword], b: Sequence[Keyword]):
        cands = sorted(((self._word_sim(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b)), key=lambda t: (-t[0], t[1], t[2]))
        used_a, used_b, pairs = set(), set(), []
        for s, i, j in cands:
            if s <= 0:
                break
            if i not in used_a and j not in used_b:
                used_a.add(i); used_b.add(j); pairs.append((i, j, s))
        return pairs, used_a, used_b

    def _compare(self, A: Analysis, B: Analysis) -> SimilarityResult:
        cfg = self.config
        pairs, used_a, used_b = self._align(A.keywords, B.keywords)
        wa = sum(self._weight(k.key) * self._POS_WEIGHT.get(k.pos, 1) for k in A.keywords)
        wb = sum(self._weight(k.key) * self._POS_WEIGHT.get(k.pos, 1) for k in B.keywords)
        ca = sum(self._weight(A.keywords[i].key) * self._POS_WEIGHT.get(A.keywords[i].pos, 1) * s for i, _, s in pairs) / wa if wa else 0.0
        cb = sum(self._weight(B.keywords[j].key) * self._POS_WEIGHT.get(B.keywords[j].pos, 1) * s for _, j, s in pairs) / wb if wb else 0.0
        score = 2 * ca * cb / (ca + cb) if ca + cb else 0.0
        only_a = [k for i, k in enumerate(A.keywords) if i not in used_a]
        only_b = [k for j, k in enumerate(B.keywords) if j not in used_b]

        hard: List[str] = []
        soft: List[str] = []
        if A.numbers and B.numbers and A.numbers != B.numbers:
            hard.append(f"different number: {sorted(A.numbers)} vs {sorted(B.numbers)}")
        elif bool(A.numbers) != bool(B.numbers):
            hard.append(f"number only in one question: {sorted(A.numbers or B.numbers)}")
        an, bn = A.names - B.names, B.names - A.names
        if an and bn:
            hard.append(f"different name: {sorted(an)} vs {sorted(bn)}")
        elif an or bn:
            lone = an or bn
            # a place / event name that the KB knows but the other question lacks: that question is about something else
            (hard if lone & self._kb_names else soft).append(f"name only in one question: {sorted(lone)}")
        if A.qtype and B.qtype and A.qtype != B.qtype and {A.qtype, B.qtype} != {"where", "how_to"}:
            hard.append(f"question type: {A.qtype} vs {B.qtype}")
        if A.shifts != B.shifts and (A.shifts or B.shifts):
            (hard if A.shifts and B.shifts else soft).append(f"time reference: {sorted(A.shifts) or '-'} vs {sorted(B.shifts) or '-'}")
        if A.negated != B.negated:
            msg = "negation differs"
            (hard if min(ca, cb) >= 0.5 else soft).append(msg)   # with otherwise different words it may just be a paraphrase
        for x in only_a:
            for y in only_b:
                if self.lexicon.contrast(x.key, y.key):
                    hard.append(f"different detail: {x.lemma} vs {y.lemma}")
        if len(only_a) == 1 and len(only_b) == 1 and not any(h.startswith("different detail") for h in hard):
            x, y = only_a[0], only_b[0]
            if not (self.lexicon.related(x.lemma, y.lemma) or self.lexicon.related(x.key, y.key)):
                s = self.analyzer.word_vector_similarity(x.lemma, y.lemma)
                known = self._df.get(x.key, 0) > 0 and self._df.get(y.key, 0) > 0      # both are topics of the KB
                if s is None or s < cfg.noun_swap_threshold or (known and s < 0.85):
                    (hard if known else soft).append(f"different detail: {x.lemma} vs {y.lemma}")

        score = max(0.0, score - cfg.soft_penalty * len(soft))
        raw = score
        if hard:
            score = min(score, cfg.hard_cap)
            label = RELATED if raw >= cfg.related_threshold else UNRELATED
        elif score >= cfg.same_threshold and min(ca, cb) >= cfg.min_coverage:
            label = SAME
        elif score >= cfg.related_threshold:
            label = RELATED
        else:
            label = UNRELATED
        reasons = [f"conflict -> {h}" for h in hard] + [f"note -> {s}" for s in soft]
        conf = self._confidence(label, score, bool(hard), len(soft))
        return SimilarityResult(
            label, round(score * 100, 1), round(conf, 3), tuple(hard), tuple(soft), tuple(reasons), round(raw * 100, 1),
            tuple((A.keywords[i].lemma, B.keywords[j].lemma) for i, j, _ in pairs),
            tuple(k.lemma for k in only_a), tuple(k.lemma for k in only_b),
            frozenset(k.key for k in only_a) | frozenset(k.key for k in only_b))

    def _confidence(self, label: str, score: float, hard: bool, n_soft: int) -> float:
        cfg = self.config
        if label == SAME:
            c = 0.55 + 0.45 * min(1.0, (score - cfg.same_threshold) / (1 - cfg.same_threshold)) - 0.05 * n_soft
        elif hard:
            c = 0.85
        elif label == RELATED:
            c = 0.55 + 0.45 * min(1.0, abs(score - cfg.same_threshold) / cfg.same_threshold)
        else:
            c = 0.6 + 0.4 * min(1.0, (cfg.related_threshold - score) / cfg.related_threshold)
        return max(0.5, min(0.99, c))

    def _distrust_after_rejection(self, results: List[SimilarityResult]) -> List[SimilarityResult]:
        """If a rejected candidate differs from the question in a STRICT SUBSET of the words an accepted candidate differs in,
        the accepted one cannot be the better match (it differs in everything the rejected one does, plus more)."""
        rejected = [r.differing_words for r in results if r.hard_conflicts and r.differing_words]
        if not rejected:
            return results
        note = "a closer question was rejected, so this weaker match is not trusted"
        return [replace(r, label=RELATED, hard_conflicts=r.hard_conflicts + (note,), reasons=r.reasons + (f"conflict -> {note}",))
                if r.is_same and any(d < r.differing_words for d in rejected) else r for r in results]
