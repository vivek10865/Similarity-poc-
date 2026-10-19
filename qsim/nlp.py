"""Deterministic question analysis with a FROZEN spaCy pipeline (tagger, lemmatiser, parser, entity recogniser,
static word vectors). Nothing is trained or fine-tuned here and the same text always gives the same analysis."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Optional, Tuple

from .text import stem

_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
_NAME_ENTS = {"GPE", "LOC", "FAC", "ORG", "PERSON", "EVENT", "NORP", "WORK_OF_ART", "PRODUCT"}
_LIGHT_VERBS = {"be", "do", "have", "get", "go", "make", "let", "tell", "know", "want", "need", "take", "give", "use",
                "possible", "able", "expect", "find", "someone", "anyone", "anything", "something", "else", "another", "other", "process", "procedure"}
_NEGATIVE_WORDS = {"prohibit", "forbid", "ban", "disallow", "illegal", "unallowed", "forbidden", "banned", "prohibited", "no", "never", "without"}
_SHIFTS = {"last": "past", "previous": "past", "prior": "past", "past": "past", "next": "future", "upcoming": "future", "future": "future",
           "this": "present", "current": "present", "present": "present"}
_TIME_NOUNS = {"year", "month", "week", "season", "edition", "time", "day", "event"}
_CONTRACT = re.compile(r"\b(what|whats|where|wheres|who|whos|how|hows|when|whens|there|theres|here|heres|that|thats)(?:'s|’s|s)\b(?=\s)", re.I)
_NUMBER = re.compile(r"\d+(?:[.:]\d+)?")
_QTYPES = [("when", re.compile(r"^(?:when|what time|what date|by when)\b")), ("where", re.compile(r"^where\b")),
           ("who", re.compile(r"^who\b")), ("how_many", re.compile(r"^how many\b")),
           ("how_much", re.compile(r"^how much\b")),
           ("how_to", re.compile(r"^how (?:do|does|can|could|should|would|to)\b")), ("why", re.compile(r"^why\b"))]


@dataclass(frozen=True)
class Keyword:
    key: str        # stem, used for matching
    lemma: str
    pos: str        # NOUN | PROPN | VERB | ADJ | ADV


@dataclass(frozen=True)
class Analysis:
    text: str
    keywords: Tuple[Keyword, ...]
    numbers: FrozenSet[str]
    names: FrozenSet[str]
    qtype: Optional[str]
    negated: bool
    shifts: FrozenSet[str] = frozenset()      # time reference: past / future / present (last year vs this year)


def load_spacy(model: str = "en_core_web_md"):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("pip install spacy") from exc
    try:
        return spacy.load(model)
    except OSError as exc:
        raise RuntimeError(f"spaCy model '{model}' is not installed: py -m spacy download {model}") from exc


class Analyzer:
    def __init__(self, nlp=None, dominant: FrozenSet[str] = frozenset(), model: str = "en_core_web_md", cache_size: int = 4096) -> None:
        self.nlp = nlp or load_spacy(model)
        self.model = model
        self.dominant = frozenset(w.lower() for w in dominant)
        self.known_names: FrozenSet[str] = frozenset()
        self.analyze = lru_cache(maxsize=cache_size)(self._analyze)

    def set_known_names(self, names) -> None:
        """Names found in the KB (capitalised there), so they are still recognised when a user types them in lower case."""
        self.known_names = frozenset(names)
        self.analyze.cache_clear()

    def word_vector_similarity(self, a: str, b: str) -> Optional[float]:
        va, vb = self.nlp.vocab[a], self.nlp.vocab[b]
        if not (va.has_vector and vb.has_vector):
            return None
        k2r = self.nlp.vocab.vectors.key2row
        sa, sb = self.nlp.vocab.strings[a], self.nlp.vocab.strings[b]
        if a != b and k2r.get(sa) is not None and k2r.get(sa) == k2r.get(sb):
            return None          # the model stores one shared vector for both words (cosine 1.0 would be meaningless)
        sim = float(va.similarity(vb))
        return None if a != b and sim > 0.98 else sim      # two different words with an identical vector: a model artefact, not a meaning

    def _analyze(self, text: str) -> Analysis:
        text = _CONTRACT.sub(lambda m: m.group(1) + " is", text)       # whats / what's -> what is, so the 's' is not read as a word
        doc = self.nlp(text)
        low = text.lower().strip()
        kws, names, seen, negated, shifts = [], set(), set(), False, set()
        for tok in doc:
            lemma, word = tok.lemma_.lower(), tok.lower_
            if word in _SHIFTS and tok.i + 1 < len(doc) and doc[tok.i + 1].lemma_.lower() in _TIME_NOUNS:
                shifts.add(_SHIFTS[word])
            if tok.is_space or tok.is_punct or len(word) == 1 and word.isalpha() and word not in ("i", "a"):
                continue
            if tok.dep_ == "neg" or lemma in _NEGATIVE_WORDS or word in _NEGATIVE_WORDS:
                negated = True
                continue
            if lemma in self.dominant or word in self.dominant:
                continue
            if tok.pos_ == "PROPN" or tok.ent_type_ in _NAME_ENTS or word in self.known_names:
                if not tok.is_stop and not any(c.isdigit() for c in word):
                    names.add(lemma)
                continue
            if tok.like_num or any(c.isdigit() for c in word):
                continue
            if tok.is_stop or tok.pos_ not in _CONTENT_POS or lemma in _LIGHT_VERBS:
                continue
            if tok.pos_ == "VERB":
                parts = [c for c in tok.children if c.dep_ == "prt"]
                if parts:
                    lemma = f"{lemma}_{parts[0].lemma_.lower()}"
            key = stem(lemma.replace("_", "").replace("-", ""))
            if key not in seen:
                seen.add(key)
                kws.append(Keyword(key, lemma, tok.pos_))
        return Analysis(text, tuple(kws), frozenset(_NUMBER.findall(low)), frozenset(names),
                        next((n for n, rx in _QTYPES if rx.search(low)), None), negated, frozenset(shifts))
