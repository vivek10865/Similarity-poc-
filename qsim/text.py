"""Text cleaning, normalisation and tokenisation helpers (pure functions)."""
from __future__ import annotations

import re
import unicodedata

_QUOTES = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)
_WS = re.compile(r"\s+")

_CONTRACTIONS = [
    (re.compile(r"\bwhats\b"), "what is"),
    (re.compile(r"\bcan't\b"), "cannot"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"\bi'm\b"), "i am"),
    (re.compile(r"'s\b"), " is"),
]
# Fold negated phrases into a single positive term so "not allowed" == "prohibited".
_NEGATION_FOLDS = [
    (re.compile(r"\b(?:is |are |am )?not (?:allowed|permitted|legal)\b"), " prohibited "),
    (re.compile(r"\bcannot (?:use|wear|bring)\b"), " prohibited "),
    (re.compile(r"\b(?:is |are )?not (?:required|mandatory|compulsory)\b"), " optional "),
]

STOPWORDS = frozenset(
    """a an the is are was were be been being am do does did doing have has had i me my we our you your
    it its this that these those of to in on at for from by with about as into and or if than then so
    what which who whom whose when where why how can could should would will shall may might must
    there here any some please tell know want like also get need possible way wondering just really very
    much many more most up out off over under again too""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def clean(text: str) -> str:
    """Unicode-normalise and collapse whitespace; keeps case."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).translate(_QUOTES)).strip()


def normalize(text: str) -> str:
    """Lower-case, expand contractions and fold negated phrases."""
    t = clean(text).lower()
    for rx, rep in _CONTRACTIONS:
        t = rx.sub(rep, t)
    for rx, rep in _NEGATION_FOLDS:
        t = rx.sub(rep, t)
    return _WS.sub(" ", t).strip()


def stem(tok: str) -> str:
    """Very light stemmer (deterministic, dependency free)."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        tok = tok[:-3] + "y"
    elif tok.endswith("ing") and len(tok) > 5:
        tok = tok[:-3]
    elif tok.endswith("ed") and len(tok) > 4:
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith("ss") and len(tok) > 3:
        tok = tok[:-1]
    if tok.endswith("e") and len(tok) > 4:
        tok = tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def content_tokens(canonical_text: str) -> list[str]:
    """Stopword-free, stemmed tokens of an already canonicalised string."""
    return [stem(t) for t in tokenize(canonical_text) if t not in STOPWORDS]


def dominant_words(texts, fraction: float = 0.4) -> frozenset:
    """Words that appear in at least ``fraction`` of the texts carry little information (like 'ironman' in an IRONMAN
    knowledge base). Derived from the data, nothing hard-coded."""
    texts = list(texts)
    if not texts:
        return frozenset()
    counts: dict = {}
    for t in texts:
        for w in set(tokenize(normalize(t))) - STOPWORDS:
            counts[w] = counts.get(w, 0) + 1
    return frozenset(w for w, c in counts.items() if c / len(texts) >= fraction)
