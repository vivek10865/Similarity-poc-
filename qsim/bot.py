"""Question-answering bot with automatic similarity checks.

For every new question:
  1. compare it with EVERY question already asked in this session;
     if one is SAME -> reuse the answer that was already given to it;
  2. otherwise compare it with the predefined knowledge-base questions;
     if one is SAME -> use its predefined answer;
  3. otherwise say no answer was found and suggest the closest known questions.
Every question is saved to a local file (one JSON line per question) and the similarity report is printed in the terminal.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .checker import SimilarityChecker, SimilarityResult
from .lexicon import Lexicon
from .nlp import Analyzer
from .spell import SpellFixer
from .text import dominant_words

SESSION, KNOWLEDGE_BASE, NONE = "session", "knowledge_base", "none"
DEFAULT_KB = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


class KnowledgeBaseError(ValueError):
    pass


def load_knowledge_base(path: str | Path) -> List[Tuple[str, str]]:
    """Read ``[{"question":..., "answer":...}, ...]`` (or ``{"qa": [...]}``) and validate it."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KnowledgeBaseError(f"knowledge base not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"knowledge base is not valid JSON: {exc}") from exc
    items = raw.get("qa") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise KnowledgeBaseError("knowledge base must be a non-empty list of {question, answer}")
    out: List[Tuple[str, str]] = []
    for n, it in enumerate(items, 1):
        q, a = (it.get("question"), it.get("answer")) if isinstance(it, dict) else (None, None)
        if not (isinstance(q, str) and q.strip() and isinstance(a, str) and a.strip()):
            raise KnowledgeBaseError(f"item {n}: needs non-empty 'question' and 'answer' strings")
        out.append((q.strip(), a.strip()))
    return out


def load_kb_extras(path: str | Path) -> Tuple[Dict[str, List[str]], List[List[str]], List[str]]:
    """Optional data in the KB file: per-question ``paraphrases`` (extra ways to ask it) and top-level ``synonyms`` (word groups)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("qa") if isinstance(raw, dict) else raw
    paras = {it["question"].strip(): [p.strip() for p in it.get("paraphrases", []) if isinstance(p, str) and p.strip()]
             for it in items if isinstance(it, dict) and isinstance(it.get("question"), str)}
    groups = raw.get("synonyms", []) if isinstance(raw, dict) else []
    names = [n for n in (raw.get("names", []) if isinstance(raw, dict) else []) if isinstance(n, str)]
    return paras, [[w for w in g if isinstance(w, str)] for g in groups if isinstance(g, list)], names


@dataclass(frozen=True)
class Turn:
    n: int
    question: str
    answer: str
    source: str                          # session | knowledge_base | none
    matched_question: Optional[str]
    result: Optional[SimilarityResult]   # similarity with matched_question (or the closest one when none)


def _bar(pct: float, width: int = 30) -> str:
    n = int(round(max(0.0, min(100.0, pct)) / 100 * width))
    return "[" + "#" * n + "-" * (width - n) + "]"


class QABot:
    """
    bot = QABot()                       # loads data/knowledge_base.json, creates sessions/session_<time>.jsonl lazily
    turn = bot.ask("What is the swim cutoff?")   # prints the similarity report, returns the answer in turn.answer
    """

    NO_ANSWER = "Sorry, I don't have an answer for that yet."

    def __init__(self, kb_path: str | Path = DEFAULT_KB, session_dir: str | Path = "sessions",
                 checker: SimilarityChecker | None = None, printer: Callable[[str], None] = print,
                 session_id: str | None = None, wordnet: str = "auto") -> None:
        self.kb = load_knowledge_base(kb_path)
        self._kb_answers: Dict[str, str] = dict(self.kb)
        kb_questions = [q for q, _ in self.kb]
        paras, groups, names = load_kb_extras(kb_path)
        # candidate text -> canonical KB question (the question itself plus its optional paraphrases)
        self._cand: Dict[str, str] = {}
        for q in kb_questions:
            self._cand.setdefault(q, q)
            for p in paras.get(q, []):
                self._cand.setdefault(p, q)
        self.checker = checker or SimilarityChecker(
            Analyzer(dominant=dominant_words(kb_questions)), spell_fixer=SpellFixer.from_texts(kb_questions),
            lexicon=Lexicon(groups, wordnet), kb_questions=kb_questions, extra_names=names)
        self.printer = printer
        sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = Path(session_dir) / f"session_{sid}.jsonl"
        self._asked: Dict[str, str] = {}      # question -> answer given (this session)
        self._count = 0
        self._lock = threading.Lock()
        self.printer(f"[similarity] ready - algorithm only (spaCy {self.checker.analyzer.model}, WordNet {self.checker.lexicon.wordnet_status}); "
                     f"{len(self.kb)} predefined answers; session file: {self.session_file}")

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> Turn:
        with self._lock:
            earlier = self.checker.rank(question, list(self._asked))      # also validates the question
            kb = []
            top_earlier = earlier[0] if earlier else None
            if top_earlier and top_earlier[1].is_same:
                answer, source, matched, result = self._asked[top_earlier[0]], SESSION, top_earlier[0], top_earlier[1]
            else:
                kb = self._rank_kb(question)
                if kb and kb[0][1].is_same:
                    answer, source, matched, result = self._kb_answers[kb[0][0]], KNOWLEDGE_BASE, kb[0][0], kb[0][1]
                else:
                    answer, source = self.NO_ANSWER, NONE
                    matched, result = (kb[0] if kb else (None, None))
                    close = [q for q, r in kb[:3] if r.raw_pct >= 50]
                    if close:
                        answer += " Related questions I can answer: " + " | ".join(close)
            self._count += 1
            if source != NONE:           # only answered questions can be reused later
                self._asked[question] = answer
            turn = Turn(self._count, question, answer, source, matched, result)
            self._save(turn)
        self.printer(self._report(turn, top_earlier, kb[0] if kb else None))
        return turn

    def _rank_kb(self, question: str) -> List[Tuple[str, SimilarityResult]]:
        """Rank the KB questions; every question may also be matched through its paraphrases (best text per question)."""
        out, seen = [], set()
        for text, r in self.checker.rank(question, list(self._cand)):
            canon = self._cand[text]
            if canon not in seen:
                seen.add(canon)
                out.append((canon, r))
        return out

    # ------------------------------------------------------------------ internals
    def _save(self, t: Turn) -> None:
        try:
            self.session_file.parent.mkdir(parents=True, exist_ok=True)
            row = {"time": datetime.now().isoformat(timespec="seconds"), "n": t.n, "question": t.question,
                   "answer": t.answer, "source": t.source, "matched_question": t.matched_question,
                   "similarity_pct": t.result.similarity_pct if t.result else None,
                   "label": t.result.label if t.result else None}
            with open(self.session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:  # saving must never break the chatbot
            self.printer(f"[similarity] WARNING: could not save question: {exc}")

    @staticmethod
    def _line(title: str, pair) -> List[str]:
        q, r = pair
        out = [f"  {title}: \"{q}\"", f"    similarity {_bar(r.similarity_pct)} {r.similarity_pct:5.1f}%   verdict {r.label.upper()}"
               f"   confidence {r.confidence:.0%}",
               f"    matched keywords: {', '.join(f'{a}={b}' if a != b else a for a, b in r.matched) or '-'}"
               f"   | only in yours: {', '.join(r.only_a) or '-'}   | only in it: {', '.join(r.only_b) or '-'}"]
        out += [f"    why not same: {h}" for h in r.hard_conflicts]
        return out

    def _report(self, t: Turn, earlier, kb) -> str:
        lines = [f"\n[similarity] Q{t.n}: \"{t.question}\""]
        lines += self._line("Closest earlier question (this session)", earlier) if earlier else \
            ["  No earlier question in this session."]
        if kb:
            lines += self._line("Closest predefined question", kb)
        lines.append({SESSION: "  => SAME as an earlier question: reusing the answer already given",
                      KNOWLEDGE_BASE: "  => not asked before: answer taken from the predefined answers",
                      NONE: "  => no similar question found: no answer available"}[t.source])
        return "\n".join(lines)
