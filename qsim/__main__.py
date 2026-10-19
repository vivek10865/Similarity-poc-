"""CLI:  py -m qsim ui | chat | compare "q1" "q2" | words w1 w2   (algorithms only: spaCy static vectors + keyword/slot rules)"""
from __future__ import annotations

import argparse

from .bot import DEFAULT_KB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="qsim")
    ap.add_argument("--wordnet", default="auto", choices=["auto", "on", "off"], help="use NLTK WordNet synonyms if installed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    w = sub.add_parser("words", help="static-vector similarity of two single words")
    w.add_argument("a"); w.add_argument("b")
    ui = sub.add_parser("ui", help="chatbot web page (answers in the page, similarity report in the terminal)")
    ui.add_argument("--port", type=int, default=8000); ui.add_argument("--no-browser", action="store_true")
    ui.add_argument("--kb", default=None); ui.add_argument("--sessions", default="sessions")
    ch = sub.add_parser("chat", help="terminal chat with predefined answers + similarity checks")
    ch.add_argument("--kb", default=None); ch.add_argument("--sessions", default="sessions")
    args = ap.parse_args(argv)

    if args.cmd in ("compare", "words"):
        from .checker import SimilarityChecker
        from .bot import QABot
        if args.cmd == "words":
            from .nlp import Analyzer
            s = Analyzer().word_vector_similarity(args.a, args.b)
            print(f"static word-vector similarity {args.a!r} vs {args.b!r}: {s if s is None else round(s, 2)}")
            return 0
        bot = QABot(DEFAULT_KB, "sessions", printer=lambda *_: None, wordnet=args.wordnet)   # same setup as the chatbot, nothing saved
        r = bot.checker.compare(args.a, args.b)
        print(r.summary())
        print("matched:", ", ".join(f"{a}={b}" for a, b in r.matched) or "-", "| only A:", ", ".join(r.only_a) or "-",
              "| only B:", ", ".join(r.only_b) or "-")
        for line in r.reasons:
            print("  ", line)
        return 0
    if args.cmd == "ui":
        from .ui import serve
        serve(args.kb or str(DEFAULT_KB), args.sessions, args.port, not args.no_browser, args.wordnet)
        return 0
    from .chat import run
    return run(args.kb or str(DEFAULT_KB), args.sessions, args.wordnet)


if __name__ == "__main__":
    raise SystemExit(main())
