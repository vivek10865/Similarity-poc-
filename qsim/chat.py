"""Terminal chat:  py -m qsim chat   (answers from the predefined Q&A, reusing answers for similar questions)."""
from __future__ import annotations

import sys

from .bot import DEFAULT_KB, KnowledgeBaseError, QABot
from .checker import InvalidQuestionError


def run(kb: str = str(DEFAULT_KB), session_dir: str = "sessions", wordnet: str = "auto") -> int:
    try:
        bot = QABot(kb, session_dir, wordnet=wordnet)
    except KnowledgeBaseError as exc:
        print(f"[error] {exc}")
        return 2
    print("Ask a question. Blank line or Ctrl+C to quit.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        try:
            turn = bot.ask(q)
        except InvalidQuestionError as exc:
            print(f"  [error] {exc}")
            continue
        print(f"\nBot: {turn.answer}")


if __name__ == "__main__":
    sys.exit(run())
