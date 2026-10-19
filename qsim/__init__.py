"""qsim - deterministic semantic question matching (frozen spaCy analysis + keyword/slot rules; no LLM, no encoders)."""
from .checker import RELATED, SAME, UNRELATED, InvalidQuestionError, SimilarityChecker, SimilarityResult
from .config import Config
from .lexicon import Lexicon
from .nlp import Analyzer
from .spell import SpellFixer

__all__ = ["SimilarityChecker", "SimilarityResult", "Config", "InvalidQuestionError", "SAME", "RELATED", "UNRELATED",
           "Analyzer", "Lexicon", "SpellFixer"]
__version__ = "0.2.0"
