"""Validated configuration for the keyword/slot matcher."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    same_threshold: float = 0.55      # keyword score >= this (and no hard conflict) => SAME
    related_threshold: float = 0.30   # score >= this => RELATED (same topic, not the same question)
    hard_cap: float = 0.40            # score is capped here when a hard conflict exists
    soft_penalty: float = 0.05        # per soft difference (a name/number present in only one question)
    vector_threshold: float = 0.70    # static word-vector cosine at/above which two words count as the same word
    noun_swap_threshold: float = 0.45  # one unmatched noun on each side with cosine below this = different thing
    min_coverage: float = 0.50        # SAME also needs BOTH questions to have at least this share of their keywords matched
    max_chars: int = 2000
    cache_size: int = 4096

    def __post_init__(self) -> None:
        if not 0 < self.related_threshold < self.same_threshold <= 1:
            raise ValueError("need 0 < related_threshold < same_threshold <= 1")
        if not 0 < self.hard_cap < self.same_threshold:
            raise ValueError("hard_cap must be below same_threshold, otherwise conflicts could still pass")
        if not 0 < self.vector_threshold <= 1 or not 0 <= self.noun_swap_threshold < 1:
            raise ValueError("invalid vector_threshold / noun_swap_threshold")
        if not 0 <= self.min_coverage < 1:
            raise ValueError("min_coverage must be in [0, 1)")
        if self.max_chars < 1 or self.cache_size < 0 or self.soft_penalty < 0:
            raise ValueError("invalid max_chars / cache_size / soft_penalty")
