"""Tokenisation used by every word-level metric.

Kept in one place so the metrics are auditable and reproducible: change the
stoplist here and every score moves together.
"""
from __future__ import annotations

import re

# Units and structural words appear in nearly every recipe, so counting them
# inflates overlap without carrying identity. salt/water/oil are excluded for
# the same reason: they belong to almost everything.
STOPWORDS = {
    "the", "and", "for", "with", "into", "from", "that", "this", "then", "are",
    "was", "add", "until", "about", "all", "its", "you", "your", "have", "has",
    "not", "but", "any", "over", "off", "out", "use", "using", "can", "let",
    "tsp", "tbsp", "cup", "cups", "teaspoon", "teaspoons", "tablespoon",
    "tablespoons", "gram", "grams", "litre", "litres", "ounce", "ounces",
    "minute", "minutes", "hour", "hours", "each", "some", "more", "few",
    "little", "well", "also", "when", "while", "till",
    "salt", "water", "oil",
}


def tokenise(text: str) -> set[str]:
    """Distinct content words: lowercase, letters only, 4+ chars, no stopwords."""
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in STOPWORDS}