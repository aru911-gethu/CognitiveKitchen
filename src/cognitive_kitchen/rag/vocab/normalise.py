"""Tier 1: rule-based reduction of raw ingredient lines.

Strips quantities, units and unambiguous preparation words. Deliberately
conservative: words that might change identity rather than form are left alone
for the LLM to judge.

    "1 inch fresh ginger, chopped"  ->  "fresh ginger"
    "Salt to taste"                 ->  "salt"
    "1 tsp chilli powder"           ->  "chilli powder"   (powder kept: a dried
                                                            spice is not the same
                                                            as a fresh chilli)
"""
from __future__ import annotations

import re

UNIT = (r"cups?|tbsp|tsps?|tablespoons?|teaspoons?|lbs?|pounds?|oz|ounces?|kgs?|"
        r"gms?|grams?|ml|litres?|liters?|quarts?|qts?|pints?|gallons?|inch|inches|"
        r"cloves?|bunch(?:es)?|sticks?|pinch(?:es)?|glass(?:es)?|nos?|packets?|"
        r"cans?|strands?|sprigs?|slices?|pieces?")

# Unambiguous preparation: changes the form, never the ingredient.
PREP = (r"finely|coarsely|roughly|thinly|freshly|well|lightly|slightly|"
        r"chopped|minced|grated|sliced|diced|cubed|crushed|halved|quartered|"
        r"julienne[d]?|shredded|mashed|beaten|whisked|peeled|seeded|deseeded|"
        r"washed|rinsed|drained|soaked|boiled|cooked|cleaned|trimmed|broken|"
        r"cut|torn|bruised|deveined|skinned|boneless|skinless|melted|softened|warmed")

# Notes about amount or optionality, never identity.
NOTE = (r"to taste|as per taste|as needed|as required|as desired|if available|"
        r"if desired|optional(?:ly)?|opt\.?|for garnish|to garnish|for frying|"
        r"for deep frying|for pan frying|for tempering|a few|few|handful|"
        r"approximate(?:ly)?|approx\.?|or more|or less|plus more|to serve")

# Size or vagueness qualifiers.
QUALIFIER = r"small|medium|large|big|tiny|whole|half|quarter|thick|thin|ripe|raw"

# Words that look like preparation but change what the thing IS. Never stripped
# here; the LLM decides whether they merge.
IDENTITY_BEARING = {"powder", "ground", "dried", "dry", "fresh", "paste",
                    "puree", "juice", "milk", "seeds", "leaves", "flour",
                    "oil", "sauce", "concentrate", "extract", "essence"}


# "gram" is both a unit of mass and a pulse family -- gram flour, green gram,
# black gram. Digits are stripped before units, so by then there is no number
# left to disambiguate, and a bare "gram flour" would lose its first word and
# become plain "flour", which is wheat. These phrases are masked instead,
# stripped around, and restored.
#
# "cloves" is deliberately NOT protected: this corpus uses it as a unit of
# garlic ("2 cloves garlic"), and guarding it split a clean "garlic" into six
# variants. The spice loses, because the unit is the common case here.
PROTECTED = (
    r"(?:green|black|bengal|horse|roasted|split)\s+gram",
    r"gram\s+(?:flour|dal|dhal|daal|lentils?)",
)
_MASK = "zqx{}xqz"


def _protect(s: str) -> tuple[str, list[str]]:
    held: list[str] = []
    for pattern in PROTECTED:
        def keep(match: re.Match) -> str:
            held.append(match.group(0))
            return f" {_MASK.format(len(held) - 1)} "
        s = re.sub(pattern, keep, s)
    return s, held


def _restore(s: str, held: list[str]) -> str:
    for index, text in enumerate(held):
        s = s.replace(_MASK.format(index), text.strip())
    return re.sub(r"\s+", " ", s).strip()


def strip_quantities(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)                       # (15-ounce), (optional)
    s = re.sub(r"[\u00bd\u00bc\u00be\u2153\u2154]", " ", s)
    s = re.sub(r"\b\d+\s*/\s*\d+\b", " ", s)             # 1/2
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)             # 3, 2.5
    s, held = _protect(s)
    s = re.sub(rf"\b(?:{UNIT})\b\.?", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.-/&")
    return _restore(s, held)


def strip_prep(text: str) -> str:
    s = text
    s = re.sub(rf"\b(?:{NOTE})\b", " ", s)
    s = re.sub(rf"\b(?:{PREP})\b", " ", s)
    s = re.sub(rf"\b(?:{QUALIFIER})\b", " ", s)
    s = re.sub(r"\b(?:of|and|or|the|a|an|with|in|on|for|each|per)\b", " ", s)
    s = re.sub(r"[^a-z /&-]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" ,.-/&")


def normalise(line: str) -> str | None:
    """Reduce a raw ingredient line to a candidate ingredient name."""
    s = strip_prep(strip_quantities(line))
    if not s or len(s) < 3:
        return None
    words = s.split()
    if len(words) > 6:                    # prose that leaked into the list
        return None
    return " ".join(words)


def is_identity_bearing(name: str) -> bool:
    """True when the name contains a word the rules must not collapse."""
    return any(w in IDENTITY_BEARING for w in name.split())