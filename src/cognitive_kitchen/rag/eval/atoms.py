"""Content matching between retrieved text and a golden recipe.

No id mapping. The golden dataset records `atoms`, strings unique to one recipe
within the curated 50. They are not necessarily unique across the 184 pipeline
recipes, so each atom is filtered by how many recipes actually contain it: an
atom appearing everywhere carries no identity.

A passage matches a recipe when it contains the title atom, or two independent
specific atoms.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import Corpus
from .golden import GoldenRecipe

MAX_RECIPES_PER_ATOM = 3
MIN_ATOM_LENGTH = 8


@dataclass(frozen=True)
class Fingerprint:
    recipe_id: str
    title: str
    title_atom: str | None
    atoms: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.title_atom is not None or len(self.atoms) >= 2


def build_fingerprints(golden: tuple[GoldenRecipe, ...],
                       corpus: Corpus) -> dict[str, Fingerprint]:
    """Keep only atoms specific enough to identify a recipe in this corpus."""
    cards = [corpus.text[s.start:s.end].lower() for s in corpus.spans]

    def spread(atom: str) -> int:
        return sum(1 for card in cards if atom in card)

    out: dict[str, Fingerprint] = {}
    for recipe in golden:
        title_lower = recipe.title.lower()
        title_atom = None
        specific: list[str] = []
        for atom in recipe.atoms:
            if len(atom) < MIN_ATOM_LENGTH:
                continue
            n = spread(atom)
            if n == 0 or n > MAX_RECIPES_PER_ATOM:
                continue
            if atom == title_lower or atom in title_lower:
                title_atom = atom
            else:
                specific.append(atom)
        out[recipe.recipe_id] = Fingerprint(recipe.recipe_id, recipe.title,
                                            title_atom, tuple(specific))
    return out


def matches(text: str, fingerprint: Fingerprint) -> bool:
    low = text.lower()
    if fingerprint.title_atom and fingerprint.title_atom in low:
        return True
    return sum(1 for atom in fingerprint.atoms if atom in low) >= 2


def coverage_report(fingerprints: dict[str, Fingerprint]) -> dict:
    usable = [f for f in fingerprints.values() if f.usable]
    return {
        "recipes": len(fingerprints),
        "usable": len(usable),
        "with_title_atom": sum(1 for f in usable if f.title_atom),
        "mean_specific_atoms": round(
            sum(len(f.atoms) for f in usable) / max(len(usable), 1), 2),
        "unusable": [f.recipe_id for f in fingerprints.values() if not f.usable],
    }