"""Can I actually make this? -- the graph auditing the generator.

The chat answers from retrieved text. This checks that answer against what the
cook actually has, which retrieval cannot do: "missing" appears in no document,
and no amount of similarity search will infer it.

Two steps, deliberately split by what each is good at.

  1. Resolve the pantry. Deterministic vocabulary first, an LLM only for the
     lines it could not place. "urad daal" is a spelling the rule-based
     normaliser already handles; "black gram" is a different name for the same
     pulse and it does not. Paying for a model on every line would be waste, so
     the model sees only the leftovers -- and its answer is cached, because a
     pantry changes far more slowly than it is asked about.

  2. Traverse. Everything after resolution is graph work and stays
     deterministic: what is missing, and which of the missing are already
     covered by something on the shelf.

The recipe is identified by its graph key, taken from the answer's cited source
-- never by parsing the generated text. Generated prose is not a data structure,
and an ingredient list re-derived from it would be a second extractor to
maintain and a second thing to be wrong.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ...config import settings
from ..vocab import ingredients as vocab
from . import client, traverse

PROMPT = """A cook listed what is in their kitchen. Some lines did not match our
ingredient vocabulary. Map each one to the closest name in the KNOWN list, or to
null if it is genuinely not there.

Return only JSON: {{"line": "known name or null", ...}}

Map only genuine synonyms, regional names and misspellings:
  "black gram"   -> "urad dal"      same pulse, different name
  "curd"         -> "yogurt"        same thing
  "kadai"        -> null            equipment, not an ingredient
  "unicorn milk" -> null            not real, do not guess

Do NOT map an ingredient to a different ingredient just because they are
similar. "cashew" is not "almond". A wrong match silently tells the cook they
can make something they cannot.

KNOWN: {known}

LINES: {lines}"""


@dataclass
class PantryResolution:
    """What the cook typed, mapped onto names the graph can match."""

    canonical: list[str] = field(default_factory=list)
    mapped: dict[str, str] = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    llm_used: bool = False
    cost_usd: float = 0.0


def known_ingredients() -> list[str]:
    """Every ingredient name the graph holds, which is what we diff against."""
    rows = client.run("MATCH (i:Ingredient) RETURN i.name AS name ORDER BY name")
    return [row["name"] for row in rows]


def _cache_path(lines: list[str]) -> Path:
    digest = hashlib.sha256("\x00".join(sorted(lines)).encode("utf-8")).hexdigest()
    return settings.data_dir / "eval" / f"pantry-{digest[:16]}.json"


def resolve_pantry(lines, model: str = "gpt4o-mini",
                   use_llm: bool = True) -> PantryResolution:
    """Free text -> canonical ingredient names. Rules first, model for the rest.

    The gate is membership in the graph, not whether the vocabulary returned
    something. `curated.py` deliberately lets an unrecognised term become its own
    canonical name -- safe for building a vocabulary, useless here, because
    "black gram" would resolve to "black gram" and quietly match nothing. Asking
    whether the name is a real Ingredient node is what routes genuine synonyms to
    the model instead of dropping them.
    """
    out = PantryResolution()
    raws = [str(x).strip() for x in lines if str(x).strip()]

    try:
        valid = set(known_ingredients())
    except Exception:
        # No graph. Fall back to vocabulary-only so the page still works, and
        # say nothing we cannot stand behind.
        for raw in raws:
            name = vocab.canonical(raw)
            if name:
                out.mapped[raw] = name
                out.canonical.append(name)
            else:
                out.unknown.append(raw)
        return _dedupe(out)

    leftovers: list[str] = []
    for raw in raws:
        name = vocab.canonical(raw)
        if name and name in valid:
            out.mapped[raw] = name
            out.canonical.append(name)
        else:
            leftovers.append(raw)

    if not leftovers:
        return _dedupe(out)
    if not use_llm:
        out.unknown = leftovers
        return _dedupe(out)

    cache = _cache_path(leftovers)
    guessed: dict = {}
    cached = False
    if cache.exists():
        try:
            guessed = json.loads(cache.read_text(encoding="utf-8"))
            cached = True
            out.llm_used = True
        except Exception:
            guessed, cached = {}, False

    if not cached:
        try:
            from ..registry import build, discover

            discover("cognitive_kitchen.rag.generate")
            generator = build("generator", model)
            reply = generator.complete(
                PROMPT.format(known=", ".join(sorted(valid)),
                              lines=json.dumps(leftovers)), max_tokens=400)
            brace, close = reply.find("{"), reply.rfind("}")
            guessed = json.loads(reply[brace:close + 1]) if close > brace >= 0 else {}
            out.llm_used = True
            out.cost_usd = float(getattr(generator, "cost_usd", 0.0) or 0.0)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(guessed, indent=2), encoding="utf-8")
        except Exception:
            # No key, no network: admit we do not know. Guessing which pulse
            # someone meant is worse than leaving it blank.
            out.unknown = leftovers
            return _dedupe(out)

    valid = set(known_ingredients())
    for raw in leftovers:
        name = guessed.get(raw)
        if isinstance(name, str) and name in valid:
            out.mapped[raw] = name
            out.canonical.append(name)
        else:
            out.unknown.append(raw)
    return _dedupe(out)


def _dedupe(res: PantryResolution) -> PantryResolution:
    res.canonical = sorted(set(res.canonical))
    return res


def can_i_make(key: str, pantry: list[str], max_swaps: int = 6) -> dict:
    """Given a recipe and a resolved pantry: what is missing, what can be swapped.

    `buy` is the honest answer to "what do I need to go out for": the missing
    ingredients that nothing on the shelf can stand in for. `swaps` are the ones
    already covered, and they come from the same distributional traversal Stage 5
    uses -- ingredients that keep the same company, not ones that merely
    co-occur.
    """
    cards = traverse.recipe_cards([key])
    card = cards.get(key) or {}
    needed = sorted({n for n in (card.get("ingredients") or []) if n})
    have = set(pantry)

    missing = [n for n in needed if n not in have]
    swaps: dict[str, list[str]] = {}
    buy: list[str] = []

    for item in missing[:max_swaps]:
        try:
            covered = [c["name"] for c in traverse.substitutes(item, limit=8)
                       if c["name"] in have]
        except Exception:
            covered = []
        if covered:
            swaps[item] = covered
        else:
            buy.append(item)
    buy.extend(missing[max_swaps:])

    return {"key": key, "title": card.get("title") or "(untitled)",
            "needed": needed, "have": sorted(have & set(needed)),
            "missing": missing, "buy": sorted(buy), "swaps": swaps,
            "can_make": not buy and bool(needed)}