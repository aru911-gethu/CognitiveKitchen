"""Chat memory that survives a restart, and follow-up resolution.

Two separate jobs, and the second is the one that matters:

    persistence   conversations live in data/chat/<id>.json, so closing the
                  browser or restarting the server does not lose the thread
    resolution    "what about without dairy?" is not a question a retriever can
                  answer. It has no subject. Before retrieval, a follow-up is
                  rewritten against the recent turns into something standalone:
                  "a rice dish without dairy"

Without resolution, persistence is just a transcript. The retriever sees each
message alone, so the second turn of any real conversation retrieves nothing
useful. That is the whole reason this module exists rather than a list in
session state.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import settings

# Turns handed to the rewriter. Enough for a thread of follow-ups, short enough
# that the prompt stays cheap.
CONTEXT_TURNS = 6

RESOLVE_PROMPT = """Rewrite the latest question so it stands on its own.

Recent conversation:
{history}

Latest question: {question}

If the latest question depends on what came before -- "what about without
dairy?", "and the second one?", "can I use butter instead?" -- rewrite it into a
full question that carries its own subject. If it already stands alone, repeat it
unchanged. Reply with the question and nothing else."""

FOLLOW_UP = re.compile(
    r"^\s*(what about|how about|and |but |ok |okay |also |instead|"
    r"can i|could i|is there|any other|the second|the first|that one|"
    r"without|with |make it|same |show me more|more\b)",
    re.IGNORECASE)


@dataclass
class Turn:
    role: str
    content: str
    at: float = field(default_factory=time.time)
    meta: str = ""
    resolved: str = ""          # what was actually sent to the retriever

    @property
    def when(self) -> str:
        return time.strftime("%H:%M", time.localtime(self.at))


@dataclass
class Conversation:
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New conversation"
    started: float = field(default_factory=time.time)
    pipeline: str = ""
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str, meta: str = "",
            resolved: str = "") -> Turn:
        turn = Turn(role=role, content=content, meta=meta, resolved=resolved)
        self.turns.append(turn)
        if role == "user" and self.title == "New conversation":
            self.title = content[:58]
        return turn

    def recent(self, count: int = CONTEXT_TURNS) -> list[Turn]:
        return self.turns[-count:]

    def transcript(self, count: int = CONTEXT_TURNS) -> str:
        return "\n".join(
            f"{'You' if t.role == 'user' else 'Assistant'}: {t.content[:300]}"
            for t in self.recent(count))


# ------------------------------------------------------------------ storage
def directory() -> Path:
    path = settings.data_dir / "chat"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(conversation: Conversation) -> Path:
    path = directory() / f"{conversation.conversation_id}.json"
    path.write_text(json.dumps(asdict(conversation), indent=2,
                               ensure_ascii=False), encoding="utf-8")
    return path


def load(conversation_id: str) -> Conversation | None:
    path = directory() / f"{conversation_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    turns = [Turn(**t) for t in data.pop("turns", [])]
    return Conversation(**data, turns=turns)


def listing() -> list[dict[str, Any]]:
    """Every saved conversation, newest first, for the sidebar."""
    out: list[dict[str, Any]] = []
    for path in directory().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out.append({"conversation_id": data.get("conversation_id", path.stem),
                    "title": data.get("title", path.stem),
                    "started": data.get("started", 0),
                    "turns": len(data.get("turns", []))})
    return sorted(out, key=lambda c: -c["started"])


def delete(conversation_id: str) -> None:
    (directory() / f"{conversation_id}.json").unlink(missing_ok=True)


# --------------------------------------------------------------- resolution
def has_own_subject(question: str) -> bool:
    """Does the question name something concrete to retrieve on?

    Uses the ingredient vocabulary rather than a word count. "Which recipes use
    coconut?" is four words and needs no rewriting; "and the second one?" is
    four words and is meaningless alone. Length cannot tell them apart, and a
    named ingredient can.
    """
    try:
        from .vocab import ingredients as vocab

        # Membership, not resolve(): resolve falls back to returning the
        # normalised string for anything it has never seen, so it answers "yes"
        # for every word including "second" and "one".
        known = vocab.all_canonical()
        words = re.findall(r"[a-z]+", question.lower())
        for size in (2, 1):        # prefer "lemon juice" over "lemon"
            for start in range(len(words) - size + 1):
                phrase = " ".join(words[start:start + size])
                if len(phrase) > 3 and phrase in known:
                    return True
    except Exception:
        pass
    return False


def looks_like_follow_up(question: str) -> bool:
    """Cheap filter so a standalone question never pays for a rewrite."""
    if FOLLOW_UP.match(question):
        # "without dairy" opens like a follow-up but carries its own subject
        return not has_own_subject(question) or len(question.split()) <= 5
    return len(question.split()) <= 4 and not has_own_subject(question)


def resolve(question: str, conversation: Conversation,
            generator=None) -> tuple[str, bool]:
    """Return (question to retrieve with, whether it was rewritten).

    Falls back to the question as typed on any failure. A rewrite that goes
    wrong should cost nothing more than an unhelpful answer, and never an error.
    """
    if not conversation.turns or not looks_like_follow_up(question):
        return question, False
    try:
        if generator is None:
            from .registry import build, discover

            discover("cognitive_kitchen.rag.generate")
            generator = build("generator", "gpt4o-mini")
        reply = generator.complete(
            RESOLVE_PROMPT.format(history=conversation.transcript(),
                                  question=question),
            max_tokens=80).strip().strip('"')
    except Exception:
        return question, False
    if not reply or len(reply) > 400:
        return question, False
    return reply, reply.lower() != question.strip().lower()