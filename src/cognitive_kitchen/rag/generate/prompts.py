"""Prompt templates.

The answer prompt is deliberately strict about grounding: the point of the
pipeline is that the model uses the retrieved recipes and says so when they do
not contain the answer.
"""
from __future__ import annotations

REFUSAL = "NOT_IN_CONTEXT"

ANSWER_SYSTEM = (
    "You are a careful cooking assistant. Answer only from the recipes provided. "
    f"If they do not contain the answer, reply exactly {REFUSAL}. "
    "Never invent an ingredient or a quantity."
)

ANSWER_USER = """Recipes:
{context}

Question: {question}

Answer using only the recipes above. List ingredients with their quantities where given."""


def answer_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": ANSWER_USER.format(context=context,
                                                       question=question)},
    ]