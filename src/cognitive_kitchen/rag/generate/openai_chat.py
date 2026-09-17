"""OpenAI chat generator.

Used for two jobs only, both of which need judgement a 1.5B local model could
not supply:

    the vocabulary build   one-off, cached to disk, ~1.5 cents
    the judged metrics     Faithfulness and Cookable in the generation stage

The answering path stays local. Every call records tokens and dollars into the
Ledger, so the Lab can show what the paid parts actually cost.
"""
from __future__ import annotations

import time

from ...config import settings
from ..registry import register
from ..telemetry import Ledger, price_of, traced
from ..types import Answer, Passage
from .prompts import REFUSAL, answer_messages


class OpenAIChat:
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 220,
                 temperature: float = 0.0) -> None:
        self.model_name = model_name or settings.judge_model
        self.name = f"openai:{self.model_name}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._client = None
        self.ledger = Ledger()

    def _ensure(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set; add it to .env to use the "
                    "paid generator")
            from openai import OpenAI

            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def _chat(self, messages: list[dict], max_tokens: int, span: str) -> str:
        client = self._ensure()
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model_name, messages=messages,
            max_tokens=max_tokens, temperature=self.temperature)
        elapsed = time.perf_counter() - started
        usage = response.usage
        n_in = getattr(usage, "prompt_tokens", 0) or 0
        n_out = getattr(usage, "completion_tokens", 0) or 0
        self.ledger.record(span, elapsed, tokens_in=n_in, tokens_out=n_out,
                           model=self.model_name)
        return (response.choices[0].message.content or "").strip()

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for span in self.ledger.spans.values():
            total += price_of(self.model_name, span.tokens_in, span.tokens_out)
        return round(total, 6)

    @traced("generate.complete")
    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        return self._chat([{"role": "user", "content": prompt}],
                          max_tokens or self.max_new_tokens, "complete")

    @traced("generate.answer")
    def answer(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        context = "\n\n---\n\n".join(p.text for p in contexts)
        reply = self._chat(answer_messages(question, context),
                           self.max_new_tokens, "answer")
        return Answer(question=question, text=reply, contexts=list(contexts),
                      refused=REFUSAL in reply, model=self.name,
                      elapsed_s=round(time.perf_counter() - started, 2))


@register("generator", "gpt4o-mini")
def make(model_name: str | None = None, max_new_tokens: int = 220,
         temperature: float = 0.0) -> OpenAIChat:
    return OpenAIChat(model_name, max_new_tokens, temperature)