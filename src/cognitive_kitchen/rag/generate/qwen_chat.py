"""Local causal LM, default Qwen2.5-1.5B-Instruct.

Loaded on first use, not on import, so the rest of the pipeline runs without
pulling weights. Greedy decoding by default so runs are reproducible.
"""
from __future__ import annotations

import time

from ...config import settings
from ..registry import register
from ..telemetry import Ledger, traced
from ..types import Answer, Passage
from .prompts import REFUSAL, answer_messages


class QwenChat:
    def __init__(self, model_name: str | None = None, max_new_tokens: int = 220,
                 temperature: float = 0.0) -> None:
        self.model_name = model_name or settings.generation_model
        self.name = f"qwen:{self.model_name.split('/')[-1]}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tok = None
        self.ledger = Ledger()          # tokens and latency for this instance

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            # no device_map: it pulls in accelerate and this is CPU-only anyway
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float32)
            self._model.to("cpu")
            self._model.eval()
            torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        return self._model, self._tok

    def _generate(self, text: str, max_tokens: int) -> str:
        import time as _time

        import torch

        model, tok = self._ensure()
        inputs = tok(text, return_tensors="pt")
        n_in = int(inputs["input_ids"].shape[1])
        started = _time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=tok.eos_token_id)
        n_out = int(output[0].shape[0]) - n_in
        # local model: no dollars, but this latency is the real budget
        self.ledger.record("generate", _time.perf_counter() - started,
                           tokens_in=n_in, tokens_out=n_out)
        return tok.decode(output[0][n_in:], skip_special_tokens=True).strip()

    @property
    def tokens_per_second(self) -> float:
        span = self.ledger.spans.get("generate")
        if not span or span.seconds <= 0:
            return 0.0
        return round(span.tokens_out / span.seconds, 2)

    # -- plain completion, used by HyDE and decomposition ------------------
    @traced("generate.complete")
    def complete(self, prompt: str, max_tokens: int | None = None) -> str:
        _, tok = self._ensure()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        return self._generate(text, max_tokens or self.max_new_tokens)

    # -- grounded answer, used by the generation stage ---------------------
    @traced("generate.answer")
    def answer(self, question: str, contexts: list[Passage]) -> Answer:
        started = time.perf_counter()
        _, tok = self._ensure()
        context = "\n\n---\n\n".join(p.text for p in contexts)
        text = tok.apply_chat_template(answer_messages(question, context),
                                       tokenize=False, add_generation_prompt=True)
        reply = self._generate(text, self.max_new_tokens)
        return Answer(question=question, text=reply, contexts=list(contexts),
                      refused=REFUSAL in reply, model=self.name,
                      elapsed_s=round(time.perf_counter() - started, 2))


@register("generator", "qwen")
def make(model_name: str | None = None, max_new_tokens: int = 220,
         temperature: float = 0.0) -> QwenChat:
    return QwenChat(model_name, max_new_tokens, temperature)