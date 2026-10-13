"""Generators the web view lets a reader choose. Retrieval, reranking and the verifier stay the same; only the
model that writes the answer changes. Gold-set numbers come from evals/generators.py (runs/gen-<slug>/, measured
with the fine-tuned retrieval); blind-set numbers from evals/harness.py, run once per configuration
(runs/blind-final, runs/blind-sonnet-5-5).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .llm import LLM


@dataclass(frozen=True)
class Model:
    slug: str
    label: str
    model_id: str            # Bedrock model or inference-profile id
    region: str              # Bedrock endpoint the request goes to
    maker: str
    open_weights: bool
    where: str               # where the request is processed, in words
    blurb: str
    gold: float | None = None    # judged correct, 96-question gold set (development)
    blind: float | None = None   # judged correct, 30-question blind set, run once

    def client(self) -> LLM:
        return LLM(f"bedrock:{self.model_id}", region=self.region)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("model_id")
        return d


DEFAULT = "claude-sonnet-5-5"
MODELS: dict[str, Model] = {m.slug: m for m in [
    Model("claude-sonnet-5-5", "Claude Sonnet 5.5", "global.anthropic.claude-sonnet-5-5", "ap-south-1", "Anthropic",
          False, "global routing", "Best balance of accuracy and speed", gold=0.917, blind=0.967),
    Model("claude-opus-5-5", "Claude Opus 5.5", "global.anthropic.claude-opus-5-5", "ap-south-1", "Anthropic",
          False, "global routing", "Most capable; about twice as slow", gold=0.927),
    Model("claude-haiku-5-5", "Claude Haiku 5.5", "global.anthropic.claude-haiku-5-5", "ap-south-1", "Anthropic",
          False, "global routing", "Fastest Claude", gold=0.891),
    Model("deepseek-v3", "DeepSeek V3", "deepseek.v3-v1:0", "ap-south-1", "DeepSeek", True, "Mumbai",
          "Open weights", gold=0.880),
    Model("qwen3-235b", "Qwen3 235B", "qwen.qwen3-235b-a22b-2507-v1:0", "ap-south-1", "Alibaba", True, "Mumbai",
          "Open weights, strong multilingual", gold=0.875),
    Model("gpt-oss-120b", "gpt-oss-120b", "openai.gpt-oss-120b-1:0", "ap-south-1", "OpenAI", True, "Mumbai",
          "Open weights; the configuration submitted first", gold=0.849, blind=0.900),
    Model("gpt-oss-20b", "gpt-oss-20b", "openai.gpt-oss-20b-1:0", "ap-south-1", "OpenAI", True, "Mumbai",
          "Open weights, small enough to self-host on one GPU", gold=0.839),
]}
