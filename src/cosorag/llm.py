"""One chat interface over two transports: AWS Bedrock (Converse) and any OpenAI-compatible endpoint.

Provider is chosen by env so the same code runs on Bedrock, Ollama, Groq, vLLM or Gemini's OpenAI endpoint:
  COSORAG_LLM=bedrock:<model-id>      e.g. bedrock:openai.gpt-oss-120b-1:0  (region from AWS_REGION)
  COSORAG_LLM=openai:<model>          uses OPENAI_BASE_URL / OPENAI_API_KEY
Responses are cached on disk keyed by (model, messages, params), so eval reruns are free and deterministic.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE_DIR, load_env

load_env()


@dataclass
class LLMReply:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False


def _cache_path(key: dict) -> Path:
    h = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]
    return CACHE_DIR / "llm" / f"{h}.json"


_THINK = re.compile(r"<reasoning>.*?</reasoning>|<think>.*?</think>", re.S)


class LLM:
    def __init__(self, spec: str | None = None, region: str | None = None):
        spec = spec or os.environ.get("COSORAG_LLM", "bedrock:openai.gpt-oss-120b-1:0")
        self.provider, self.model = spec.split(":", 1)
        # gpt-oss / Qwen run in ap-south-1 (Mumbai) by default; "eu." Claude profiles need an EU region
        self.region = region or os.environ.get("COSORAG_LLM_REGION") or ("eu-north-1" if self.model.startswith("eu.") else "ap-south-1")
        self._client = None

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _bedrock(self):
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region,
                config=Config(retries={"max_attempts": 8, "mode": "adaptive"}, read_timeout=180),
            )
        return self._client

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "none"), max_retries=6)
        return self._client

    def chat(self, system: str, user: str | list, *, max_tokens: int = 1200, temperature: float = 0.0,
             images: list[bytes] | None = None, use_cache: bool = True) -> LLMReply:
        key = {"m": self.name, "s": system, "u": user, "t": temperature, "mt": max_tokens,
               "img": [hashlib.sha256(b).hexdigest() for b in images or []]}
        path = _cache_path(key)
        if use_cache and path.exists():
            d = json.loads(path.read_text(encoding="utf-8"))
            return LLMReply(**{**d, "cached": True})
        t0 = time.time()
        if self.provider == "bedrock":
            content = [{"text": user}]
            for b in images or []:
                fmt = "jpeg" if b[:3] == b"\xff\xd8\xff" else "png"
                content.append({"image": {"format": fmt, "source": {"bytes": b}}})
            params = {"maxTokens": max_tokens}
            if not re.search(r"anthropic\.claude-(sonnet|opus|haiku)-5", self.model):   # Claude 5 takes no temperature
                params["temperature"] = temperature
            r = self._bedrock().converse(
                modelId=self.model, system=[{"text": system}],
                messages=[{"role": "user", "content": content}], inferenceConfig=params,
            )
            parts = r["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts)
            usage = r.get("usage", {})
            reply = LLMReply(text, self.name, usage.get("inputTokens", 0), usage.get("outputTokens", 0))
            truncated = r.get("stopReason") == "max_tokens"
        else:
            r = self._openai().chat.completions.create(
                model=self.model, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            u = r.usage
            truncated = r.choices[0].finish_reason == "length"
            reply = LLMReply(r.choices[0].message.content or "", self.name,
                             getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        reply.text = _THINK.sub("", reply.text).strip()
        reply.seconds = round(time.time() - t0, 2)
        if truncated or not reply.text:   # never cache a cut-off or empty reply
            return reply
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: v for k, v in reply.__dict__.items() if k != "cached"}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except OSError:   # read-only or full disk: the answer is still good, it just is not cached
            pass
        return reply


def llm_configured() -> bool:
    """True if the configured provider has credentials; lets the CLI fall back to retrieval-only cleanly."""
    spec = os.environ.get("COSORAG_LLM", "bedrock:")
    if spec.startswith("openai:"):
        return bool(os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_KEY"))
    return bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("AWS_PROFILE"))


def parse_json(text: str):
    """Pull the first JSON object/array out of a model reply (tolerates code fences and prose)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = text.find("{")  # prefer an object: "[1]"-style citations before the JSON must not parse as a list
    if start == -1:
        start = text.find("[")
    if start == -1:
        raise ValueError(f"no JSON in reply: {text[:200]}")
    return json.JSONDecoder().raw_decode(text[start:])[0]
