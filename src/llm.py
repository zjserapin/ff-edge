"""Provider-agnostic LLM client — the one place a model API is touched.

The swap between the Anthropic API (personal/PoC) and AWS Bedrock (the Ally
production pattern) happens in `config.py` and here, never in logic files.
`claims.py` calls `complete()` and gets back a normalized `LLMResponse`; it
neither knows nor cares which provider ran it, which is exactly the property
that makes the Bedrock port a config change instead of a rewrite.

The client is deliberately thin: one synchronous `complete()`. Extraction runs
over a few dozen headlines a day — concurrency would be complexity without a
payoff, and the SDK already retries rate limits and server errors on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import (
    ANTHROPIC_API_KEY,
    AWS_REGION,
    BEDROCK_MODEL_ID,
    LLM_MODEL,
    LLM_PROVIDER,
)


@dataclass
class LLMResponse:
    """Normalized response across providers."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str


class LLMClient:
    """Wraps the Anthropic API or AWS Bedrock behind a single interface.

    Lazily constructs the underlying client on first use so importing this
    module (or anything that imports it) never requires credentials — the
    ledger's non-LLM paths (depth charts, scoring, the app) work with no key.
    """

    def __init__(
        self, provider: str = LLM_PROVIDER, model: str = LLM_MODEL
    ) -> None:
        self.provider = provider
        self.model = model
        self._client: Any = None

    def _init_client(self) -> Any:
        if self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Claim extraction needs it; "
                    "export it in your shell (the rest of the ledger — depth "
                    "charts, scoring, the app — runs without it)."
                )
            import anthropic

            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        if self.provider == "bedrock":
            import boto3

            return boto3.client("bedrock-runtime", region_name=AWS_REGION)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def complete(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Single-turn completion, normalized across providers."""
        if self._client is None:
            self._client = self._init_client()

        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
            text = next((b.text for b in resp.content if b.type == "text"), "")
            return LLMResponse(
                content=text,
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                stop_reason=resp.stop_reason or "",
            )

        import json

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
        )
        resp = self._client.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
        result = json.loads(resp["body"].read())
        text = next(
            (b["text"] for b in result["content"] if b.get("type") == "text"), ""
        )
        return LLMResponse(
            content=text,
            model=result["model"],
            input_tokens=result["usage"]["input_tokens"],
            output_tokens=result["usage"]["output_tokens"],
            stop_reason=result.get("stop_reason", ""),
        )
