"""Groq (GroqCloud) as a Hermes model provider: pick Groq in ``hermes model`` or pass ``--provider groq``.

What this adds over pointing a custom endpoint at api.groq.com, each checked against the live API:

* Reasoning effort is sent in the form each model accepts. gpt-oss takes only low/medium/high
  (``default`` and ``none`` are rejected with HTTP 400), Qwen 3.8 also takes ``none``, and a model
  without reasoning rejects the field, so it is left out for those.
* The model list holds only models Hermes can run. Groq's ``/models`` also returns speech-to-text,
  text-to-speech and prompt-guard models, and models whose context window is below the 64K
  tokens Hermes requires.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

API_KEY_ENV = "GROQ_API_KEY"
BASE_URL = "https://api.groq.com/openai/v1"

# Hermes' reasoning-effort levels, weakest first.
EFFORT_LADDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")

# The reasoning_effort values Groq accepts, by model-id prefix. Only models whose values were
# checked are listed; any other model gets no reasoning_effort and runs at Groq's own default.
REASONING_EFFORTS = (
    ("openai/gpt-oss-", ("low", "medium", "high")),
    ("qwen/qwen3.8-", ("none", "low", "medium", "high")),
)

# Id markers of the Groq models that cannot run a Hermes conversation: whisper (speech-to-text,
# no chat endpoint), orpheus (text-to-speech, 4K window), prompt-guard (512-token classifier) and
# allam-2-7b (4K window). Hermes refuses to start a model with less than 64K tokens of context.
EXCLUDED_MODEL_MARKERS = ("whisper", "orpheus", "prompt-guard", "allam")


def runs_in_hermes(model_id: Any) -> bool:
    """Whether *model_id* belongs in the model picker."""
    lowered = str(model_id or "").lower()
    return not any(marker in lowered for marker in EXCLUDED_MODEL_MARKERS)


def reasoning_effort_for(model: Any, reasoning_config: Any) -> str | None:
    """The ``reasoning_effort`` value to send for *model*, or None to leave the field out.

    A level the model does not take becomes the nearest weaker one it does, or its weakest level
    when there is none, so a request never costs more reasoning than was asked for. Turning
    reasoning off maps to ``none`` where the model has it and to its weakest level otherwise.
    """
    if not isinstance(reasoning_config, dict):
        return None
    lowered = str(model or "").lower()
    supported = next((levels for prefix, levels in REASONING_EFFORTS if lowered.startswith(prefix)), None)
    if supported is None:
        return None
    floor = next(level for level in supported if level != "none")
    requested = str(reasoning_config.get("effort") or "").strip().lower()
    if reasoning_config.get("enabled") is False or requested == "none":
        return "none" if "none" in supported else floor
    if requested not in EFFORT_LADDER:
        return None
    if requested in supported:
        return requested
    rank = EFFORT_LADDER.index(requested)
    weaker = [level for level in supported if level != "none" and EFFORT_LADDER.index(level) < rank]
    return weaker[-1] if weaker else floor


class GroqProfile(ProviderProfile):
    """Groq's OpenAI-compatible endpoint with per-model reasoning values and a runnable model list."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        effort = reasoning_effort_for(context.get("model"), reasoning_config)
        return {}, ({"reasoning_effort": effort} if effort else {})

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if models is None:
            return None
        return [model for model in models if runs_in_hermes(model)]


groq = GroqProfile(
    name="groq",
    display_name="Groq",
    description="Groq — fast inference for open models (OpenAI-compatible)",
    signup_url="https://console.groq.com/keys",
    env_vars=(API_KEY_ENV,),
    base_url=BASE_URL,
    auth_type="api_key",
    default_aux_model="openai/gpt-oss-20b",
    fallback_models=("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"),
)

register_provider(groq)
