#!/usr/bin/env python3
"""
FENRIR - Local Model Backend

Drop-in replacement for AITradingAnalyst that routes inference to a local
vLLM or llama.cpp server running an abliterated model (via OBLITERATUS).

Why a local abliterated model?
  - Zero refusal on trading edge cases, rug analysis, pump/dump pattern discussion
  - No per-token API costs at inference scale
  - Sub-100ms latency on local GPU vs 1-5s cloud round-trip
  - Full audit — every prompt/response stays on your hardware

Setup (one-time):
  1. Install OBLITERATUS:
        pip install obliteratus
  2. Abliterate Llama-3.1-8B (takes ~10 min on any modern GPU):
        obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct --method advanced \\
            --output-dir ./models/fenrir-brain
  3. Serve with vLLM (OpenAI-compatible):
        vllm serve ./models/fenrir-brain \\
            --port 8000 \\
            --served-model-name fenrir-brain \\
            --max-model-len 4096
  4. Set in .env:
        AI_LOCAL_MODEL_ENABLED=true
        AI_LOCAL_MODEL_URL=http://localhost:8000/v1/chat/completions
        AI_LOCAL_MODEL_NAME=fenrir-brain

  Or with llama.cpp (server mode):
        ./llama-server -m ./models/fenrir-brain.gguf --port 8000

Usage:
    # brain.py uses this automatically when ai_local_model_enabled=True
    from fenrir.ai.local_backend import LocalAITradingAnalyst

    analyst = LocalAITradingAnalyst(
        base_url="http://localhost:8000/v1/chat/completions",
        model_name="fenrir-brain",
        temperature=0.2,
        timeout_seconds=10,
    )
    await analyst.initialize()
"""

import logging

from fenrir.ai.decision_engine import AITradingAnalyst

logger = logging.getLogger(__name__)


class LocalAITradingAnalyst(AITradingAnalyst):
    """
    AITradingAnalyst routed to a local OpenAI-compatible inference server.

    Inherits the full prompt engineering, response parsing, and performance
    tracking from AITradingAnalyst. Only the transport layer changes:
    - URL → local server instead of OpenRouter
    - API key → "none" (local servers don't authenticate)
    - Model name → whatever you named the served model

    The local server must implement the OpenAI /v1/chat/completions API.
    Both vLLM and llama.cpp server mode do this out of the box.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1/chat/completions",
        model_name: str = "fenrir-brain",
        temperature: float = 0.2,
        timeout_seconds: int = 10,
    ):
        # Call parent with a dummy API key — local servers don't need one
        super().__init__(
            api_key="none",
            model=model_name,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        # Override the URL that _call_llm posts to
        self.OPENROUTER_API = base_url
        self._base_url = base_url
        self._model_name = model_name

        logger.info(
            f"🦙 Local Brain: targeting {base_url} "
            f"(model={model_name}, temp={temperature})"
        )

    async def _call_llm(self, prompt: str) -> str:
        """
        Override to use Authorization: Bearer none (local servers ignore auth).

        The parent _call_llm uses self.OPENROUTER_API which we've already
        overridden in __init__, so most cases this just falls through.
        We override explicitly for clarity and to strip OpenRouter-specific
        headers that some local servers reject.
        """
        import aiohttp

        if not self.session:
            await self.initialize()

        # Local servers use the same OpenAI wire format but don't need auth
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer none",  # Required by spec, value ignored locally
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert memecoin analyst. You provide honest, "
                        "data-driven assessments. You err on the side of caution. "
                        "You respond ONLY with valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 2000,
        }

        try:
            async with self.session.post(
                self._base_url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Local model error: {response.status} - {error_text[:200]}"
                    )
                    return self._get_conservative_default()

                data = await response.json()

                if "choices" not in data or not data["choices"]:
                    logger.error("Local model returned no choices")
                    return self._get_conservative_default()

                content = data["choices"][0]["message"]["content"]
                logger.debug(f"Local model response ({len(content)} chars)")
                return content

        except aiohttp.ClientConnectorError:
            logger.error(
                f"Cannot reach local model at {self._base_url}. "
                "Is vLLM/llama.cpp running? Falling back to conservative default."
            )
            return self._get_conservative_default()
        except Exception as e:
            logger.error(f"Local model call failed: {e}")
            return self._get_conservative_default()

    async def health_check(self) -> tuple[bool, str]:
        """
        Ping the local model server to confirm it's reachable.

        Returns:
            (is_healthy, status_message)
        """
        import aiohttp

        # Try the /health endpoint (vLLM) or models endpoint (llama.cpp)
        health_urls = [
            self._base_url.replace("/chat/completions", "/health"),
            self._base_url.replace("/v1/chat/completions", "/health"),
            self._base_url.replace("/chat/completions", "/models"),
        ]

        for url in health_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                        if resp.status == 200:
                            return (True, f"Local model healthy at {self._base_url}")
            except Exception:
                continue

        return (
            False,
            f"Local model unreachable at {self._base_url}. "
            "Start vLLM: vllm serve ./models/fenrir-brain --port 8000",
        )
