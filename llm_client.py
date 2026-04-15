"""Single gateway for all LLM API calls in the project.

Every LLM call in the entire project must go through this module.
No other file is permitted to import or call any LLM SDK directly.
Supports Google Gemini (primary) and Anthropic Claude (fallback).
"""

import os
import time
import logging
from typing import Optional

import google.generativeai as genai
import anthropic

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for Gemini and Anthropic LLM providers.

    Args:
        config: The full parsed config.yaml dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._llm_cfg = config["llm"]
        self._primary_provider = self._llm_cfg["primary_provider"]
        self._fallback_provider = self._llm_cfg["fallback_provider"]
        self._max_retries = int(self._llm_cfg.get("max_retries", 3))
        self._retry_delay = float(self._llm_cfg.get("retry_delay_seconds", 2.0))
        self._gemini_model = self._llm_cfg["gemini_model"]
        self._anthropic_model = self._llm_cfg["anthropic_model"]
        self._gemini_client: Optional[genai.GenerativeModel] = None
        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize SDK clients using environment-variable API keys."""
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini_client = genai.GenerativeModel(self._gemini_model)
        else:
            logger.warning("GEMINI_API_KEY not set; Gemini provider unavailable.")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            self._anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
        else:
            logger.warning("ANTHROPIC_API_KEY not set; Anthropic provider unavailable.")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        provider: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate a text response from the specified LLM provider.

        Args:
            prompt: The user-facing prompt text.
            system_prompt: Optional system/instruction prefix.
            provider: Which provider to use ("gemini" or "anthropic").
                      Defaults to the primary provider from config.
            temperature: Sampling temperature. Use 0.0 for deterministic
                         extraction/scoring tasks; 0.7 for synthesis.

        Returns:
            The model's response as a plain string.

        Raises:
            RuntimeError: If all retry attempts fail on all available providers.
        """
        chosen_provider = provider or self._primary_provider
        providers_to_try = self._build_provider_order(chosen_provider)

        last_error: Optional[Exception] = None
        for current_provider in providers_to_try:
            for attempt in range(self._max_retries):
                try:
                    if current_provider == "gemini":
                        return self._call_gemini(prompt, system_prompt, temperature)
                    elif current_provider == "anthropic":
                        return self._call_anthropic(prompt, system_prompt, temperature)
                    else:
                        raise ValueError(
                            f"Unknown provider '{current_provider}'. "
                            "Must be 'gemini' or 'anthropic'."
                        )
                except (ValueError, NotImplementedError) as exc:
                    raise exc
                except Exception as exc:
                    last_error = exc
                    wait_time = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        "Provider '%s' attempt %d/%d failed: %s. Retrying in %.1fs.",
                        current_provider,
                        attempt + 1,
                        self._max_retries,
                        exc,
                        wait_time,
                    )
                    if attempt < self._max_retries - 1:
                        time.sleep(wait_time)

            logger.error(
                "Provider '%s' exhausted %d retries. Trying fallback.",
                current_provider,
                self._max_retries,
            )

        raise RuntimeError(
            f"All providers exhausted after retries. Last error: {last_error}"
        )

    def _build_provider_order(self, preferred_provider: str) -> list[str]:
        """Return providers in order of preference with fallback appended.

        Args:
            preferred_provider: The caller's preferred provider string.

        Returns:
            List of provider names to try, starting with preferred.
        """
        order = [preferred_provider]
        fallback = self._fallback_provider
        if fallback and fallback != preferred_provider:
            order.append(fallback)
        return order

    def _call_gemini(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call the Gemini API.

        Args:
            prompt: User prompt text.
            system_prompt: System instruction text.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        if self._gemini_client is None:
            raise RuntimeError("Gemini client not initialized. Set GEMINI_API_KEY.")

        generation_config = genai.types.GenerationConfig(temperature=temperature)

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        response = self._gemini_client.generate_content(
            full_prompt,
            generation_config=generation_config,
        )
        return response.text.strip()

    def _call_anthropic(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call the Anthropic Claude API.

        Args:
            prompt: User prompt text.
            system_prompt: System instruction text.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        if self._anthropic_client is None:
            raise RuntimeError(
                "Anthropic client not initialized. Set ANTHROPIC_API_KEY."
            )

        kwargs: dict = {
            "model": self._anthropic_model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        message = self._anthropic_client.messages.create(**kwargs)
        return message.content[0].text.strip()

    @property
    def temperature_extraction(self) -> float:
        """Temperature for deterministic extraction and scoring tasks."""
        return float(self._llm_cfg["temperature_extraction"])

    @property
    def temperature_synthesis(self) -> float:
        """Temperature for dialogue synthesis tasks."""
        return float(self._llm_cfg["temperature_synthesis"])
