"""Single gateway for all LLM API calls in the project.

Every LLM call in the entire project must go through this module.
No other file is permitted to import or call any LLM SDK directly.
Supports Google Gemini, Anthropic Claude, and OpenAI.
"""

import os
import time
import logging
from typing import TYPE_CHECKING, Optional

# SDK imports are deferred to _init_clients() so that this module can be
# imported without the optional provider packages installed.  Only the
# clients whose API keys are present will be initialised at runtime.
if TYPE_CHECKING:
    import google.genai as genai  # type: ignore
    import anthropic  # type: ignore
    import openai  # type: ignore

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
        self._openai_model = self._llm_cfg.get("openai_model", "gpt-4o-mini")
        self._gemini_client: Optional[object] = None
        self._anthropic_client: Optional[object] = None
        self._openai_client: Optional[object] = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize SDK clients using environment-variable API keys.

        Each provider SDK is imported lazily here so that this module can be
        imported without all three packages installed.  Only clients for which
        an API key is present are initialised.
        """
        gemini_key = os.environ.get("GEMINI_API_KEY", "")
        if gemini_key:
            try:
                import google.genai as _genai  # type: ignore
                self._gemini_client = _genai.Client(api_key=gemini_key)
            except ImportError:
                logger.warning("google-genai package not installed; Gemini provider unavailable.")
        else:
            logger.warning("GEMINI_API_KEY not set; Gemini provider unavailable.")

        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic as _anthropic  # type: ignore
                self._anthropic_client = _anthropic.Anthropic(api_key=anthropic_key)
            except ImportError:
                logger.warning("anthropic package not installed; Anthropic provider unavailable.")
        else:
            logger.warning("ANTHROPIC_API_KEY not set; Anthropic provider unavailable.")

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            try:
                import openai as _openai  # type: ignore
                self._openai_client = _openai.OpenAI(api_key=openai_key)
            except ImportError:
                logger.warning("openai package not installed; OpenAI provider unavailable.")
        else:
            logger.warning("OPENAI_API_KEY not set; OpenAI provider unavailable.")

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

        if not providers_to_try:
            raise RuntimeError(
                "No LLM providers are initialized. "
                "Set GEMINI_API_KEY and/or ANTHROPIC_API_KEY."
            )

        last_error: Optional[Exception] = None
        for current_provider in providers_to_try:
            for attempt in range(self._max_retries):
                try:
                    if current_provider == "gemini":
                        return self._call_gemini(prompt, system_prompt, temperature)
                    elif current_provider == "anthropic":
                        return self._call_anthropic(prompt, system_prompt, temperature)
                    elif current_provider == "openai":
                        return self._call_openai(prompt, system_prompt, temperature)
                    else:
                        raise ValueError(
                            f"Unknown provider '{current_provider}'. "
                            "Must be 'gemini', 'anthropic', or 'openai'."
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
        """Return initialized providers in order of preference with fallback appended.

        Skips providers whose clients are None (API key not set) to avoid
        burning retry attempts before the real fallback is tried.

        Args:
            preferred_provider: The caller's preferred provider string.

        Returns:
            List of provider names to try, starting with preferred.
        """
        client_ready = {
            "gemini": self._gemini_client is not None,
            "anthropic": self._anthropic_client is not None,
            "openai": self._openai_client is not None,
        }
        candidates = [preferred_provider]
        fallback = self._fallback_provider
        if fallback and fallback != preferred_provider:
            candidates.append(fallback)
        return [p for p in candidates if client_ready.get(p, False)]

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

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        import google.genai as _genai  # type: ignore
        response = self._gemini_client.models.generate_content(
            model=self._gemini_model,
            contents=full_prompt,
            config=_genai.types.GenerateContentConfig(temperature=temperature),
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

    def _call_openai(
        self, prompt: str, system_prompt: str, temperature: float
    ) -> str:
        """Call the OpenAI chat completions API.

        Args:
            prompt: User prompt text.
            system_prompt: System instruction text.
            temperature: Sampling temperature.

        Returns:
            Model response text.
        """
        if self._openai_client is None:
            raise RuntimeError(
                "OpenAI client not initialized. Set OPENAI_API_KEY."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._openai_client.chat.completions.create(
            model=self._openai_model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    @property
    def temperature_extraction(self) -> float:
        """Temperature for deterministic extraction and scoring tasks."""
        return float(self._llm_cfg["temperature_extraction"])

    @property
    def temperature_synthesis(self) -> float:
        """Temperature for dialogue synthesis tasks."""
        return float(self._llm_cfg["temperature_synthesis"])
