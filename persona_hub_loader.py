"""Load and expand PersonaHub personas for RQ2.

Responsibility: load the configured number of PersonaHub personas via DataLoader,
expand each with the Figure 15 prompt (socio-demographic description generation),
and save expanded personas to disk. These personas are then scored with the same
HEXACO/CSI pipeline used for RQ1, and used to generate ESC dialogues for RQ2.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

# Figure 15 prompt from the paper (verbatim).
_FIGURE_15_TEMPLATE = Template(
    "Please generate a socio-demographic description of the individual based on the "
    "provided persona in English. Use 'the person' to refer to the individual in "
    "your description."
    "\n\nPersona: $persona_text"
)


class PersonaHubLoader:
    """Loads PersonaHub personas and expands them with socio-demographic descriptions.

    Args:
        config: The full parsed config.yaml dict.
        llm_client: Shared LLMClient instance.
        data_loader: Shared DataLoader instance.
    """

    def __init__(
        self, config: dict, llm_client: LLMClient, data_loader: DataLoader
    ) -> None:
        self._config = config
        self._llm = llm_client
        self._loader = data_loader
        self._output_path = config["paths"]["intermediate"]["persona_hub_expanded"]
        self._sample_size = int(
            config["datasets"]["persona_hub"].get("sample_size", 1000)
        )

    def load_and_expand(self) -> list[dict]:
        """Load PersonaHub personas and expand each with a socio-demographic description.

        Resumes from disk if a partial output file already exists.

        Returns:
            List of expanded persona dicts, each with:
              - persona_id: str
              - persona_text: str (original PersonaHub description)
              - socio_demographic_description: str (LLM-generated expansion)
              - source: "persona_hub"
        """
        existing = self._load_existing()
        existing_ids = {p["persona_id"] for p in existing}

        raw_personas = self._loader.load_persona_hub(max_samples=self._sample_size)
        pending = [p for p in raw_personas if p["persona_id"] not in existing_ids]

        expanded = list(existing)
        logger.info(
            "PersonaHubLoader: %d done, %d pending.", len(existing_ids), len(pending)
        )

        for persona in pending:
            persona_id = persona["persona_id"]
            persona_text = persona.get("persona_text", "")
            socio_desc = self._expand_single(persona_id, persona_text)
            expanded.append({
                "persona_id": persona_id,
                "persona_text": persona_text,
                "socio_demographic_description": socio_desc,
                "source": "persona_hub",
            })
            self._save(expanded)

        logger.info("PersonaHubLoader: %d total expanded personas.", len(expanded))
        return expanded

    def _expand_single(self, persona_id: str, persona_text: str) -> str:
        """Run the Figure 15 prompt to generate a socio-demographic description.

        Args:
            persona_id: Identifier for the persona.
            persona_text: The raw PersonaHub persona description.

        Returns:
            Generated socio-demographic description string.
        """
        prompt = _FIGURE_15_TEMPLATE.substitute(persona_text=persona_text)

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        return raw_response.strip()

    def _load_existing(self) -> list[dict]:
        """Load previously expanded personas from disk.

        Returns:
            List of expanded persona dicts, or empty list if no file found.
        """
        if os.path.exists(self._output_path):
            with open(self._output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d existing expanded personas from '%s'.",
                len(data),
                self._output_path,
            )
            return data
        return []

    def _save(self, expanded: list[dict]) -> None:
        """Persist the current expanded personas list to disk.

        Args:
            expanded: Full list of expanded persona dicts.
        """
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
