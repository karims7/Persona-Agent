"""Extract supporter persona cards from emotional support dialogues (RQ1 and RQ2).

Responsibility: given a list of normalized dialogues, prompt the LLM (Figure 11)
to extract the supporter's persona card (age, gender, occupation,
socio-demographic description). Saves results to disk for resumability.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

# Figure 11 prompt from the paper (verbatim).
_FIGURE_11_TEMPLATE = Template(
    "You need to complete a persona card based on a dialogue between a Seeker "
    "and a Supporter. Your task is to extract the supporter's persona. The persona "
    "card includes five dimensions: age, gender, occupation, and socio-demographic "
    "description. For age, you can choose from 'teenage, young, middle-aged, old'. "
    "For gender, choose from 'male, female'. If you cannot determine the age or "
    "gender, you can select 'unknown'. If the occupation is not specified, provide "
    "a relevant occupation description or create a suitable occupation. For the "
    "socio-demographic description, generate the description of the Supporter based "
    "on the dialogue and combine with your own speculations. In the description "
    "sections, refer to the Supporter as 'the person'. Please complete the extraction "
    "based on the dialogue given below: $dialogue"
)

_JSON_INSTRUCTION = (
    "\n\nReturn your answer as a JSON object with exactly these keys: "
    '"age", "gender", "occupation", "socio_demographic_description". '
    "Do not include any text outside the JSON object."
)


class PersonaExtractor:
    """Extracts supporter persona cards from ESConv-style dialogues.

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
        self._output_path = config["paths"]["intermediate"]["extracted_personas"]

    def extract_personas(self, dialogues: list[dict]) -> list[dict]:
        """Extract persona cards for all dialogues, resuming if output exists.

        Args:
            dialogues: Normalized ESConv dialogue dicts from DataLoader.

        Returns:
            List of persona dicts, each containing:
              - dialogue_id: str
              - age: str
              - gender: str
              - occupation: str
              - socio_demographic_description: str
              - raw_response: str (full LLM output for debugging)
        """
        existing_personas = self._load_existing()
        existing_ids = {p["dialogue_id"] for p in existing_personas}

        personas = list(existing_personas)
        pending = [d for d in dialogues if d["dialogue_id"] not in existing_ids]
        logger.info(
            "PersonaExtractor: %d done, %d pending.", len(existing_ids), len(pending)
        )

        for dialogue in pending:
            dialogue_id = dialogue["dialogue_id"]
            dialogue_text = self._loader.dialogue_to_text(dialogue)
            persona = self._extract_single(dialogue_id, dialogue_text)
            personas.append(persona)
            self._save(personas)

        return personas

    def _extract_single(self, dialogue_id: str, dialogue_text: str) -> dict:
        """Run the Figure 11 extraction prompt for one dialogue.

        Args:
            dialogue_id: Identifier for the dialogue.
            dialogue_text: Plain-text formatted dialogue string.

        Returns:
            Parsed persona dict with dialogue_id attached.
        """
        prompt = _FIGURE_11_TEMPLATE.substitute(dialogue=dialogue_text)
        prompt = prompt + _JSON_INSTRUCTION

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        persona = self._parse_response(raw_response)
        persona["dialogue_id"] = dialogue_id
        persona["raw_response"] = raw_response
        return persona

    def _parse_response(self, raw_response: str) -> dict:
        """Parse the LLM JSON response into a persona dict.

        Falls back to empty strings if JSON parsing fails.

        Args:
            raw_response: Raw string returned by the LLM.

        Returns:
            Dict with keys: age, gender, occupation, socio_demographic_description.
        """
        default: dict = {
            "age": "unknown",
            "gender": "unknown",
            "occupation": "unknown",
            "socio_demographic_description": "",
        }

        cleaned = raw_response.strip()
        # Strip optional markdown code fences.
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            parsed = json.loads(cleaned)
            return {
                "age": str(parsed.get("age", "unknown")),
                "gender": str(parsed.get("gender", "unknown")),
                "occupation": str(parsed.get("occupation", "unknown")),
                "socio_demographic_description": str(
                    parsed.get("socio_demographic_description", "")
                ),
            }
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse persona JSON: %s. Response: %s", exc, raw_response[:200])
            return default

    def _load_existing(self) -> list[dict]:
        """Load previously extracted personas from disk if the file exists.

        Returns:
            List of persona dicts, or empty list if no file found.
        """
        if os.path.exists(self._output_path):
            with open(self._output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d existing personas from '%s'.", len(data), self._output_path)
            return data
        return []

    def _save(self, personas: list[dict]) -> None:
        """Persist the current personas list to disk.

        Args:
            personas: Full list of extracted persona dicts to write.
        """
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(personas, f, ensure_ascii=False, indent=2)
