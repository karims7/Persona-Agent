"""Generate HEXACO and CSI behavioral indicator descriptions for persona cards (Figure 12).

Responsibility: given a persona's socio-demographic description and the six
dimension definitions from config, prompt the LLM (Figure 12) to produce one
behavioral sentence per dimension. These sentences are later used by
InventoryScorer (Figure 13) as the description input for each question.
Saves results to disk for resumability.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Figure 12 prompt from the paper (verbatim).
_FIGURE_12_TEMPLATE = Template(
    "Based on this person's socio-demographic description, please write a sentence "
    "for each of the following six indicators that best describes what the person "
    "tend to do in the daily life. Each sentence should be of similar length. In "
    "the generated sentences, refer to the person as 'the person'. Below are the "
    "descriptions of these six indicators: $indicator_descriptions. Please complete "
    "the task based on the socio-demographic description: $socio_demographic_description"
)

_JSON_INSTRUCTION = (
    "\n\nReturn your answer as a JSON object where the key is the indicator name "
    "and the value is the single behavioral sentence. "
    "Use exactly these keys: $dimension_keys. "
    "Do not include any text outside the JSON object."
)


class TraitDescriber:
    """Generates per-dimension behavioral sentences from persona socio-demographic text.

    Args:
        config: The full parsed config.yaml dict.
        llm_client: Shared LLMClient instance.
    """

    def __init__(self, config: dict, llm_client: LLMClient) -> None:
        self._config = config
        self._llm = llm_client
        self._hexaco_output_path = config["paths"]["intermediate"]["hexaco_descriptions"]
        self._csi_output_path = config["paths"]["intermediate"]["csi_descriptions"]
        self._hexaco_dims = config["hexaco"]["dimensions"]
        self._csi_dims = config["csi"]["dimensions"]

    def describe_hexaco(self, personas: list[dict]) -> list[dict]:
        """Generate HEXACO behavioral descriptions for all personas.

        Args:
            personas: List of persona dicts with socio_demographic_description.

        Returns:
            List of dicts, each with:
              - persona_id: str (dialogue_id or persona_id)
              - descriptions: dict mapping HEXACO dimension name -> sentence
        """
        return self._describe_all(
            personas=personas,
            dimensions=self._hexaco_dims,
            output_path=self._hexaco_output_path,
            inventory_name="HEXACO",
        )

    def describe_csi(self, personas: list[dict]) -> list[dict]:
        """Generate CSI behavioral descriptions for all personas.

        Args:
            personas: List of persona dicts with socio_demographic_description.

        Returns:
            List of dicts, each with:
              - persona_id: str
              - descriptions: dict mapping CSI dimension name -> sentence
        """
        return self._describe_all(
            personas=personas,
            dimensions=self._csi_dims,
            output_path=self._csi_output_path,
            inventory_name="CSI",
        )

    def _describe_all(
        self,
        personas: list[dict],
        dimensions: list[dict],
        output_path: str,
        inventory_name: str,
    ) -> list[dict]:
        """Run the Figure 12 prompt for all personas, resuming if output exists.

        Args:
            personas: Persona dicts to process.
            dimensions: List of dimension dicts from config (each has "name" and "description" or
                        "short" key).
            output_path: Path to the JSON output file.
            inventory_name: "HEXACO" or "CSI" for log messages.

        Returns:
            Full list of description result dicts.
        """
        existing = self._load_existing(output_path)
        existing_ids = {r["persona_id"] for r in existing}

        results = list(existing)
        pending = [
            p for p in personas
            if self._get_persona_id(p) not in existing_ids
        ]
        logger.info(
            "TraitDescriber [%s]: %d done, %d pending.",
            inventory_name,
            len(existing_ids),
            len(pending),
        )

        indicator_descriptions = self._format_indicator_descriptions(dimensions)
        dimension_keys = json.dumps([d["name"] for d in dimensions])

        for persona in pending:
            persona_id = self._get_persona_id(persona)
            socio = persona.get("socio_demographic_description", persona.get("persona_text", ""))
            description_map = self._describe_single(
                persona_id=persona_id,
                socio_demographic_description=socio,
                indicator_descriptions=indicator_descriptions,
                dimension_keys=dimension_keys,
                dimensions=dimensions,
            )
            results.append({"persona_id": persona_id, "descriptions": description_map})
            self._save(results, output_path)

        return results

    def _describe_single(
        self,
        persona_id: str,
        socio_demographic_description: str,
        indicator_descriptions: str,
        dimension_keys: str,
        dimensions: list[dict],
    ) -> dict:
        """Run the Figure 12 prompt for one persona.

        Args:
            persona_id: Identifier for the persona.
            socio_demographic_description: The persona's socio-demographic text.
            indicator_descriptions: Pre-formatted string of all indicator descriptions.
            dimension_keys: JSON-encoded list of dimension names for the JSON instruction.
            dimensions: List of dimension config dicts.

        Returns:
            Dict mapping dimension name -> behavioral sentence.
        """
        prompt = _FIGURE_12_TEMPLATE.substitute(
            indicator_descriptions=indicator_descriptions,
            socio_demographic_description=socio_demographic_description,
        )
        json_instr = Template(_JSON_INSTRUCTION).substitute(dimension_keys=dimension_keys)
        prompt = prompt + json_instr

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        return self._parse_response(raw_response, dimensions)

    def _parse_response(self, raw_response: str, dimensions: list[dict]) -> dict:
        """Parse the LLM JSON response into a dimension -> sentence mapping.

        Falls back to empty strings on parse failure.

        Args:
            raw_response: Raw string returned by the LLM.
            dimensions: List of dimension config dicts to provide fallback keys.

        Returns:
            Dict mapping dimension name -> sentence string.
        """
        default = {d["name"]: "" for d in dimensions}

        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        try:
            parsed = json.loads(cleaned)
            return {d["name"]: str(parsed.get(d["name"], "")) for d in dimensions}
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Failed to parse trait description JSON: %s. Response: %s",
                exc,
                raw_response[:200],
            )
            return default

    def _format_indicator_descriptions(self, dimensions: list[dict]) -> str:
        """Build a readable string of dimension name and description pairs.

        Args:
            dimensions: List of dimension dicts from config.

        Returns:
            Multi-line string listing each dimension with its description.
        """
        lines = []
        for dim in dimensions:
            name = dim["name"]
            # Config stores 'description' at the top level of hexaco/csi config,
            # not per-dimension. Per-dimension description comes from the
            # inventory question files, but here we use the dimension name
            # as the indicator label and pull its description from config if present.
            desc = dim.get("description", f"The {name} dimension.")
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _get_persona_id(self, persona: dict) -> str:
        """Extract the canonical identifier from a persona dict.

        Args:
            persona: Persona dict (may have dialogue_id or persona_id).

        Returns:
            String identifier.
        """
        return str(persona.get("dialogue_id", persona.get("persona_id", "")))

    def _load_existing(self, output_path: str) -> list[dict]:
        """Load previously computed descriptions from disk.

        Args:
            output_path: Path to the JSON output file.

        Returns:
            List of description result dicts, or empty list if no file found.
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded %d existing descriptions from '%s'.", len(data), output_path)
            return data
        return []

    def _save(self, results: list[dict], output_path: str) -> None:
        """Persist the current descriptions list to disk.

        Args:
            results: Full list of description result dicts.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
