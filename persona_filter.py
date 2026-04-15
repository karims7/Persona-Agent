"""Filter extracted persona cards for quality before downstream scoring (RQ1).

Responsibility: apply the Figure 14 quality-gate prompt to each persona card.
Accepts personas with sufficient socio-demographic and problem description detail.
Saves the filtered list to disk for resumability.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Figure 14 prompt from the paper (verbatim).
_FIGURE_14_TEMPLATE = Template(
    "Your task is to evaluate whether the following socio-demographic and problem "
    "descriptions meet my criteria: "
    "1. Socio-demographic Description: Make sure the description includes the "
    "individual's emotions and the events they are experiencing, while also "
    "providing a clear social background that outlines the person's demographic "
    "context, giving a strong sense of who they are. "
    "2. Problem Description: Ensure that it covers the individual's emotions and "
    "clearly details the event they are facing, including its cause and resulting "
    "consequences. The event should be specific enough to fit into a distinct "
    "classification and be detailed enough to reach a granular level of "
    "categorization."
    "\n\nSocio-demographic Description: $socio_demographic_description"
    "\nProblem Description: $problem_description"
    "\n\nDoes this persona meet both criteria? Reply with exactly 'YES' or 'NO', "
    "followed by a brief reason."
)


class PersonaFilter:
    """Filters persona cards using the Figure 14 quality-gate prompt.

    Args:
        config: The full parsed config.yaml dict.
        llm_client: Shared LLMClient instance.
    """

    def __init__(self, config: dict, llm_client: LLMClient) -> None:
        self._config = config
        self._llm = llm_client
        self._output_path = config["paths"]["intermediate"]["filtered_personas"]

    def filter_personas(
        self,
        personas: list[dict],
        dialogues: list[dict],
    ) -> list[dict]:
        """Apply Figure 14 quality filter to all personas, resuming if output exists.

        Args:
            personas: Extracted persona dicts (from PersonaExtractor).
            dialogues: Original normalized dialogue dicts used to retrieve
                       the seeker's problem description for each persona.

        Returns:
            Filtered list of persona dicts that passed the quality gate.
            Each passing persona gains a "filter_reason" key.
        """
        existing_filtered = self._load_existing()
        existing_ids = {p["dialogue_id"] for p in existing_filtered}

        # Build lookup: dialogue_id -> seeker_problem
        problem_lookup: dict[str, str] = {
            d["dialogue_id"]: d.get("seeker_problem", "") for d in dialogues
        }

        filtered = list(existing_filtered)
        pending = [p for p in personas if p["dialogue_id"] not in existing_ids]
        logger.info(
            "PersonaFilter: %d done, %d pending.", len(existing_ids), len(pending)
        )

        for persona in pending:
            dialogue_id = persona["dialogue_id"]
            problem_description = problem_lookup.get(dialogue_id, "")
            passes, reason = self._evaluate_single(persona, problem_description)
            if passes:
                persona_copy = dict(persona)
                persona_copy["filter_reason"] = reason
                filtered.append(persona_copy)
                logger.debug("PASS [%s]: %s", dialogue_id, reason)
            else:
                logger.debug("FAIL [%s]: %s", dialogue_id, reason)

            self._save(filtered)

        logger.info(
            "PersonaFilter: %d/%d personas passed quality gate.",
            len(filtered),
            len(personas),
        )
        return filtered

    def _evaluate_single(
        self, persona: dict, problem_description: str
    ) -> tuple[bool, str]:
        """Run the Figure 14 prompt for one persona.

        Args:
            persona: Persona dict with socio_demographic_description.
            problem_description: Seeker's problem text from the original dialogue.

        Returns:
            Tuple of (passes_filter: bool, reason: str).
        """
        socio = persona.get("socio_demographic_description", "")
        prompt = _FIGURE_14_TEMPLATE.substitute(
            socio_demographic_description=socio,
            problem_description=problem_description,
        )

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        passes, reason = self._parse_response(raw_response)
        return passes, reason

    def _parse_response(self, raw_response: str) -> tuple[bool, str]:
        """Parse YES/NO from the LLM filter response.

        Args:
            raw_response: Full LLM response string.

        Returns:
            Tuple of (passes: bool, reason: str).
        """
        stripped = raw_response.strip()
        upper = stripped.upper()
        if upper.startswith("YES"):
            reason = stripped[3:].lstrip(": ").strip()
            return True, reason
        elif upper.startswith("NO"):
            reason = stripped[2:].lstrip(": ").strip()
            return False, reason
        else:
            # Ambiguous: default to reject and log.
            logger.warning(
                "Ambiguous filter response (defaulting to FAIL): %s", stripped[:100]
            )
            return False, stripped

    def _load_existing(self) -> list[dict]:
        """Load previously filtered personas from disk if the file exists.

        Returns:
            List of filtered persona dicts, or empty list if no file found.
        """
        if os.path.exists(self._output_path):
            with open(self._output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d existing filtered personas from '%s'.",
                len(data),
                self._output_path,
            )
            return data
        return []

    def _save(self, filtered_personas: list[dict]) -> None:
        """Persist the current filtered personas list to disk.

        Args:
            filtered_personas: Full list of accepted persona dicts.
        """
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_personas, f, ensure_ascii=False, indent=2)
