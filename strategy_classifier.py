"""Classify supporter turns in synthesized dialogues with ESConv strategies (RQ3).

Responsibility: given synthesized dialogue strings (from DialogueSynthesizer),
parse out each Supporter turn and classify it with one of the 8 ESConv emotional
support strategies using an LLM prompt at temperature=0.
Saves classification results to disk for resumability.
"""

import json
import logging
import os
import re

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT_TEMPLATE = (
    "You are classifying the emotional support strategy used in a supporter's "
    "response in a mental health conversation.\n\n"
    "The 8 possible strategies and their definitions are:\n"
    "{strategy_definitions}\n\n"
    "Classify the following supporter turn with exactly ONE strategy name from the "
    "list above. Reply with only the strategy name, nothing else.\n\n"
    "Supporter turn: {supporter_turn}"
)

_VALID_STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


class StrategyClassifier:
    """Classifies each Supporter turn in synthesized dialogues with an ESConv strategy.

    Args:
        config: The full parsed config.yaml dict.
        llm_client: Shared LLMClient instance.
    """

    def __init__(self, config: dict, llm_client: LLMClient) -> None:
        self._config = config
        self._llm = llm_client
        self._rq3_with_output_path = config["paths"]["intermediate"]["rq3_strategies_with"]
        self._rq3_without_output_path = config["paths"]["intermediate"]["rq3_strategies_without"]
        self._strategy_definitions = self._build_strategy_definitions()
        self._valid_strategies = set(_VALID_STRATEGIES)

    def classify_with_persona(self, synthesized_dialogues: list[dict]) -> list[dict]:
        """Classify all supporter turns in with-persona synthesized dialogues.

        Args:
            synthesized_dialogues: Dicts from DialogueSynthesizer.synthesize_rq3_with_persona.
                Each has "dialogue_id" and "synthesized_dialogue" (raw text).

        Returns:
            List of classification result dicts.
        """
        return self._classify_all(
            synthesized_dialogues=synthesized_dialogues,
            output_path=self._rq3_with_output_path,
            condition_label="with_persona",
        )

    def classify_without_persona(self, synthesized_dialogues: list[dict]) -> list[dict]:
        """Classify all supporter turns in without-persona synthesized dialogues.

        Args:
            synthesized_dialogues: Dicts from DialogueSynthesizer.synthesize_rq3_without_persona.
                Each has "dialogue_id" and "synthesized_dialogue" (raw text).

        Returns:
            List of classification result dicts.
        """
        return self._classify_all(
            synthesized_dialogues=synthesized_dialogues,
            output_path=self._rq3_without_output_path,
            condition_label="without_persona",
        )

    def _classify_all(
        self,
        synthesized_dialogues: list[dict],
        output_path: str,
        condition_label: str,
    ) -> list[dict]:
        """Classify supporter turns for all dialogues, resuming if output exists.

        Args:
            synthesized_dialogues: List of synthesized dialogue dicts.
            output_path: Target output JSON file path.
            condition_label: "with_persona" or "without_persona" for logging.

        Returns:
            Full list of classification result dicts, each with:
              - dialogue_id: str
              - condition: str
              - turn_classifications: list of {"turn_text": str, "strategy": str}
              - strategy_counts: dict mapping strategy name -> int count
        """
        existing = self._load_existing(output_path)
        existing_ids = {r["dialogue_id"] for r in existing}

        results = list(existing)
        pending = [
            d for d in synthesized_dialogues
            if d.get("dialogue_id", "") not in existing_ids
        ]
        logger.info(
            "StrategyClassifier [%s]: %d done, %d pending.",
            condition_label,
            len(existing),
            len(pending),
        )

        for dialogue_dict in pending:
            dialogue_id = dialogue_dict.get("dialogue_id", "")
            raw_text = dialogue_dict.get("synthesized_dialogue", "")
            supporter_turns = self._extract_supporter_turns(raw_text)

            turn_classifications: list[dict] = []
            strategy_counts: dict[str, int] = {s: 0 for s in _VALID_STRATEGIES}

            for turn_text in supporter_turns:
                strategy = self._classify_single_turn(turn_text)
                turn_classifications.append({"turn_text": turn_text, "strategy": strategy})
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            results.append({
                "dialogue_id": dialogue_id,
                "condition": condition_label,
                "turn_classifications": turn_classifications,
                "strategy_counts": strategy_counts,
                "n_supporter_turns": len(supporter_turns),
            })
            self._save(results, output_path)

        return results

    def _classify_single_turn(self, supporter_turn: str) -> str:
        """Classify one supporter turn with an LLM prompt at temperature=0.

        Args:
            supporter_turn: The text of a single supporter turn.

        Returns:
            One of the 8 valid strategy names.
        """
        prompt = _CLASSIFICATION_PROMPT_TEMPLATE.format(
            strategy_definitions=self._strategy_definitions,
            supporter_turn=supporter_turn,
        )

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        return self._parse_strategy(raw_response)

    def _parse_strategy(self, raw_response: str) -> str:
        """Match the LLM response to a valid strategy name.

        Falls back to "Others" if no valid strategy is identified.

        Args:
            raw_response: Raw LLM response string.

        Returns:
            A valid strategy name string.
        """
        stripped = raw_response.strip()

        # Exact match first.
        if stripped in self._valid_strategies:
            return stripped

        # Case-insensitive partial match.
        stripped_lower = stripped.lower()
        for strategy in _VALID_STRATEGIES:
            if strategy.lower() in stripped_lower:
                return strategy

        logger.warning(
            "Could not match strategy from response: '%s'. Defaulting to 'Others'.",
            stripped[:100],
        )
        return "Others"

    def _extract_supporter_turns(self, raw_dialogue_text: str) -> list[str]:
        """Parse supporter turn texts from raw synthesized dialogue string.

        Handles two common formats:
          1. "Supporter: <text>" lines.
          2. JSON array of {"role": "Supporter", "text": "..."} objects.

        Args:
            raw_dialogue_text: Raw string output from the LLM synthesizer.

        Returns:
            List of supporter turn text strings.
        """
        raw_dialogue_text = raw_dialogue_text.strip()

        # Try JSON array format.
        if raw_dialogue_text.startswith("["):
            try:
                turns_data = json.loads(raw_dialogue_text)
                return [
                    str(t.get("text", t.get("content", ""))).strip()
                    for t in turns_data
                    if str(t.get("role", t.get("speaker", ""))).lower() == "supporter"
                    and str(t.get("text", t.get("content", ""))).strip()
                ]
            except (json.JSONDecodeError, TypeError):
                pass

        # Try JSON object with "conversation" or "dialogue" key.
        if raw_dialogue_text.startswith("{"):
            try:
                obj = json.loads(raw_dialogue_text)
                turns_data = obj.get("conversation", obj.get("dialogue", obj.get("turns", [])))
                if isinstance(turns_data, list):
                    return [
                        str(t.get("text", t.get("content", ""))).strip()
                        for t in turns_data
                        if str(t.get("role", t.get("speaker", ""))).lower() == "supporter"
                        and str(t.get("text", t.get("content", ""))).strip()
                    ]
            except (json.JSONDecodeError, TypeError):
                pass

        # Plain text "Supporter: <text>" pattern.
        supporter_turns = []
        for line in raw_dialogue_text.split("\n"):
            match = re.match(r"(?i)^supporter\s*:\s*(.+)", line.strip())
            if match:
                supporter_turns.append(match.group(1).strip())

        return supporter_turns

    def _build_strategy_definitions(self) -> str:
        """Build a numbered list of strategy definitions from config.

        Returns:
            Multi-line string of strategy definitions.
        """
        strategies = self._config["esconv"]["strategies"]
        lines = [
            f"{i + 1}. {s['name']}: {s['definition']}"
            for i, s in enumerate(strategies)
        ]
        return "\n".join(lines)

    def _load_existing(self, output_path: str) -> list[dict]:
        """Load previously computed classifications from disk.

        Args:
            output_path: Path to the JSON output file.

        Returns:
            List of classification result dicts, or empty list if no file found.
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d existing classifications from '%s'.",
                len(data),
                output_path,
            )
            return data
        return []

    def _save(self, results: list[dict], output_path: str) -> None:
        """Persist the current classification results to disk.

        Args:
            results: Full list of classification result dicts.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
