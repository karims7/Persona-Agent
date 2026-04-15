"""Synthesize follow-up emotional support conversations (RQ2 and RQ3).

Responsibility: generate follow-up ESC dialogues from ESConv conversation history
in two modes:
  - WITH persona traits (Figure 16): injects HEXACO + CSI scores as context.
  - WITHOUT persona traits (Figure 18): no persona injection.
Uses temperature=0.7 for all synthesis calls (as specified in config).
Saves results to separate output files for resumability.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient
from src.data_loader import DataLoader

logger = logging.getLogger(__name__)

# Figure 16 prompt from the paper (verbatim) — WITH persona traits.
_FIGURE_16_TEMPLATE = Template(
    "Your task is to review the previous conversation between the Seeker and "
    "Supporter, which focuses on the Seeker's mental or emotional challenges. Based "
    "on the traits and concerns expressed in that dialogue, simulate a follow-up "
    "conversation that takes place three days later. The new conversation should "
    "delve deeper into the Seeker's mental health or emotional struggles, either "
    "revisiting the issues discussed earlier or exploring any new challenges that "
    "have arisen since their last interaction. In the simulation, the Supporter's "
    "replies need to utilize of the 8 emotional support strategies, unless the reply "
    "is very simple. The 8 emotional support strategies are: $strategy_definitions. "
    "For historical conversations, the data will be provided in JSON format. "
    "The previous conversation is below: $dialogue_json. "
    "HEXACO Personality Indicators: $hexaco_indicator_description. "
    "Below is the HEXACO scores: $hexaco_scores. "
    "Communication Style Inventory (CSI): $csi_indicator_description. "
    "Below is the CSI scores: $csi_scores"
)

# Figure 18 prompt from the paper (verbatim) — WITHOUT persona traits.
_FIGURE_18_TEMPLATE = Template(
    "Your task is to review the previous conversation between the Seeker and "
    "Supporter, which focuses on the Seeker's mental or emotional challenges. Based "
    "on the traits and concerns expressed in that dialogue, simulate a follow-up "
    "conversation that takes place three days later. The new conversation should "
    "delve deeper into the Seeker's mental health or emotional struggles, either "
    "revisiting the issues discussed earlier or exploring any new challenges that "
    "have arisen since their last interaction. In the simulation, the Supporter's "
    "replies need to utilize of the 8 emotional support strategies, unless the reply "
    "is very simple. The 8 emotional support strategies are: $strategy_definitions. "
    "The previous conversation is below: $dialogue_json"
)


class DialogueSynthesizer:
    """Synthesizes follow-up ESC dialogues with and without persona injection.

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
        self._rq2_output_path = config["paths"]["intermediate"]["rq2_dialogues"]
        self._rq3_with_output_path = config["paths"]["intermediate"]["rq3_dialogues_with_persona"]
        self._rq3_without_output_path = config["paths"]["intermediate"]["rq3_dialogues_without_persona"]
        self._strategy_definitions = self._build_strategy_definitions()
        self._hexaco_description = config["hexaco"]["description"]
        self._csi_description = config["csi"]["description"]

    def synthesize_rq2_dialogues(
        self,
        personas: list[dict],
        hexaco_scores: list[dict],
        csi_scores: list[dict],
        base_dialogues: list[dict],
    ) -> list[dict]:
        """Generate follow-up dialogues for RQ2 using PersonaHub personas.

        Uses Figure 16 (WITH persona) for each persona paired with a base ESConv
        dialogue. Resumes if output file exists.

        Args:
            personas: Expanded PersonaHub persona dicts (from PersonaHubLoader).
            hexaco_scores: HEXACO scores for those personas (from InventoryScorer).
            csi_scores: CSI scores for those personas (from InventoryScorer).
            base_dialogues: ESConv dialogues used as conversation history.

        Returns:
            List of synthesized dialogue dicts, each with:
              - persona_id: str
              - dialogue_id: str (base ESConv dialogue used)
              - synthesized_dialogue: str (raw LLM output)
              - condition: "with_persona"
        """
        hexaco_lookup = {s["persona_id"]: s for s in hexaco_scores}
        csi_lookup = {s["persona_id"]: s for s in csi_scores}

        existing = self._load_existing(self._rq2_output_path)
        existing_keys = {(r["persona_id"], r["dialogue_id"]) for r in existing}

        results = list(existing)

        # Pair each persona with a base dialogue (round-robin if fewer dialogues).
        pairs = self._pair_personas_with_dialogues(personas, base_dialogues)
        pending = [
            (persona, dialogue)
            for persona, dialogue in pairs
            if (persona["persona_id"], dialogue["dialogue_id"]) not in existing_keys
        ]

        logger.info(
            "DialogueSynthesizer [RQ2]: %d done, %d pending.",
            len(existing),
            len(pending),
        )

        for persona, dialogue in pending:
            persona_id = persona["persona_id"]
            dialogue_id = dialogue["dialogue_id"]
            hexaco = hexaco_lookup.get(persona_id)
            csi = csi_lookup.get(persona_id)

            if hexaco is None or csi is None:
                logger.warning(
                    "Missing scores for persona_id '%s'. Skipping.", persona_id
                )
                continue

            synthesized = self._synthesize_with_persona(
                dialogue=dialogue,
                hexaco_scores=hexaco["dimension_scores"],
                csi_scores=csi["dimension_scores"],
            )
            results.append({
                "persona_id": persona_id,
                "dialogue_id": dialogue_id,
                "synthesized_dialogue": synthesized,
                "condition": "with_persona",
            })
            self._save(results, self._rq2_output_path)

        return results

    def synthesize_rq3_with_persona(
        self,
        dialogues: list[dict],
        hexaco_scores: list[dict],
        csi_scores: list[dict],
        persona_id_by_dialogue_id: dict[str, str],
    ) -> list[dict]:
        """Generate follow-up dialogues WITH persona injection for RQ3 (Figure 16).

        Args:
            dialogues: ESConv dialogues to use as conversation history.
            hexaco_scores: HEXACO scores keyed by persona_id.
            csi_scores: CSI scores keyed by persona_id.
            persona_id_by_dialogue_id: Mapping from dialogue_id to persona_id.

        Returns:
            List of synthesized dialogue dicts with condition="with_persona".
        """
        return self._synthesize_rq3(
            dialogues=dialogues,
            hexaco_scores=hexaco_scores,
            csi_scores=csi_scores,
            persona_id_by_dialogue_id=persona_id_by_dialogue_id,
            condition="with_persona",
            output_path=self._rq3_with_output_path,
        )

    def synthesize_rq3_without_persona(self, dialogues: list[dict]) -> list[dict]:
        """Generate follow-up dialogues WITHOUT persona injection for RQ3 (Figure 18).

        Args:
            dialogues: ESConv dialogues to use as conversation history.

        Returns:
            List of synthesized dialogue dicts with condition="without_persona".
        """
        existing = self._load_existing(self._rq3_without_output_path)
        existing_ids = {r["dialogue_id"] for r in existing}

        results = list(existing)
        pending = [d for d in dialogues if d["dialogue_id"] not in existing_ids]
        logger.info(
            "DialogueSynthesizer [RQ3 without_persona]: %d done, %d pending.",
            len(existing),
            len(pending),
        )

        for dialogue in pending:
            synthesized = self._synthesize_without_persona(dialogue)
            results.append({
                "dialogue_id": dialogue["dialogue_id"],
                "synthesized_dialogue": synthesized,
                "condition": "without_persona",
            })
            self._save(results, self._rq3_without_output_path)

        return results

    def _synthesize_rq3(
        self,
        dialogues: list[dict],
        hexaco_scores: list[dict],
        csi_scores: list[dict],
        persona_id_by_dialogue_id: dict[str, str],
        condition: str,
        output_path: str,
    ) -> list[dict]:
        """Internal helper for RQ3 with_persona synthesis.

        Args:
            dialogues: ESConv dialogues.
            hexaco_scores: HEXACO scores list.
            csi_scores: CSI scores list.
            persona_id_by_dialogue_id: Maps dialogue_id -> persona_id for score lookup.
            condition: Condition label string.
            output_path: Target output file path.

        Returns:
            Full list of synthesized dialogue dicts.
        """
        hexaco_lookup = {s["persona_id"]: s for s in hexaco_scores}
        csi_lookup = {s["persona_id"]: s for s in csi_scores}

        existing = self._load_existing(output_path)
        existing_ids = {r["dialogue_id"] for r in existing}

        results = list(existing)
        pending = [d for d in dialogues if d["dialogue_id"] not in existing_ids]
        logger.info(
            "DialogueSynthesizer [RQ3 %s]: %d done, %d pending.",
            condition,
            len(existing),
            len(pending),
        )

        for dialogue in pending:
            dialogue_id = dialogue["dialogue_id"]
            persona_id = persona_id_by_dialogue_id.get(dialogue_id)
            hexaco = hexaco_lookup.get(persona_id) if persona_id else None
            csi = csi_lookup.get(persona_id) if persona_id else None

            if hexaco is None or csi is None:
                logger.warning(
                    "Missing scores for dialogue_id '%s'. Skipping.", dialogue_id
                )
                continue

            synthesized = self._synthesize_with_persona(
                dialogue=dialogue,
                hexaco_scores=hexaco["dimension_scores"],
                csi_scores=csi["dimension_scores"],
            )
            results.append({
                "dialogue_id": dialogue_id,
                "persona_id": persona_id,
                "synthesized_dialogue": synthesized,
                "condition": condition,
            })
            self._save(results, output_path)

        return results

    def _synthesize_with_persona(
        self,
        dialogue: dict,
        hexaco_scores: dict,
        csi_scores: dict,
    ) -> str:
        """Run the Figure 16 prompt for one dialogue + persona.

        Args:
            dialogue: Normalized ESConv dialogue dict.
            hexaco_scores: Dict of HEXACO dimension name -> score float.
            csi_scores: Dict of CSI dimension name -> score float.

        Returns:
            Raw LLM-synthesized follow-up dialogue string.
        """
        dialogue_json = self._loader.dialogue_to_json_text(dialogue)
        hexaco_scores_str = json.dumps(hexaco_scores, ensure_ascii=False)
        csi_scores_str = json.dumps(csi_scores, ensure_ascii=False)

        prompt = _FIGURE_16_TEMPLATE.substitute(
            strategy_definitions=self._strategy_definitions,
            dialogue_json=dialogue_json,
            hexaco_indicator_description=self._hexaco_description,
            hexaco_scores=hexaco_scores_str,
            csi_indicator_description=self._csi_description,
            csi_scores=csi_scores_str,
        )

        return self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_synthesis,
        )

    def _synthesize_without_persona(self, dialogue: dict) -> str:
        """Run the Figure 18 prompt for one dialogue (no persona context).

        Args:
            dialogue: Normalized ESConv dialogue dict.

        Returns:
            Raw LLM-synthesized follow-up dialogue string.
        """
        dialogue_json = self._loader.dialogue_to_json_text(dialogue)

        prompt = _FIGURE_18_TEMPLATE.substitute(
            strategy_definitions=self._strategy_definitions,
            dialogue_json=dialogue_json,
        )

        return self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_synthesis,
        )

    def _build_strategy_definitions(self) -> str:
        """Build a human-readable string of all 8 ESConv strategy definitions from config.

        Returns:
            Multi-line string listing each strategy name and definition.
        """
        strategies = self._config["esconv"]["strategies"]
        lines = [
            f"{i + 1}. {s['name']}: {s['definition']}"
            for i, s in enumerate(strategies)
        ]
        return "\n".join(lines)

    def _pair_personas_with_dialogues(
        self, personas: list[dict], dialogues: list[dict]
    ) -> list[tuple[dict, dict]]:
        """Pair each persona with a base dialogue (round-robin on dialogues).

        Args:
            personas: List of persona dicts.
            dialogues: List of base dialogue dicts.

        Returns:
            List of (persona, dialogue) tuples.
        """
        if not dialogues:
            return []
        n_dialogues = len(dialogues)
        return [
            (persona, dialogues[i % n_dialogues])
            for i, persona in enumerate(personas)
        ]

    def _load_existing(self, output_path: str) -> list[dict]:
        """Load previously synthesized dialogues from disk.

        Args:
            output_path: Path to the JSON output file.

        Returns:
            List of synthesized dialogue dicts, or empty list if no file found.
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d existing dialogues from '%s'.", len(data), output_path
            )
            return data
        return []

    def _save(self, results: list[dict], output_path: str) -> None:
        """Persist the current synthesized dialogues list to disk.

        Args:
            results: Full list of synthesized dialogue dicts.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
