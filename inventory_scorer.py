"""Score personas on HEXACO-60 and CSI inventories using Figure 13 prompt.

Responsibility: for each persona and each inventory question, prompt the LLM
(Figure 13) to produce a 1-5 Likert score. Apply reverse-scoring where required.
Compute mean dimension scores. Saves results to disk for resumability.

Scoring:
  Raw response: integer 1-5 (5=strongly agree, 1=strongly disagree)
  Reverse-scored items: reversed = 6 - raw
  Dimension score: mean of 10 items (after reversing)
"""

import json
import logging
import os
import re
from string import Template

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Figure 13 prompt from the paper (verbatim).
_FIGURE_13_TEMPLATE = Template(
    "You are provided with a statement about the person. Please read it and decide "
    "how much the person will agree or disagree about the statement on the basis of "
    "the person's personality description. Write your answer in the following scale: "
    "5=strongly agree, 4=agree, 3=neutral, 2=disagree, 1=strongly disagree. The "
    "answer of the statement should be a numerical value of 1, 2, 3, 4, 5. Please "
    "answer the statement even if you're not completely sure. "
    "The personality description: $description. "
    "The statement: $statement"
)

_SCORE_PATTERN = re.compile(r"\b([1-5])\b")


class InventoryScorer:
    """Scores persona trait descriptions on HEXACO and CSI inventories.

    Args:
        config: The full parsed config.yaml dict.
        llm_client: Shared LLMClient instance.
    """

    def __init__(self, config: dict, llm_client: LLMClient) -> None:
        self._config = config
        self._llm = llm_client
        self._hexaco_output_path = config["paths"]["intermediate"]["hexaco_scores"]
        self._csi_output_path = config["paths"]["intermediate"]["csi_scores"]
        self._hexaco_questions = self._load_questions(config["paths"]["hexaco_questions"])
        self._csi_questions = self._load_questions(config["paths"]["csi_questions"])

    def score_hexaco(self, trait_descriptions: list[dict]) -> list[dict]:
        """Score all personas on the HEXACO-60 inventory.

        Args:
            trait_descriptions: List of dicts from TraitDescriber.describe_hexaco,
                each with "persona_id" and "descriptions" (dim_name -> sentence).

        Returns:
            List of score dicts, each with:
              - persona_id: str
              - dimension_scores: dict mapping HEXACO dim name -> float (1.0-5.0)
              - item_scores: dict mapping item_id -> {"raw": int, "scored": int}
        """
        return self._score_all(
            trait_descriptions=trait_descriptions,
            questions_data=self._hexaco_questions,
            output_path=self._hexaco_output_path,
            inventory_name="HEXACO",
        )

    def score_csi(self, trait_descriptions: list[dict]) -> list[dict]:
        """Score all personas on the CSI inventory.

        Args:
            trait_descriptions: List of dicts from TraitDescriber.describe_csi,
                each with "persona_id" and "descriptions" (dim_name -> sentence).

        Returns:
            List of score dicts, each with:
              - persona_id: str
              - dimension_scores: dict mapping CSI dim name -> float (1.0-5.0)
              - item_scores: dict mapping item_id -> {"raw": int, "scored": int}
        """
        return self._score_all(
            trait_descriptions=trait_descriptions,
            questions_data=self._csi_questions,
            output_path=self._csi_output_path,
            inventory_name="CSI",
        )

    def _score_all(
        self,
        trait_descriptions: list[dict],
        questions_data: dict,
        output_path: str,
        inventory_name: str,
    ) -> list[dict]:
        """Run scoring for all personas, resuming if output exists.

        Args:
            trait_descriptions: Per-persona behavioral descriptions.
            questions_data: Parsed JSON from hexaco_questions.json or csi_questions.json.
            output_path: Target output file path.
            inventory_name: "HEXACO" or "CSI" for log messages.

        Returns:
            Full list of score dicts.
        """
        existing = self._load_existing(output_path)
        existing_ids = {r["persona_id"] for r in existing}

        results = list(existing)
        pending = [
            td for td in trait_descriptions
            if td["persona_id"] not in existing_ids
        ]
        logger.info(
            "InventoryScorer [%s]: %d done, %d pending.",
            inventory_name,
            len(existing_ids),
            len(pending),
        )

        dimensions_config = questions_data["dimensions"]

        for trait_desc in pending:
            persona_id = trait_desc["persona_id"]
            descriptions = trait_desc.get("descriptions", {})
            score_result = self._score_single_persona(
                persona_id=persona_id,
                descriptions=descriptions,
                dimensions_config=dimensions_config,
            )
            results.append(score_result)
            self._save(results, output_path)

        return results

    def _score_single_persona(
        self,
        persona_id: str,
        descriptions: dict,
        dimensions_config: dict,
    ) -> dict:
        """Score one persona across all dimensions and items.

        Args:
            persona_id: Identifier for the persona.
            descriptions: Dict mapping dimension name -> behavioral sentence.
            dimensions_config: Dict of dimension name -> dimension config with items.

        Returns:
            Score dict with persona_id, dimension_scores, and item_scores.
        """
        item_scores: dict[str, dict] = {}
        dimension_scores: dict[str, float] = {}

        for dim_name, dim_data in dimensions_config.items():
            description = descriptions.get(dim_name, "")
            items = dim_data.get("items", [])
            dim_item_scores: list[int] = []

            for item in items:
                item_id = item["id"]
                statement = item["text"]
                is_reverse = bool(item.get("reverse", False))

                raw_score = self._score_single_item(
                    description=description,
                    statement=statement,
                )
                scored = (6 - raw_score) if is_reverse else raw_score
                item_scores[item_id] = {"raw": raw_score, "scored": scored}
                dim_item_scores.append(scored)

            if dim_item_scores:
                dimension_scores[dim_name] = round(
                    sum(dim_item_scores) / len(dim_item_scores), 4
                )
            else:
                dimension_scores[dim_name] = 0.0

        return {
            "persona_id": persona_id,
            "dimension_scores": dimension_scores,
            "item_scores": item_scores,
        }

    def _score_single_item(self, description: str, statement: str) -> int:
        """Prompt the LLM (Figure 13) for a single inventory item score.

        Args:
            description: Behavioral sentence for the dimension.
            statement: The inventory question/statement text.

        Returns:
            Integer score in range 1-5.
        """
        prompt = _FIGURE_13_TEMPLATE.substitute(
            description=description,
            statement=statement,
        )

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        return self._parse_score(raw_response)

    def _parse_score(self, raw_response: str) -> int:
        """Extract a 1-5 integer from the LLM response.

        Defaults to 3 (neutral) if no valid score is found.

        Args:
            raw_response: Full LLM response string.

        Returns:
            Integer in range 1-5.
        """
        matches = _SCORE_PATTERN.findall(raw_response.strip())
        if matches:
            return int(matches[0])
        logger.warning(
            "Could not extract 1-5 score from response: '%s'. Defaulting to 3.",
            raw_response[:100],
        )
        return 3

    def _load_questions(self, questions_path: str) -> dict:
        """Load inventory questions from a JSON data file.

        Args:
            questions_path: Relative path to the questions JSON file.

        Returns:
            Parsed JSON dict.
        """
        with open(questions_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing(self, output_path: str) -> list[dict]:
        """Load previously computed scores from disk.

        Args:
            output_path: Path to the JSON output file.

        Returns:
            List of score dicts, or empty list if no file found.
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(
                "Loaded %d existing scores from '%s'.", len(data), output_path
            )
            return data
        return []

    def _save(self, results: list[dict], output_path: str) -> None:
        """Persist the current scores list to disk.

        Args:
            results: Full list of score dicts.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
