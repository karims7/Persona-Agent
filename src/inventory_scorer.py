"""Score personas on HEXACO-60 and CSI inventories using Figure 13 prompt.

Responsibility: for each persona and each dimension, prompt the LLM (Figure 13)
to produce 1-5 Likert scores for all items in that dimension in a single call.
Apply reverse-scoring where required. Compute mean dimension scores.
Saves results to disk for resumability.

API call budget per persona:
  Batched (default): 1 call per dimension  →  6 HEXACO + 6 CSI = 12 calls
  Fallback (on parse error): 1 call per item for that dimension only

Scoring:
  Raw response: JSON array of integers, one per item, 1-5 scale
    (5=strongly agree, 4=agree, 3=neutral, 2=disagree, 1=strongly disagree)
  Reverse-scored items: reversed = 6 - raw
  Dimension score: mean of 10 items (after reversing)
"""

import json
import logging
import os
import re
from string import Template
from typing import Optional

from src.llm_client import LLMClient

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

# Batch prompt: all items for one dimension in a single call.
_FIGURE_13_BATCH_TEMPLATE = Template(
    "You are provided with a personality description and $n_items statements. "
    "For each statement, rate how much the person would agree or disagree on the "
    "basis of the person's personality description. Use this scale: "
    "5=strongly agree, 4=agree, 3=neutral, 2=disagree, 1=strongly disagree. "
    "Return ONLY a JSON array of $n_items integers (each between 1 and 5), "
    "one per statement in order. Do not include any explanation. "
    "The personality description: $description. "
    "Statements:\n$numbered_statements"
)

# Single-item fallback prompt (original Figure 13, verbatim from paper).
_FIGURE_13_SINGLE_TEMPLATE = Template(
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
        """Score one persona across all dimensions using batched dimension calls.

        One LLM call is made per dimension (all items in one prompt). If the
        batch response cannot be parsed, falls back to one call per item for
        that dimension only.

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

            # Attempt batched scoring for the entire dimension.
            batch_results = self._score_dimension_batch(
                description=description,
                items=items,
                dim_name=dim_name,
            )

            dim_item_scores: list[int] = []
            for item, scored_val in zip(items, batch_results):
                item_id = item["id"]
                is_reverse = bool(item.get("reverse", False))
                raw = scored_val  # batch already delivers the raw 1-5 integer
                scored = (6 - raw) if is_reverse else raw
                item_scores[item_id] = {"raw": raw, "scored": scored}
                dim_item_scores.append(scored)

            dimension_scores[dim_name] = (
                round(sum(dim_item_scores) / len(dim_item_scores), 4)
                if dim_item_scores else 0.0
            )

        return {
            "persona_id": persona_id,
            "dimension_scores": dimension_scores,
            "item_scores": item_scores,
        }

    def _score_dimension_batch(
        self,
        description: str,
        items: list[dict],
        dim_name: str,
    ) -> list[int]:
        """Score all items in one dimension via a single batched LLM call.

        Returns one raw integer per item (1-5). On any parse failure, falls
        back to individual per-item calls for this dimension only.

        Args:
            description: Behavioral sentence for the dimension.
            items: List of item dicts with "id", "text", "reverse" keys.
            dim_name: Dimension name used only in log messages.

        Returns:
            List of raw integer scores, one per item, in order.
        """
        n = len(items)
        if n == 0:
            return []

        numbered = "\n".join(
            f"{i + 1}. {item['text']}" for i, item in enumerate(items)
        )
        prompt = _FIGURE_13_BATCH_TEMPLATE.safe_substitute(
            n_items=n,
            description=description,
            numbered_statements=numbered,
        )

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        parsed = self._parse_batch_response(raw_response, expected_count=n)
        if parsed is not None:
            return parsed

        # Fallback: one call per item for this dimension.
        logger.warning(
            "Batch parse failed for dim '%s'; falling back to per-item scoring.",
            dim_name,
        )
        return [
            self._score_single_item(description=description, statement=item["text"])
            for item in items
        ]

    def _parse_batch_response(
        self, raw_response: str, expected_count: int
    ) -> Optional[list[int]]:
        """Parse a JSON array of integers from a batched LLM response.

        Args:
            raw_response: Full LLM response string.
            expected_count: How many integers we expect in the array.

        Returns:
            List of integers clamped to [1, 5], or None if parsing fails.
        """
        cleaned = raw_response.strip()
        # Strip optional markdown code fences.
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        # Find the first JSON array in the response.
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(parsed, list) or len(parsed) != expected_count:
            return None

        try:
            return [max(1, min(5, int(v))) for v in parsed]
        except (TypeError, ValueError):
            return None

    def _score_single_item(self, description: str, statement: str) -> int:
        """Fallback: prompt the LLM (Figure 13) for a single inventory item score.

        Args:
            description: Behavioral sentence for the dimension.
            statement: The inventory question/statement text.

        Returns:
            Integer score in range 1-5.
        """
        prompt = _FIGURE_13_SINGLE_TEMPLATE.safe_substitute(
            description=description,
            statement=statement,
        )

        raw_response = self._llm.generate(
            prompt=prompt,
            temperature=self._llm.temperature_extraction,
        )

        return self._parse_score(raw_response)

    def _parse_score(self, raw_response: str) -> int:
        """Extract a 1-5 integer from a single-item LLM response.

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

        Resolves the path relative to the project root when not absolute.

        Args:
            questions_path: Path to the questions JSON file (relative or absolute).

        Returns:
            Parsed JSON dict.
        """
        if not os.path.isabs(questions_path):
            questions_path = os.path.join(_PROJECT_ROOT, questions_path)
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
