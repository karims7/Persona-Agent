"""Analyze emotional support strategy distributions across conditions (RQ3).

Responsibility: given classified strategy results for with-persona and
without-persona conditions, compute aggregate strategy frequency distributions,
run a chi-squared test for distributional difference, and compute per-strategy
proportions and differences. No LLM calls. Saves results to disk.
"""

import json
import logging
import os

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

_ALL_STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


class StrategyAnalyzer:
    """Compares strategy distributions between with-persona and without-persona conditions.

    Args:
        config: The full parsed config.yaml dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._output_path = config["paths"]["intermediate"]["rq3_analysis"]

    def analyze(
        self,
        with_persona_results: list[dict],
        without_persona_results: list[dict],
    ) -> dict:
        """Compute strategy distribution comparison between conditions.

        Args:
            with_persona_results: Classification results from StrategyClassifier
                (condition="with_persona"), each with "strategy_counts".
            without_persona_results: Classification results from StrategyClassifier
                (condition="without_persona"), each with "strategy_counts".

        Returns:
            Dict with:
              - with_persona_counts: dict mapping strategy -> int total count
              - without_persona_counts: dict mapping strategy -> int total count
              - with_persona_proportions: dict mapping strategy -> float proportion
              - without_persona_proportions: dict mapping strategy -> float proportion
              - proportion_differences: dict mapping strategy -> float (with - without)
              - chi2_statistic: float
              - chi2_p_value: float
              - chi2_df: int
              - n_with_dialogues: int
              - n_without_dialogues: int
              - total_turns_with: int
              - total_turns_without: int
        """
        with_counts = self._aggregate_counts(with_persona_results)
        without_counts = self._aggregate_counts(without_persona_results)

        with_proportions = self._to_proportions(with_counts)
        without_proportions = self._to_proportions(without_counts)

        proportion_differences = {
            s: round(with_proportions.get(s, 0.0) - without_proportions.get(s, 0.0), 4)
            for s in _ALL_STRATEGIES
        }

        chi2_stat, chi2_p, chi2_df = self._chi_squared_test(with_counts, without_counts)

        total_with = sum(with_counts.values())
        total_without = sum(without_counts.values())

        result = {
            "with_persona_counts": with_counts,
            "without_persona_counts": without_counts,
            "with_persona_proportions": with_proportions,
            "without_persona_proportions": without_proportions,
            "proportion_differences": proportion_differences,
            "chi2_statistic": round(float(chi2_stat), 4),
            "chi2_p_value": round(float(chi2_p), 6),
            "chi2_df": int(chi2_df),
            "n_with_dialogues": len(with_persona_results),
            "n_without_dialogues": len(without_persona_results),
            "total_turns_with": total_with,
            "total_turns_without": total_without,
        }

        self._save(result)
        self._log_summary(result)
        return result

    def _aggregate_counts(self, classification_results: list[dict]) -> dict[str, int]:
        """Sum strategy counts across all dialogues in a condition.

        Args:
            classification_results: List of classification result dicts from
                StrategyClassifier, each with "strategy_counts" dict.

        Returns:
            Dict mapping strategy name -> total count across all dialogues.
        """
        total: dict[str, int] = {s: 0 for s in _ALL_STRATEGIES}
        for result in classification_results:
            counts = result.get("strategy_counts", {})
            for strategy in _ALL_STRATEGIES:
                total[strategy] += int(counts.get(strategy, 0))
        return total

    def _to_proportions(self, counts: dict[str, int]) -> dict[str, float]:
        """Convert raw counts to proportions (fraction of total turns).

        Args:
            counts: Dict mapping strategy name -> count.

        Returns:
            Dict mapping strategy name -> proportion (0.0 to 1.0).
            All values are 0.0 if total is 0.
        """
        total = sum(counts.values())
        if total == 0:
            return {s: 0.0 for s in _ALL_STRATEGIES}
        return {s: round(counts.get(s, 0) / total, 4) for s in _ALL_STRATEGIES}

    def _chi_squared_test(
        self,
        with_counts: dict[str, int],
        without_counts: dict[str, int],
    ) -> tuple[float, float, int]:
        """Run chi-squared test comparing two strategy count distributions.

        Adds 0.5 Laplace smoothing to avoid zero-cell issues.

        Args:
            with_counts: Strategy counts for with-persona condition.
            without_counts: Strategy counts for without-persona condition.

        Returns:
            Tuple of (chi2_statistic, p_value, degrees_of_freedom).
        """
        with_vec = np.array(
            [with_counts.get(s, 0) + 0.5 for s in _ALL_STRATEGIES], dtype=float
        )
        without_vec = np.array(
            [without_counts.get(s, 0) + 0.5 for s in _ALL_STRATEGIES], dtype=float
        )

        # Normalize without_vec to expected counts matching with_vec total.
        without_expected = without_vec / without_vec.sum() * with_vec.sum()

        chi2_stat, p_value = stats.chisquare(f_obs=with_vec, f_exp=without_expected)
        degrees_of_freedom = len(_ALL_STRATEGIES) - 1
        return float(chi2_stat), float(p_value), degrees_of_freedom

    def _log_summary(self, result: dict) -> None:
        """Log a human-readable strategy distribution summary.

        Args:
            result: Full analysis result dict.
        """
        logger.info("=== RQ3 Strategy Distribution Summary ===")
        logger.info(
            "Dialogues: with_persona=%d, without_persona=%d",
            result["n_with_dialogues"],
            result["n_without_dialogues"],
        )
        logger.info(
            "Total turns: with=%d, without=%d",
            result["total_turns_with"],
            result["total_turns_without"],
        )
        logger.info(
            "Chi-squared: stat=%.4f, p=%.6f, df=%d",
            result["chi2_statistic"],
            result["chi2_p_value"],
            result["chi2_df"],
        )
        logger.info("Per-strategy proportions (with | without | diff):")
        for strategy in _ALL_STRATEGIES:
            w = result["with_persona_proportions"].get(strategy, 0.0)
            wo = result["without_persona_proportions"].get(strategy, 0.0)
            diff = result["proportion_differences"].get(strategy, 0.0)
            logger.info("  %-35s %.4f | %.4f | %+.4f", strategy, w, wo, diff)

    def _save(self, result: dict) -> None:
        """Persist the analysis result to disk.

        Args:
            result: Full analysis result dict.
        """
        os.makedirs(os.path.dirname(self._output_path), exist_ok=True)
        with open(self._output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("RQ3 analysis saved to '%s'.", self._output_path)
