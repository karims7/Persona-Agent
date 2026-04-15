"""Compute Pearson correlations between HEXACO and CSI dimension scores (RQ1 and RQ2).

Responsibility: given parallel HEXACO and CSI score lists (both keyed by persona_id),
compute the full 6x6 Pearson correlation matrix between all dimension pairs.
Report which CSI dimension correlates most strongly with each HEXACO dimension.
No LLM calls. No data loading. Pure statistical analysis. Saves results to disk.
"""

import json
import logging
import os
from typing import Optional

from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """Computes and reports HEXACO-CSI Pearson correlation matrices.

    Args:
        config: The full parsed config.yaml dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._output_path = config["paths"]["intermediate"]["rq1_correlations"]
        self._hexaco_dims = [d["name"] for d in config["hexaco"]["dimensions"]]
        self._csi_dims = [d["name"] for d in config["csi"]["dimensions"]]
        self._expected_pairs = {
            d["name"]: d["csi_pair"] for d in config["hexaco"]["dimensions"]
        }

    def analyze(
        self,
        hexaco_scores: list[dict],
        csi_scores: list[dict],
        output_path: Optional[str] = None,
    ) -> dict:
        """Compute the full HEXACO x CSI Pearson correlation matrix.

        Args:
            hexaco_scores: List of score dicts from InventoryScorer.score_hexaco.
                Each dict has "persona_id" and "dimension_scores".
            csi_scores: List of score dicts from InventoryScorer.score_csi.
                Each dict has "persona_id" and "dimension_scores".
            output_path: Override the default output path from config. If None,
                uses config path.

        Returns:
            Dict with:
              - correlation_matrix: dict of {hexaco_dim: {csi_dim: r_value}}
              - p_value_matrix: dict of {hexaco_dim: {csi_dim: p_value}}
              - strongest_csi_per_hexaco: dict of {hexaco_dim: {"csi_dim": str, "r": float}}
              - expected_pair_ranks: dict of {hexaco_dim: int rank of expected CSI pair}
              - n_personas: int
        """
        save_path = output_path or self._output_path

        # Align by persona_id (inner join — only personas present in both).
        hexaco_lookup = {s["persona_id"]: s["dimension_scores"] for s in hexaco_scores}
        csi_lookup = {s["persona_id"]: s["dimension_scores"] for s in csi_scores}
        shared_ids = sorted(set(hexaco_lookup) & set(csi_lookup))

        if len(shared_ids) < 2:
            raise ValueError(
                f"Need at least 2 personas with both HEXACO and CSI scores to compute "
                f"correlations. Got {len(shared_ids)}."
            )

        logger.info(
            "CorrelationAnalyzer: computing correlations for %d personas.", len(shared_ids)
        )

        # Build score vectors for each dimension.
        hexaco_vectors = self._build_vectors(
            shared_ids, hexaco_lookup, self._hexaco_dims
        )
        csi_vectors = self._build_vectors(shared_ids, csi_lookup, self._csi_dims)

        # Compute full correlation matrix.
        correlation_matrix: dict[str, dict[str, float]] = {}
        p_value_matrix: dict[str, dict[str, float]] = {}

        for h_dim in self._hexaco_dims:
            correlation_matrix[h_dim] = {}
            p_value_matrix[h_dim] = {}
            h_vec = np.array(hexaco_vectors[h_dim])
            for c_dim in self._csi_dims:
                c_vec = np.array(csi_vectors[c_dim])
                r_value, p_value = stats.pearsonr(h_vec, c_vec)
                correlation_matrix[h_dim][c_dim] = round(float(r_value), 4)
                p_value_matrix[h_dim][c_dim] = round(float(p_value), 6)

        # Find strongest CSI dimension for each HEXACO dimension.
        strongest_csi_per_hexaco: dict[str, dict] = {}
        for h_dim in self._hexaco_dims:
            row = correlation_matrix[h_dim]
            best_csi = max(row, key=lambda c: abs(row[c]))
            strongest_csi_per_hexaco[h_dim] = {
                "csi_dim": best_csi,
                "r": row[best_csi],
            }

        # Rank the expected CSI pair for each HEXACO dimension.
        expected_pair_ranks: dict[str, int] = {}
        for h_dim in self._hexaco_dims:
            expected_csi = self._expected_pairs.get(h_dim, "")
            row = correlation_matrix[h_dim]
            sorted_csi = sorted(self._csi_dims, key=lambda c: abs(row[c]), reverse=True)
            rank = sorted_csi.index(expected_csi) + 1 if expected_csi in sorted_csi else -1
            expected_pair_ranks[h_dim] = rank

        result = {
            "correlation_matrix": correlation_matrix,
            "p_value_matrix": p_value_matrix,
            "strongest_csi_per_hexaco": strongest_csi_per_hexaco,
            "expected_pairs": self._expected_pairs,
            "expected_pair_ranks": expected_pair_ranks,
            "n_personas": len(shared_ids),
        }

        self._save(result, save_path)
        self._log_summary(result)
        return result

    def _build_vectors(
        self,
        shared_ids: list[str],
        score_lookup: dict[str, dict],
        dimensions: list[str],
    ) -> dict[str, list[float]]:
        """Build parallel score vectors for each dimension across all personas.

        Args:
            shared_ids: Ordered list of persona IDs present in both inventories.
            score_lookup: Mapping from persona_id to dimension_scores dict.
            dimensions: List of dimension names to extract.

        Returns:
            Dict mapping dimension name -> list of float scores (one per persona).
        """
        vectors: dict[str, list[float]] = {dim: [] for dim in dimensions}
        for pid in shared_ids:
            dim_scores = score_lookup[pid]
            for dim in dimensions:
                vectors[dim].append(float(dim_scores.get(dim, 3.0)))
        return vectors

    def _log_summary(self, result: dict) -> None:
        """Log a human-readable correlation summary.

        Args:
            result: The full correlation result dict.
        """
        logger.info("=== RQ1 Correlation Summary (n=%d) ===", result["n_personas"])
        for h_dim, strongest in result["strongest_csi_per_hexaco"].items():
            expected = result["expected_pairs"].get(h_dim, "?")
            rank = result["expected_pair_ranks"].get(h_dim, -1)
            match = "MATCH" if strongest["csi_dim"] == expected else "MISMATCH"
            logger.info(
                "HEXACO[%s] -> strongest CSI[%s] r=%.4f | expected[%s] rank=%d [%s]",
                h_dim,
                strongest["csi_dim"],
                strongest["r"],
                expected,
                rank,
                match,
            )

    def _save(self, result: dict, output_path: str) -> None:
        """Save the correlation result dict to disk as JSON.

        Args:
            result: Full correlation result dict.
            output_path: Target file path.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Correlation results saved to '%s'.", output_path)
