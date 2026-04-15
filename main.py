"""Entry point for the 'From Personas to Talks' paper implementation.

Orchestrates three research question pipelines:
  RQ1: HEXACO-CSI correlation from ESConv-extracted personas.
  RQ2: Trait consistency before/after LLM dialogue generation (PersonaHub).
  RQ3: Persona injection effect on emotional support strategy distribution.

Usage:
  python main.py --rq 1
  python main.py --rq 2
  python main.py --rq 3
  python main.py --rq all
"""

import argparse
import json
import logging
import os
import sys

import yaml

from src.llm_client import LLMClient
from src.data_loader import DataLoader
from src.persona_extractor import PersonaExtractor
from src.persona_filter import PersonaFilter
from src.trait_describer import TraitDescriber
from src.inventory_scorer import InventoryScorer
from src.correlation_analyzer import CorrelationAnalyzer
from src.persona_hub_loader import PersonaHubLoader
from src.dialogue_synthesizer import DialogueSynthesizer
from src.strategy_classifier import StrategyClassifier
from src.strategy_analyzer import StrategyAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path: str = _CONFIG_PATH) -> dict:
    """Load and return the config.yaml as a dict.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Parsed config dict.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_rq1(
    config: dict,
    llm_client: LLMClient,
    data_loader: DataLoader,
) -> dict:
    """Run RQ1: HEXACO-CSI correlation from ESConv-extracted personas.

    Pipeline:
      1. Load ESConv dialogues.
      2. Extract supporter persona cards (Figure 11).
      3. Filter personas for quality (Figure 14).
      4. Generate HEXACO and CSI behavioral descriptions (Figure 12).
      5. Score each persona on HEXACO-60 and CSI (Figure 13).
      6. Compute Pearson correlation matrix.

    Args:
        config: Parsed config dict.
        llm_client: Shared LLMClient.
        data_loader: Shared DataLoader.

    Returns:
        Correlation analysis result dict.
    """
    logger.info("=== RQ1: HEXACO-CSI trait correlation ===")

    max_dialogues = config["rq1"].get("max_dialogues")
    dialogues = data_loader.load_esconv(max_samples=max_dialogues)

    extractor = PersonaExtractor(config, llm_client, data_loader)
    raw_personas = extractor.extract_personas(dialogues)

    persona_filter = PersonaFilter(config, llm_client)
    filtered_personas = persona_filter.filter_personas(raw_personas, dialogues)

    describer = TraitDescriber(config, llm_client)
    hexaco_descriptions = describer.describe_hexaco(filtered_personas)
    csi_descriptions = describer.describe_csi(filtered_personas)

    scorer = InventoryScorer(config, llm_client)
    hexaco_scores = scorer.score_hexaco(hexaco_descriptions)
    csi_scores = scorer.score_csi(csi_descriptions)

    analyzer = CorrelationAnalyzer(config)
    result = analyzer.analyze(hexaco_scores, csi_scores)

    logger.info("RQ1 complete. Results saved to '%s'.", config["paths"]["intermediate"]["rq1_correlations"])
    return result


def run_rq2(
    config: dict,
    llm_client: LLMClient,
    data_loader: DataLoader,
) -> dict:
    """Run RQ2: trait consistency before/after dialogue generation (PersonaHub).

    Pipeline:
      1. Load and expand 1000 PersonaHub personas (Figure 15).
      2. Score original personas on HEXACO and CSI.
      3. Load ESConv dialogues as conversation history.
      4. Synthesize follow-up dialogues using persona traits (Figure 16).
      5. Extract supporter personas from generated dialogues (Figure 11).
      6. Score extracted personas on HEXACO and CSI.
      7. Compare original vs. extracted dimension scores.

    Args:
        config: Parsed config dict.
        llm_client: Shared LLMClient.
        data_loader: Shared DataLoader.

    Returns:
        Dict with original and extracted scores and comparison summary.
    """
    logger.info("=== RQ2: Trait consistency before/after dialogue generation ===")

    # Step 1: expand PersonaHub personas.
    hub_loader = PersonaHubLoader(config, llm_client, data_loader)
    expanded_personas = hub_loader.load_and_expand()

    # Step 2: score originals.
    describer = TraitDescriber(config, llm_client)

    hexaco_desc_orig_path = config["paths"]["intermediate"]["rq2_original_hexaco"]
    csi_desc_orig_path = config["paths"]["intermediate"]["rq2_original_csi"]

    hexaco_desc_orig = _describe_with_custom_paths(
        describer, expanded_personas, "hexaco", hexaco_desc_orig_path
    )
    csi_desc_orig = _describe_with_custom_paths(
        describer, expanded_personas, "csi", csi_desc_orig_path
    )

    scorer = InventoryScorer(config, llm_client)
    hexaco_scores_orig = scorer.score_hexaco(hexaco_desc_orig)
    csi_scores_orig = scorer.score_csi(csi_desc_orig)

    # Temporarily swap output paths to avoid overwriting RQ1 scores.
    orig_hexaco_path = config["paths"]["intermediate"]["hexaco_scores"]
    orig_csi_path = config["paths"]["intermediate"]["csi_scores"]
    config["paths"]["intermediate"]["hexaco_scores"] = config["paths"]["intermediate"]["rq2_original_hexaco"].replace(".json", "_scores.json")
    config["paths"]["intermediate"]["csi_scores"] = config["paths"]["intermediate"]["rq2_original_csi"].replace(".json", "_scores.json")

    hexaco_scores_orig = scorer.score_hexaco(hexaco_desc_orig)
    csi_scores_orig = scorer.score_csi(csi_desc_orig)

    config["paths"]["intermediate"]["hexaco_scores"] = orig_hexaco_path
    config["paths"]["intermediate"]["csi_scores"] = orig_csi_path

    # Step 3: load ESConv as history.
    max_dialogues = config["rq2"].get("max_dialogues")
    base_dialogues = data_loader.load_esconv(max_samples=max_dialogues)

    # Step 4: synthesize dialogues.
    synthesizer = DialogueSynthesizer(config, llm_client, data_loader)
    synthesized = synthesizer.synthesize_rq2_dialogues(
        personas=expanded_personas,
        hexaco_scores=hexaco_scores_orig,
        csi_scores=csi_scores_orig,
        base_dialogues=base_dialogues,
    )

    # Step 5: extract personas from generated dialogues.
    # Convert synthesized dialogue strings back to the dialogue dict format.
    synthetic_dialogue_dicts = _synthesized_to_dialogue_dicts(synthesized)
    extractor = PersonaExtractor(config, llm_client, data_loader)
    # Override output path for extracted rq2 personas.
    extractor._output_path = config["paths"]["intermediate"]["rq2_extracted_personas"]
    extracted_personas = extractor.extract_personas(synthetic_dialogue_dicts)

    # Step 6: score extracted personas.
    hexaco_desc_ext = _describe_with_custom_paths(
        describer, extracted_personas, "hexaco",
        config["paths"]["intermediate"]["rq2_extracted_hexaco"]
    )
    csi_desc_ext = _describe_with_custom_paths(
        describer, extracted_personas, "csi",
        config["paths"]["intermediate"]["rq2_extracted_csi"]
    )

    config["paths"]["intermediate"]["hexaco_scores"] = config["paths"]["intermediate"]["rq2_extracted_hexaco"].replace(".json", "_scores.json")
    config["paths"]["intermediate"]["csi_scores"] = config["paths"]["intermediate"]["rq2_extracted_csi"].replace(".json", "_scores.json")

    hexaco_scores_ext = scorer.score_hexaco(hexaco_desc_ext)
    csi_scores_ext = scorer.score_csi(csi_desc_ext)

    config["paths"]["intermediate"]["hexaco_scores"] = orig_hexaco_path
    config["paths"]["intermediate"]["csi_scores"] = orig_csi_path

    # Step 7: compare original vs extracted.
    comparison = _compare_scores(hexaco_scores_orig, hexaco_scores_ext, "HEXACO")
    comparison.update(_compare_scores(csi_scores_orig, csi_scores_ext, "CSI"))

    rq2_output = os.path.join(config["paths"]["outputs"]["scores"], "rq2_comparison.json")
    os.makedirs(os.path.dirname(rq2_output), exist_ok=True)
    with open(rq2_output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    logger.info("RQ2 complete. Comparison saved to '%s'.", rq2_output)
    return comparison


def run_rq3(
    config: dict,
    llm_client: LLMClient,
    data_loader: DataLoader,
) -> dict:
    """Run RQ3: persona injection effect on emotional support strategy distribution.

    Pipeline:
      1. Load ESConv dialogues (with existing extracted+filtered personas).
      2. Score personas on HEXACO and CSI.
      3. Synthesize follow-up dialogues WITH persona traits (Figure 16).
      4. Synthesize follow-up dialogues WITHOUT persona traits (Figure 18).
      5. Classify each supporter turn with an ESConv strategy.
      6. Compare strategy distributions between conditions.

    Args:
        config: Parsed config dict.
        llm_client: Shared LLMClient.
        data_loader: Shared DataLoader.

    Returns:
        Strategy distribution analysis result dict.
    """
    logger.info("=== RQ3: Persona injection effect on strategy distribution ===")

    esconv_sample = config["rq3"].get("esconv_sample", 100)
    dialogues = data_loader.load_esconv(max_samples=esconv_sample)

    # Load filtered personas and scores from RQ1 (or re-run extraction if missing).
    filtered_path = config["paths"]["intermediate"]["filtered_personas"]
    hexaco_scores_path = config["paths"]["intermediate"]["hexaco_scores"]
    csi_scores_path = config["paths"]["intermediate"]["csi_scores"]

    if os.path.exists(filtered_path):
        with open(filtered_path, "r", encoding="utf-8") as f:
            filtered_personas = json.load(f)
    else:
        logger.info("Filtered personas not found. Running RQ1 extraction first.")
        extractor = PersonaExtractor(config, llm_client, data_loader)
        raw_personas = extractor.extract_personas(dialogues)
        persona_filter = PersonaFilter(config, llm_client)
        filtered_personas = persona_filter.filter_personas(raw_personas, dialogues)

    if os.path.exists(hexaco_scores_path) and os.path.exists(csi_scores_path):
        with open(hexaco_scores_path, "r", encoding="utf-8") as f:
            hexaco_scores = json.load(f)
        with open(csi_scores_path, "r", encoding="utf-8") as f:
            csi_scores = json.load(f)
    else:
        logger.info("Scores not found. Running description + scoring.")
        describer = TraitDescriber(config, llm_client)
        hexaco_descriptions = describer.describe_hexaco(filtered_personas)
        csi_descriptions = describer.describe_csi(filtered_personas)
        scorer = InventoryScorer(config, llm_client)
        hexaco_scores = scorer.score_hexaco(hexaco_descriptions)
        csi_scores = scorer.score_csi(csi_descriptions)

    # Build dialogue_id -> persona_id mapping from filtered personas.
    persona_id_by_dialogue_id: dict[str, str] = {
        p["dialogue_id"]: p["dialogue_id"] for p in filtered_personas
    }

    # Limit dialogues to those that have persona scores.
    scored_dialogue_ids = {s["persona_id"] for s in hexaco_scores}
    dialogues_for_rq3 = [
        d for d in dialogues if d["dialogue_id"] in scored_dialogue_ids
    ]
    logger.info("RQ3: using %d dialogues with persona scores.", len(dialogues_for_rq3))

    synthesizer = DialogueSynthesizer(config, llm_client, data_loader)
    with_persona_dialogues = synthesizer.synthesize_rq3_with_persona(
        dialogues=dialogues_for_rq3,
        hexaco_scores=hexaco_scores,
        csi_scores=csi_scores,
        persona_id_by_dialogue_id=persona_id_by_dialogue_id,
    )
    without_persona_dialogues = synthesizer.synthesize_rq3_without_persona(
        dialogues=dialogues_for_rq3,
    )

    classifier = StrategyClassifier(config, llm_client)
    with_classified = classifier.classify_with_persona(with_persona_dialogues)
    without_classified = classifier.classify_without_persona(without_persona_dialogues)

    analyzer_rq3 = StrategyAnalyzer(config)
    result = analyzer_rq3.analyze(with_classified, without_classified)

    logger.info("RQ3 complete. Results saved to '%s'.", config["paths"]["intermediate"]["rq3_analysis"])
    return result


def _describe_with_custom_paths(
    describer: TraitDescriber,
    personas: list[dict],
    inventory: str,
    output_path: str,
) -> list[dict]:
    """Call the appropriate TraitDescriber method with a temporarily overridden output path.

    Args:
        describer: The shared TraitDescriber instance.
        personas: Persona dicts to describe.
        inventory: "hexaco" or "csi".
        output_path: Override path for saving results.

    Returns:
        List of description result dicts.
    """
    if inventory == "hexaco":
        original_path = describer._hexaco_output_path
        describer._hexaco_output_path = output_path
        result = describer.describe_hexaco(personas)
        describer._hexaco_output_path = original_path
    else:
        original_path = describer._csi_output_path
        describer._csi_output_path = output_path
        result = describer.describe_csi(personas)
        describer._csi_output_path = original_path
    return result


def _synthesized_to_dialogue_dicts(synthesized: list[dict]) -> list[dict]:
    """Convert synthesized dialogue records back to normalized dialogue dicts.

    Used so that the PersonaExtractor can process generated dialogues in RQ2.

    Args:
        synthesized: List of dicts from DialogueSynthesizer with
                     "persona_id" and "synthesized_dialogue" fields.

    Returns:
        List of minimal normalized dialogue dicts compatible with PersonaExtractor.
    """
    dialogue_dicts = []
    for item in synthesized:
        persona_id = item.get("persona_id", "")
        raw_text = item.get("synthesized_dialogue", "")
        dialogue_dicts.append({
            "dialogue_id": persona_id,
            "turns": _parse_turns_from_text(raw_text),
            "seeker_problem": "",
            "emotion_type": "",
            "problem_type": "",
            "source": "synthesized",
        })
    return dialogue_dicts


def _parse_turns_from_text(raw_text: str) -> list[dict]:
    """Parse turn dicts from a raw synthesized dialogue string.

    Args:
        raw_text: Raw LLM output containing dialogue turns.

    Returns:
        List of turn dicts with "role" and "text" keys.
    """
    import re
    turns = []
    raw_text = raw_text.strip()

    if raw_text.startswith("["):
        try:
            turns_data = json.loads(raw_text)
            for t in turns_data:
                role_raw = str(t.get("role", t.get("speaker", "Unknown")))
                role = "Seeker" if role_raw.lower() in ("seeker", "user") else "Supporter"
                turns.append({"role": role, "text": str(t.get("text", "")).strip(), "strategy": None})
            return turns
        except (json.JSONDecodeError, TypeError):
            pass

    for line in raw_text.split("\n"):
        match = re.match(r"(?i)^(seeker|supporter|user)\s*:\s*(.+)", line.strip())
        if match:
            role_raw = match.group(1).capitalize()
            role = "Seeker" if role_raw.lower() in ("seeker", "user") else "Supporter"
            turns.append({"role": role, "text": match.group(2).strip(), "strategy": None})

    return turns


def _compare_scores(
    original_scores: list[dict],
    extracted_scores: list[dict],
    label: str,
) -> dict:
    """Compute mean absolute difference between original and extracted dimension scores.

    Args:
        original_scores: Score dicts for original personas.
        extracted_scores: Score dicts for extracted (post-dialogue) personas.
        label: "HEXACO" or "CSI" for result keys.

    Returns:
        Dict with per-dimension mean absolute differences and overall mean.
    """
    orig_lookup = {s["persona_id"]: s["dimension_scores"] for s in original_scores}
    ext_lookup = {s["persona_id"]: s["dimension_scores"] for s in extracted_scores}
    shared_ids = sorted(set(orig_lookup) & set(ext_lookup))

    if not shared_ids:
        return {f"{label}_comparison": {}, f"{label}_n_shared": 0}

    all_dims = list(next(iter(orig_lookup.values())).keys())
    dim_diffs: dict[str, list[float]] = {d: [] for d in all_dims}

    for pid in shared_ids:
        for dim in all_dims:
            orig_val = float(orig_lookup[pid].get(dim, 3.0))
            ext_val = float(ext_lookup[pid].get(dim, 3.0))
            dim_diffs[dim].append(abs(orig_val - ext_val))

    per_dim_mad = {
        dim: round(sum(vals) / len(vals), 4) if vals else 0.0
        for dim, vals in dim_diffs.items()
    }
    overall_mad = round(
        sum(per_dim_mad.values()) / len(per_dim_mad) if per_dim_mad else 0.0, 4
    )

    return {
        f"{label}_per_dimension_mad": per_dim_mad,
        f"{label}_overall_mad": overall_mad,
        f"{label}_n_shared": len(shared_ids),
    }


def print_summary(rq: str, result: dict) -> None:
    """Print a brief results summary to stdout.

    Args:
        rq: Which RQ was run ("1", "2", "3").
        result: The result dict returned by the RQ runner.
    """
    print(f"\n{'=' * 60}")
    print(f"  RQ{rq} Results Summary")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:3000])
    if len(json.dumps(result)) > 3000:
        print("  ... (truncated; see output files for full results)")
    print("=" * 60)


def main() -> None:
    """Parse CLI arguments and run the specified research question pipeline(s)."""
    parser = argparse.ArgumentParser(
        description="From Personas to Talks — paper implementation runner."
    )
    parser.add_argument(
        "--rq",
        type=str,
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which research question to run (1, 2, 3, or all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_CONFIG_PATH,
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    llm_client = LLMClient(config)
    data_loader = DataLoader(config)

    rq_to_run = args.rq
    results: dict = {}

    if rq_to_run in ("1", "all"):
        results["rq1"] = run_rq1(config, llm_client, data_loader)
        print_summary("1", results["rq1"])

    if rq_to_run in ("2", "all"):
        results["rq2"] = run_rq2(config, llm_client, data_loader)
        print_summary("2", results["rq2"])

    if rq_to_run in ("3", "all"):
        results["rq3"] = run_rq3(config, llm_client, data_loader)
        print_summary("3", results["rq3"])

    logger.info("All requested pipelines complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
