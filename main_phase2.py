"""Entry point for Phase 2: Global Emotional Profile (MELD).

Orchestrates the Phase 2 pipeline for one MELD character:
  1. Load config
  2. Load MELD conversations for character (train split)
  3. Aggregate a Global Emotional Profile (profile_aggregator)
  4. Extract full profile context: character card + triggers + relationships (trigger_extractor)
  5. Encode the profile as a 20-dimensional numerical vector (profile_encoder)
  6. Load MELD conversations for character (test split)
  7. For each test scene: run emotion_predictor twice (with_profile=True/False)
  8. Run gep_evaluator on all predictions
  9. Save all outputs to outputs/profiles/{character}/
  10. Print final evaluation summary

Usage:
  python main_phase2.py
  python main_phase2.py --character Rachel
  python main_phase2.py --character Ross --config config.yaml
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import yaml

from src.llm_client import LLMClient
from src.data_loader import DataLoader
from src.meld_loader import load_character_conversations
from src import profile_aggregator
from src import trigger_extractor
from src import profile_encoder
from src import emotion_predictor
from src import gep_evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path: str = _CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_phase2(
    config: dict,
    llm_client: LLMClient,
    character_name: str,
) -> dict:
    """Run the full Phase 2 pipeline for one MELD character.

    Args:
        config: Parsed config dict.
        llm_client: Shared LLMClient instance.
        character_name: MELD character to profile (e.g. "Ross").

    Returns:
        Dict summarizing all Phase 2 results for this character.
    """
    output_dir = os.path.join(
        config["paths"]["outputs"]["profiles"],
        character_name.lower(),
    )
    os.makedirs(output_dir, exist_ok=True)

    profile_split = config["meld"]["profile_split"]
    eval_split = config["meld"]["eval_split"]
    context_turns = config["meld"].get("prediction_context_turns", 5)

    # Step 1: Load train conversations
    logger.info("=== Step 1: Loading MELD '%s' split for '%s' ===", profile_split, character_name)
    train_conversations = load_character_conversations(character_name, profile_split, config)
    logger.info("Loaded %d train conversations.", len(train_conversations))

    # Step 2: Aggregate Global Emotional Profile
    logger.info("=== Step 2: Aggregating Global Emotional Profile ===")
    gep = profile_aggregator.aggregate(character_name, train_conversations, llm_client, config)
    gep_path = os.path.join(output_dir, "global_profile.json")
    with open(gep_path, "w", encoding="utf-8") as f:
        json.dump(gep, f, ensure_ascii=False, indent=2)
    logger.info("Global profile saved to '%s'.", gep_path)

    # Step 3: Extract full profile context (character card + triggers + relationships)
    logger.info("=== Step 3: Extracting full profile context ===")
    ctx = trigger_extractor.extract_full_profile_context(
        character_name, train_conversations, llm_client, config
    )

    # Merge GEP + context into one combined profile for prediction
    combined_profile = {**gep, **ctx}
    combined_path = os.path.join(output_dir, "full_profile.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined_profile, f, ensure_ascii=False, indent=2)
    logger.info("Full profile saved to '%s'.", combined_path)

    # Step 4: Encode profile vector
    logger.info("=== Step 4: Encoding profile vector ===")
    vector = profile_encoder.encode(combined_profile, config)
    vector_path = os.path.join(output_dir, "profile_vector.npy")
    np.save(vector_path, vector)
    logger.info("Profile vector shape: %s, saved to '%s'.", vector.shape, vector_path)

    # Step 5: Load test conversations
    logger.info("=== Step 5: Loading MELD '%s' split for '%s' ===", eval_split, character_name)
    test_conversations = load_character_conversations(character_name, eval_split, config)
    logger.info("Loaded %d test conversations.", len(test_conversations))

    # Step 6: Predict for each test scene
    logger.info("=== Step 6: Running emotion predictions on test split ===")
    all_predictions = []

    for conv in test_conversations:
        turns = conv.get("turns", [])
        emotion_labels = conv.get("emotion_labels", [])
        emotion_idx = 0

        for i, turn in enumerate(turns):
            if turn["role"] == "Supporter":
                if i >= context_turns and emotion_idx < len(emotion_labels):
                    context = turns[max(0, i - context_turns):i]
                    actual_emotion = emotion_labels[emotion_idx]

                    # Inject actual values into profile for predictor to return
                    profile_copy = dict(combined_profile)
                    profile_copy["_actual_emotion"] = actual_emotion
                    profile_copy["_actual_intensity"] = 3

                    pred_with = emotion_predictor.predict(
                        conversation_context=context,
                        profile=profile_copy,
                        with_profile=True,
                        config=config,
                        llm_client=llm_client,
                    )
                    pred_without = emotion_predictor.predict(
                        conversation_context=context,
                        profile=profile_copy,
                        with_profile=False,
                        config=config,
                        llm_client=llm_client,
                    )
                    all_predictions.extend([pred_with, pred_without])

                if emotion_idx < len(emotion_labels):
                    emotion_idx += 1

    logger.info("Generated %d total predictions (%d scenes × 2 modes).",
                len(all_predictions), len(all_predictions) // 2)

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    # Step 7: Evaluate
    logger.info("=== Step 7: Evaluating predictions ===")
    evaluation = gep_evaluator.evaluate(all_predictions)
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    logger.info("Evaluation saved to '%s'.", eval_path)

    return {
        "character_name": character_name,
        "profile": combined_profile,
        "profile_vector_shape": list(vector.shape),
        "n_train_conversations": len(train_conversations),
        "n_test_conversations": len(test_conversations),
        "n_predictions": len(all_predictions) // 2,
        "evaluation": evaluation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Global Emotional Profile — MELD character analysis."
    )
    parser.add_argument(
        "--character",
        type=str,
        default=None,
        help="Character to profile (default: config meld.default_character).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_CONFIG_PATH,
        help="Path to config.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    character_name = args.character or config["meld"]["default_character"]

    llm_client = LLMClient(config)
    data_loader = DataLoader(config)

    result = run_phase2(config, llm_client, character_name)

    print("\n" + "=" * 60)
    print(f"  Phase 2 Results: {character_name}")
    print("=" * 60)
    summary = {
        "character_name": result["character_name"],
        "dominant_emotion": result["profile"].get("dominant_emotion"),
        "n_train_conversations": result["n_train_conversations"],
        "n_test_conversations": result["n_test_conversations"],
        "profile_vector_shape": result["profile_vector_shape"],
        "n_predictions": result["n_predictions"],
        "evaluation": result["evaluation"],
    }
    print(json.dumps(summary, indent=2))
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
