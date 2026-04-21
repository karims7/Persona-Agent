"""Build a GlobalEmotionalProfile for one MELD character from all their conversations.

Responsibility: batch conversations, run one combined LLM call per batch for
persona_card + hexaco_descriptions + csi_descriptions, score each batch using
inventory_scorer.py (12 LLM calls), then average scores across all batches.
No disk writes (caller saves).
"""

import json
import logging
import math
import os
from collections import Counter
from string import Template

from src.llm_client import LLMClient
from src.inventory_scorer import InventoryScorer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

_HEXACO_DIM_NAMES = [
    "Honesty-Humility",
    "Emotionality",
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Openness to Experience",
]

_CSI_DIM_NAMES = [
    "Impression Manipulativeness",
    "Emotionality",
    "Expressiveness",
    "Verbal Aggressiveness",
    "Preciseness",
    "Questioningness",
]

_BATCH_TEMPLATE = Template(
    "You are analyzing the character $character_name from the TV show Friends.\n\n"
    "Below are $n_scenes scenes from their conversations, each utterance tagged "
    "with the speaker's name and emotion:\n\n"
    "$scenes_text\n\n"
    "Based on these scenes, provide ALL of the following in one JSON response:\n\n"
    "1. persona_card: A dict with keys: age (teenage/young/middle-aged/old), "
    "occupation, socio_demographic_description (paragraph describing the person), "
    "problem (main emotional challenge)\n\n"
    "2. hexaco_descriptions: A dict mapping each of these exact dimension names "
    "to one behavioral sentence describing $character_name:\n"
    "$hexaco_keys\n\n"
    "3. csi_descriptions: A dict mapping each of these exact dimension names "
    "to one behavioral sentence describing $character_name's communication style:\n"
    "$csi_keys\n\n"
    "Return a JSON object with exactly these top-level keys: "
    '"persona_card", "hexaco_descriptions", "csi_descriptions". '
    "Do not include any text outside the JSON object."
)


def aggregate(
    character_name: str,
    conversations: list[dict],
    llm_client: LLMClient,
    config: dict,
) -> dict:
    """Build a GlobalEmotionalProfile from all conversations for one character.

    Process:
    - Group conversations into batches of batch_size (config meld.batch_size, default 20)
    - For each batch: ONE LLM call for persona_card + hexaco_descriptions + csi_descriptions
    - For each batch: 12 LLM calls for scoring via inventory_scorer.py
    - Average all batch scores
    - Compile emotion_sequence and emotion_distribution

    Args:
        character_name: MELD character name.
        conversations: Normalized MELD conversation dicts.
        llm_client: Shared LLMClient instance.
        config: Full parsed config dict.

    Returns:
        GlobalEmotionalProfile dict with keys:
          character, hexaco_scores, csi_scores, emotion_sequence,
          emotion_distribution, dominant_emotion, n_conversations
    """
    batch_size = int(config["meld"].get("batch_size", 20))
    scorer = _make_scorer(config, llm_client)

    all_hexaco_dim_scores: list[dict] = []
    all_csi_dim_scores: list[dict] = []

    n_batches = math.ceil(len(conversations) / batch_size) if conversations else 0
    for batch_idx in range(n_batches):
        batch = conversations[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        logger.info(
            "Aggregating batch %d/%d for '%s' (%d convs).",
            batch_idx + 1, n_batches, character_name, len(batch),
        )

        prompt = _build_batch_prompt(character_name, batch, config)
        raw = llm_client.generate(prompt, temperature=llm_client.temperature_extraction)
        parsed = _parse_batch_response(raw, character_name, batch_idx)

        hexaco_desc = parsed.get("hexaco_descriptions", {})
        csi_desc = parsed.get("csi_descriptions", {})

        h_result = scorer._score_single_persona(
            persona_id=f"{character_name}_batch_{batch_idx}",
            descriptions=hexaco_desc,
            dimensions_config=scorer._hexaco_questions["dimensions"],
        )
        c_result = scorer._score_single_persona(
            persona_id=f"{character_name}_batch_{batch_idx}",
            descriptions=csi_desc,
            dimensions_config=scorer._csi_questions["dimensions"],
        )
        all_hexaco_dim_scores.append(h_result["dimension_scores"])
        all_csi_dim_scores.append(c_result["dimension_scores"])

    mean_hexaco = _mean_scores(all_hexaco_dim_scores)
    mean_csi = _mean_scores(all_csi_dim_scores)

    emotion_sequence = []
    for conv in sorted(conversations, key=lambda c: c.get("dialogue_id", "")):
        emotion_sequence.extend(conv.get("emotion_labels", []))

    emotion_labels = config["meld"]["emotion_labels"]
    counts = Counter(emotion_sequence)
    total = sum(counts.values()) or 1
    emotion_distribution = {
        label: round(counts.get(label, 0) / total, 4)
        for label in emotion_labels
    }
    dominant_emotion = max(counts, key=counts.get) if counts else "neutral"

    return {
        "character": character_name,
        "hexaco_scores": mean_hexaco,
        "csi_scores": mean_csi,
        "emotion_sequence": emotion_sequence,
        "emotion_distribution": emotion_distribution,
        "dominant_emotion": dominant_emotion,
        "n_conversations": len(conversations),
    }


def _build_batch_prompt(
    character_name: str, batch: list[dict], config: dict
) -> str:
    scenes_lines = []
    for conv in batch:
        scenes_lines.append(f"--- Scene {conv.get('dialogue_id', '?')} ---")
        emotion_iter = iter(conv.get("emotion_labels", []))
        for turn in conv.get("turns", []):
            if turn["role"] == "Supporter":
                emotion = next(emotion_iter, "neutral")
                scenes_lines.append(f"{character_name} [{emotion}]: {turn['text']}")
            else:
                speaker = turn.get("speaker", "Other")
                scenes_lines.append(f"{speaker}: {turn['text']}")
        scenes_lines.append("")

    hexaco_keys = json.dumps(_HEXACO_DIM_NAMES)
    csi_keys = json.dumps(_CSI_DIM_NAMES)

    return _BATCH_TEMPLATE.safe_substitute(
        character_name=character_name,
        n_scenes=len(batch),
        scenes_text="\n".join(scenes_lines).strip(),
        hexaco_keys=hexaco_keys,
        csi_keys=csi_keys,
    )


def _parse_batch_response(
    raw: str, character_name: str, batch_idx: int
) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(cleaned)
        return {
            "persona_card": parsed.get("persona_card", {}),
            "hexaco_descriptions": {
                k: str(v)
                for k, v in parsed.get("hexaco_descriptions", {}).items()
            },
            "csi_descriptions": {
                k: str(v)
                for k, v in parsed.get("csi_descriptions", {}).items()
            },
        }
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "Failed to parse batch response for '%s' batch %d: %s. Response: %s",
            character_name, batch_idx, exc, raw[:200],
        )
        return {"persona_card": {}, "hexaco_descriptions": {}, "csi_descriptions": {}}


def _mean_scores(score_dicts: list[dict]) -> dict:
    if not score_dicts:
        return {}
    totals: dict[str, list[float]] = {}
    for sd in score_dicts:
        for dim, val in sd.items():
            totals.setdefault(dim, []).append(float(val))
    return {dim: round(sum(vals) / len(vals), 4) for dim, vals in totals.items()}


def _make_scorer(config: dict, llm_client: LLMClient) -> InventoryScorer:
    return InventoryScorer(config, llm_client)
