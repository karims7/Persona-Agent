"""Predict next emotion and intensity given conversation context.

Responsibility: given a GlobalEmotionalProfile combined with profile_context and
conversation context turns, prompt the LLM to predict next emotion and intensity.
Supports with_profile=True (full profile injected) and with_profile=False (context only).
"""

import json
import logging
from string import Template

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_VALID_EMOTIONS = frozenset(
    ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
)

_WITH_PROFILE_TEMPLATE = Template(
    "Character profile for $character_name:\n"
    "- HEXACO scores (personality, 1-5): $hexaco_scores\n"
    "- Communication style (CSI, 1-5): $csi_scores\n"
    "- Identity: $identity\n"
    "- Core values: $core_values\n"
    "- Goals: $goals\n"
    "- Fears: $fears\n"
    "- Close relationships: $close_relationships\n"
    "- Past events: $past_events\n"
    "- Habits: $habits\n"
    "- Hobbies and interests: $hobbies_and_interests\n"
    "- Responsibilities: $responsibilities\n"
    "- Reputation: $reputation\n"
    "- Limits: $limits\n"
    "- Emotional triggers (positive): $positive_triggers\n"
    "- Emotional triggers (negative): $negative_triggers\n"
    "- Relationship context: $relationship_context\n\n"
    "Given this profile and the following conversation context, predict:\n"
    "1. What emotion will $character_name express next? "
    "Choose from: neutral, surprise, fear, sadness, joy, disgust, anger\n"
    "2. How intensely (1=very mild, 5=extremely intense)?\n\n"
    "Conversation so far:\n$context\n\n"
    "Return JSON: {\"predicted_emotion\": \"...\", \"predicted_intensity\": N}"
)

_NO_PROFILE_TEMPLATE = Template(
    "You are analyzing a conversation from the TV show Friends. "
    "Based on the following conversation context, predict what emotion "
    "$character_name will express next.\n\n"
    "Choose exactly one emotion from: neutral, surprise, fear, sadness, joy, disgust, anger\n"
    "Rate the intensity (1=very mild, 5=extremely intense).\n\n"
    "Conversation so far:\n$context\n\n"
    "Return JSON: {\"predicted_emotion\": \"...\", \"predicted_intensity\": N}"
)


def predict(
    conversation_context: list[dict],
    profile: dict,
    with_profile: bool,
    config: dict,
    llm_client: LLMClient,
) -> dict:
    """Predict next emotion and intensity for a character.

    Args:
        conversation_context: List of turn dicts (role, text, speaker) for the first N turns.
        profile: Combined dict with GlobalEmotionalProfile keys AND profile_context keys.
        with_profile: If True, inject full profile into prompt.
        config: Full parsed config dict.
        llm_client: Shared LLMClient instance.

    Returns:
        Dict with: predicted_emotion, predicted_intensity, actual_emotion,
        actual_intensity, with_profile.
    """
    character_name = profile.get("character", "the character")
    valid_emotions = set(config["meld"]["emotion_labels"])

    context_text = _format_context(conversation_context, character_name)

    if with_profile:
        prompt = _build_with_profile_prompt(character_name, profile, context_text)
    else:
        prompt = _NO_PROFILE_TEMPLATE.safe_substitute(
            character_name=character_name,
            context=context_text,
        )

    raw = llm_client.generate(prompt, temperature=llm_client.temperature_extraction)
    predicted_emotion, predicted_intensity = _parse_response(raw, valid_emotions)

    return {
        "predicted_emotion": predicted_emotion,
        "predicted_intensity": predicted_intensity,
        "actual_emotion": profile.get("_actual_emotion", "neutral"),
        "actual_intensity": profile.get("_actual_intensity", 3),
        "with_profile": with_profile,
    }


def _build_with_profile_prompt(
    character_name: str, profile: dict, context_text: str
) -> str:
    hexaco = profile.get("hexaco_scores", {})
    csi = profile.get("csi_scores", {})
    triggers = profile.get("emotional_triggers", {})
    rel_profiles = profile.get("relationship_profiles", {})

    hexaco_str = "; ".join(f"{k[:4]}: {v:.1f}" for k, v in hexaco.items())
    csi_str = "; ".join(f"{k[:4]}: {v:.1f}" for k, v in csi.items())

    positive_triggers = ", ".join(triggers.get("positive", [])[:5]) or "none identified"
    negative_triggers = ", ".join(triggers.get("negative", [])[:5]) or "none identified"

    context_speakers = {
        t.get("speaker", "") for t in [] if t.get("role") == "Seeker"
    }
    rel_context = "Not available"
    for speaker in context_speakers:
        if speaker in rel_profiles:
            rel_data = rel_profiles[speaker]
            rel_context = (
                f"{speaker}: {rel_data.get('dynamic_description', '')}"
            )
            break

    def _list_str(key: str) -> str:
        val = profile.get(key, [])
        if isinstance(val, list):
            return ", ".join(str(v) for v in val[:5]) or "none"
        return str(val) or "none"

    return _WITH_PROFILE_TEMPLATE.safe_substitute(
        character_name=character_name,
        hexaco_scores=hexaco_str or "not available",
        csi_scores=csi_str or "not available",
        identity=str(profile.get("identity", "")) or "not available",
        core_values=_list_str("core_values"),
        goals=_list_str("goals"),
        fears=_list_str("fears"),
        close_relationships=_list_str("close_relationships"),
        past_events=_list_str("past_events"),
        habits=_list_str("habits"),
        hobbies_and_interests=_list_str("hobbies_and_interests"),
        responsibilities=_list_str("responsibilities"),
        reputation=str(profile.get("reputation", "")) or "not available",
        limits=_list_str("limits"),
        positive_triggers=positive_triggers,
        negative_triggers=negative_triggers,
        relationship_context=rel_context,
        context=context_text,
    )


def _format_context(turns: list[dict], character_name: str) -> str:
    lines = []
    for turn in turns:
        if turn["role"] == "Supporter":
            lines.append(f"{character_name}: {turn['text']}")
        else:
            speaker = turn.get("speaker", "Other")
            lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def _parse_response(raw: str, valid_emotions: set) -> tuple:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        parsed = json.loads(cleaned)
        emotion = str(parsed.get("predicted_emotion", "neutral")).lower()
        if emotion not in valid_emotions:
            emotion = "neutral"
        intensity = max(1, min(5, int(parsed.get("predicted_intensity", 3))))
        return emotion, intensity
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to parse emotion prediction: %s. Response: %s",
            exc, raw[:200],
        )
        return "neutral", 3
