"""Extract emotional triggers, character card, and relationship profiles for a MELD character.

Responsibility: one combined LLM call that extracts all 12 character card fields,
emotional triggers, and per-relationship analysis from all conversations.
Saves result to outputs/profiles/{character}/profile_context.json.
"""

import json
import logging
import os
from string import Template

from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_MAX_EXCERPT_CHARS = 12000

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FULL_PROFILE_TEMPLATE = Template(
    "You are building a deep psychological and behavioral profile of $character_name "
    "from the TV show Friends. Below are excerpts from their conversations, "
    "each with speaker names and emotion labels.\n\n"
    "$conversation_excerpts\n\n"
    "Based on these conversations, return a JSON object with EXACTLY these keys:\n\n"
    "core_values: list of strings — what $character_name believes matters most\n"
    "goals: list of strings — short-term and long-term goals\n"
    "fears: list of strings — what they avoid or protect themselves from\n"
    "close_relationships: list of strings — who shaped them, who they trust, "
    "who they resent, who they would sacrifice for\n"
    "past_events: list of strings — trauma, success, failure, loss, betrayal, "
    "love, major turning points\n"
    "habits: list of strings — what they do regularly\n"
    "hobbies_and_interests: list of strings — what they choose to spend time on\n"
    "responsibilities: list of strings — job, family role, social role\n"
    "identity: string — how they see themselves\n"
    "reputation: string — how others see them\n"
    "limits: list of strings — what they will never do and what line they cross under pressure\n"
    "emotional_triggers: a dict with keys: positive (list), negative (list), neutral (list)\n"
    "relationship_profiles: a dict mapping each of these character names to a dict "
    "with keys communication_style_with_them (string), "
    "emotional_reactions_to_them (string), dynamic_description (string) — "
    "all from $character_name's perspective only. "
    "Characters: $other_characters\n\n"
    "Do not include any text outside the JSON object."
)


def extract_full_profile_context(
    character_name: str,
    conversations: list[dict],
    llm_client: LLMClient,
    config: dict,
) -> dict:
    """Extract character card, emotional triggers, and relationship profiles in one LLM call.

    Args:
        character_name: Target character name.
        conversations: All normalized MELD conversations for this character.
        llm_client: Shared LLMClient instance.
        config: Full parsed config dict.

    Returns:
        Dict with all 12 character card fields + relationship_profiles.
        Also saved to outputs/profiles/{character}/profile_context.json.
    """
    output_dir = os.path.join(
        _PROJECT_ROOT,
        config["paths"]["outputs"]["profiles"],
        character_name.lower(),
    )
    output_path = os.path.join(output_dir, "profile_context.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        logger.info(
            "Loaded cached profile context for '%s' from '%s'.",
            character_name, output_path,
        )
        return cached

    other_characters = [
        c for c in config["meld"]["characters"] if c != character_name
    ]
    excerpts = _build_excerpts(character_name, conversations)
    prompt = _FULL_PROFILE_TEMPLATE.safe_substitute(
        character_name=character_name,
        conversation_excerpts=excerpts,
        other_characters=", ".join(other_characters),
    )

    raw = llm_client.generate(prompt, temperature=llm_client.temperature_extraction)
    result = _parse_response(raw, character_name, other_characters)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Profile context for '%s' saved to '%s'.", character_name, output_path)
    return result


def _build_excerpts(character_name: str, conversations: list[dict]) -> str:
    lines = []
    for conv in conversations:
        turns = conv.get("turns", [])
        emotion_labels = conv.get("emotion_labels", [])
        emotion_iter = iter(emotion_labels)
        lines.append(f"--- Scene {conv.get('dialogue_id', '?')} ---")
        for turn in turns:
            if turn["role"] == "Supporter":
                emotion = next(emotion_iter, "neutral")
                lines.append(f"{character_name} [{emotion}]: {turn['text']}")
            else:
                speaker = turn.get("speaker", "Other")
                lines.append(f"{speaker}: {turn['text']}")
        lines.append("")

    full_text = "\n".join(lines)
    if len(full_text) > _MAX_EXCERPT_CHARS:
        full_text = full_text[:_MAX_EXCERPT_CHARS] + "\n...[truncated]"
    return full_text


def _parse_response(
    raw: str, character_name: str, other_characters: list[str]
) -> dict:
    _default_triggers = {"positive": [], "negative": [], "neutral": []}
    default = {
        "core_values": [],
        "goals": [],
        "fears": [],
        "close_relationships": [],
        "past_events": [],
        "habits": [],
        "hobbies_and_interests": [],
        "responsibilities": [],
        "identity": "",
        "reputation": "",
        "limits": [],
        "emotional_triggers": _default_triggers,
        "relationship_profiles": {c: {} for c in other_characters},
    }

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(cleaned)
        result = {}
        for key in (
            "core_values", "goals", "fears", "close_relationships",
            "past_events", "habits", "hobbies_and_interests", "responsibilities",
            "limits",
        ):
            val = parsed.get(key, [])
            result[key] = list(val) if isinstance(val, list) else [str(val)]
        for key in ("identity", "reputation"):
            result[key] = str(parsed.get(key, ""))
        triggers = parsed.get("emotional_triggers", {})
        result["emotional_triggers"] = {
            "positive": list(triggers.get("positive", [])),
            "negative": list(triggers.get("negative", [])),
            "neutral": list(triggers.get("neutral", [])),
        }
        rel = parsed.get("relationship_profiles", {})
        result["relationship_profiles"] = dict(rel) if isinstance(rel, dict) else {}
        return result
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "Failed to parse profile context for '%s': %s. Response: %s",
            character_name, exc, raw[:200],
        )
        return default
