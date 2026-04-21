"""Load and normalize MELD dataset from local CSV files, organized by character.

Responsibility: read train_sent_emo.csv / test_sent_emo.csv, group utterances
by Dialogue_ID, and return per-character conversation lists. No LLM calls.
No disk writes. One CSV load per split.

Real MELD CSV columns (confirmed from data/meld_inspection.txt):
  Sr No., Utterance, Speaker, Emotion, Sentiment,
  Dialogue_ID, Utterance_ID, Season, Episode, StartTime, EndTime
"""

import csv
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_character_conversations(
    character_name: str,
    split: str,
    config: dict,
) -> list[dict]:
    """Load all MELD conversations containing a character and normalize them.

    Each returned dict has:
      - dialogue_id: str
      - turns: list of {"role": str, "text": str, "speaker": str, "emotion": str}
        (target character gets role "Supporter"; all others get role "Seeker")
      - speaker_name: str
      - emotion_labels: list of str (one per Supporter turn, chronological)
      - season: int | None
      - episode: int | None

    Args:
        character_name: MELD character name, e.g. "Ross".
        split: "train" or "test".
        config: Full parsed config dict.

    Returns:
        List of normalized conversation dicts containing the character.
    """
    rows = _load_csv(split, config)
    return _build_conversations(rows, character_name)


def load_all_characters(split: str, config: dict) -> dict[str, list[dict]]:
    """Load MELD conversations for all characters in config (one CSV load).

    Args:
        split: "train" or "test".
        config: Full parsed config dict.

    Returns:
        Dict mapping character_name -> list of normalized conversation dicts.
    """
    characters = config["meld"]["characters"]
    rows = _load_csv(split, config)
    return {char: _build_conversations(rows, char) for char in characters}


def _load_csv(split: str, config: dict) -> list[dict]:
    """Read the local MELD CSV for the given split.

    Args:
        split: "train" or "test".
        config: Full parsed config dict.

    Returns:
        List of row dicts with MELD column keys.

    Raises:
        FileNotFoundError: If the local CSV is not present.
    """
    key = "local_train_path" if split == "train" else "local_test_path"
    path = config["meld"][key]
    if not os.path.isabs(path):
        path = os.path.join(_PROJECT_ROOT, path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MELD local CSV not found: '{path}'. "
            "Place train_sent_emo.csv / test_sent_emo.csv in the data/ folder."
        )

    logger.info("Loading MELD from local CSV: %s", path)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    logger.info("Loaded %d rows from %s split.", len(rows), split)
    return rows


def _build_conversations(rows: list[dict], character_name: str) -> list[dict]:
    """Group rows by Dialogue_ID and return conversations containing the character.

    Args:
        rows: All CSV row dicts for a split.
        character_name: Target character name.

    Returns:
        List of normalized conversation dicts where the character appears.
    """
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[row["Dialogue_ID"]].append(row)

    conversations = []
    for did, d_rows in sorted(groups.items(), key=lambda x: _int_key(x[0])):
        d_rows.sort(key=lambda r: int(r["Utterance_ID"]))
        if not any(r["Speaker"] == character_name for r in d_rows):
            continue
        conv = _normalize_dialogue(did, d_rows, character_name)
        if conv["turns"]:
            conversations.append(conv)

    logger.info("Found %d conversations for '%s'.", len(conversations), character_name)
    return conversations


def _normalize_dialogue(
    dialogue_id: str, rows: list[dict], character_name: str
) -> dict:
    """Convert sorted MELD rows for one dialogue to the normalized format.

    Args:
        dialogue_id: Dialogue_ID string.
        rows: Rows sorted by Utterance_ID.
        character_name: Target character (role "Supporter"); others get "Seeker".

    Returns:
        Normalized conversation dict.
    """
    turns = []
    emotion_labels = []
    season = None
    episode = None

    for row in rows:
        speaker = row["Speaker"]
        text = row["Utterance"].strip()
        emotion = row["Emotion"].lower()

        if season is None and row.get("Season"):
            try:
                season = int(row["Season"])
            except (ValueError, TypeError):
                pass
        if episode is None and row.get("Episode"):
            try:
                episode = int(row["Episode"])
            except (ValueError, TypeError):
                pass

        role = "Supporter" if speaker == character_name else "Seeker"
        turns.append({"role": role, "text": text, "speaker": speaker, "emotion": emotion})
        if speaker == character_name:
            emotion_labels.append(emotion)

    return {
        "dialogue_id": dialogue_id,
        "turns": turns,
        "speaker_name": character_name,
        "emotion_labels": emotion_labels,
        "season": season,
        "episode": episode,
    }


def _int_key(val: str) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
