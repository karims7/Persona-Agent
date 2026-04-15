"""Dataset loading and normalization for all HuggingFace datasets used in the project.

Responsibility: load raw HuggingFace datasets and return them as normalized
Python lists of dicts. No LLM calls. No disk writes. Every dataset schema
quirk is isolated here so no other module touches HuggingFace directly.
"""

import json
import logging
from typing import Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and normalizes HuggingFace datasets defined in config.

    Args:
        config: The full parsed config.yaml dict.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._dataset_cfg = config["datasets"]

    def load_esconv(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load ESConv dataset and normalize to dialogue format.

        Each item has:
          - dialogue_id: str
          - turns: list of {"role": "Seeker"|"Supporter", "text": str,
                            "strategy": str|None}
          - seeker_problem: str
          - emotion_type: str
          - problem_type: str

        Args:
            max_samples: Cap on number of dialogues to return. None = all.

        Returns:
            List of normalized dialogue dicts.
        """
        cfg = self._dataset_cfg["esconv"]
        logger.info("Loading ESConv from '%s' split='%s'.", cfg["hf_name"], cfg["split"])
        raw = load_dataset(cfg["hf_name"], split=cfg["split"])

        dialogues: list[dict] = []
        for idx, sample in enumerate(raw):
            if max_samples is not None and idx >= max_samples:
                break
            dialogue = self._normalize_esconv_sample(idx, sample)
            dialogues.append(dialogue)

        logger.info("Loaded %d ESConv dialogues.", len(dialogues))
        return dialogues

    def _normalize_esconv_sample(self, idx: int, sample: dict) -> dict:
        """Convert a single ESConv sample to the normalized format.

        Args:
            idx: Integer index used as fallback dialogue_id.
            sample: Raw HuggingFace sample dict.

        Returns:
            Normalized dialogue dict.
        """
        dialogue_id = str(sample.get("conv_id", idx))
        turns: list[dict] = []

        dialog_field = sample.get("dialog", sample.get("conversation", []))

        if isinstance(dialog_field, str):
            try:
                dialog_field = json.loads(dialog_field)
            except (json.JSONDecodeError, TypeError):
                dialog_field = []

        for turn in dialog_field:
            role_raw = turn.get("speaker", turn.get("role", "Unknown"))
            role = "Seeker" if str(role_raw).lower() in ("seeker", "user") else "Supporter"
            text = str(turn.get("text", turn.get("content", ""))).strip()
            strategy = turn.get("strategy", None)
            turns.append({"role": role, "text": text, "strategy": strategy})

        seeker_problem = str(
            sample.get("situation", sample.get("problem", sample.get("context", "")))
        ).strip()
        emotion_type = str(sample.get("emotion_type", "")).strip()
        problem_type = str(sample.get("problem_type", "")).strip()

        return {
            "dialogue_id": dialogue_id,
            "turns": turns,
            "seeker_problem": seeker_problem,
            "emotion_type": emotion_type,
            "problem_type": problem_type,
            "source": "esconv",
        }

    def load_cams(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load CAMS dataset and normalize to a list of text samples.

        Each item has:
          - sample_id: str
          - text: str
          - label: str

        Args:
            max_samples: Cap on number of samples. None = all.

        Returns:
            List of normalized sample dicts.
        """
        cfg = self._dataset_cfg["cams"]
        logger.info("Loading CAMS from '%s' split='%s'.", cfg["hf_name"], cfg["split"])
        raw = load_dataset(cfg["hf_name"], split=cfg["split"])

        samples: list[dict] = []
        for idx, sample in enumerate(raw):
            if max_samples is not None and idx >= max_samples:
                break
            samples.append({
                "sample_id": str(sample.get("id", idx)),
                "text": str(sample.get("text", sample.get("post", ""))).strip(),
                "label": str(sample.get("label", sample.get("cause", ""))).strip(),
                "source": "cams",
            })

        logger.info("Loaded %d CAMS samples.", len(samples))
        return samples

    def load_dreaddit(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load Dreaddit dataset and normalize to a list of text samples.

        Each item has:
          - sample_id: str
          - text: str
          - label: int (0 = not stressed, 1 = stressed)
          - subreddit: str

        Args:
            max_samples: Cap on number of samples. None = all.

        Returns:
            List of normalized sample dicts.
        """
        cfg = self._dataset_cfg["dreaddit"]
        logger.info(
            "Loading Dreaddit from '%s' split='%s'.", cfg["hf_name"], cfg["split"]
        )
        raw = load_dataset(cfg["hf_name"], split=cfg["split"])

        samples: list[dict] = []
        for idx, sample in enumerate(raw):
            if max_samples is not None and idx >= max_samples:
                break
            samples.append({
                "sample_id": str(sample.get("id", idx)),
                "text": str(sample.get("text", "")).strip(),
                "label": int(sample.get("label", 0)),
                "subreddit": str(sample.get("subreddit", "")).strip(),
                "source": "dreaddit",
            })

        logger.info("Loaded %d Dreaddit samples.", len(samples))
        return samples

    def load_persona_hub(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load PersonaHub dataset and normalize to a list of persona dicts.

        Each item has:
          - persona_id: str
          - persona_text: str (the raw persona description)

        Args:
            max_samples: Cap on number of personas. None = all.

        Returns:
            List of normalized persona dicts.
        """
        cfg = self._dataset_cfg["persona_hub"]
        sample_size = max_samples or cfg.get("sample_size", 1000)
        logger.info(
            "Loading PersonaHub from '%s' split='%s' (max %d).",
            cfg["hf_name"],
            cfg["split"],
            sample_size,
        )
        raw = load_dataset(cfg["hf_name"], split=cfg["split"])

        personas: list[dict] = []
        for idx, sample in enumerate(raw):
            if idx >= sample_size:
                break
            persona_text = str(
                sample.get("persona", sample.get("input persona", sample.get("text", "")))
            ).strip()
            personas.append({
                "persona_id": str(idx),
                "persona_text": persona_text,
                "source": "persona_hub",
            })

        logger.info("Loaded %d PersonaHub personas.", len(personas))
        return personas

    def dialogue_to_text(self, dialogue: dict) -> str:
        """Convert a normalized dialogue dict to a plain-text string.

        Formats turns as "Role: text" lines joined by newlines.

        Args:
            dialogue: A normalized dialogue dict as returned by load_esconv.

        Returns:
            Multi-line string representation of the dialogue.
        """
        lines = []
        for turn in dialogue.get("turns", []):
            lines.append(f"{turn['role']}: {turn['text']}")
        return "\n".join(lines)

    def dialogue_to_json_text(self, dialogue: dict) -> str:
        """Convert a normalized dialogue dict to a JSON string.

        Used when the paper prompt specifies JSON-formatted input.

        Args:
            dialogue: A normalized dialogue dict as returned by load_esconv.

        Returns:
            JSON string of the turns list.
        """
        return json.dumps(dialogue.get("turns", []), ensure_ascii=False)
