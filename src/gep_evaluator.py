"""Evaluate emotion prediction accuracy with vs without the Global Emotional Profile.

Responsibility: compute accuracy, per-emotion F1, and mean intensity error
from pre-computed prediction results. No LLM calls.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_EMOTION_LABELS = [
    "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
]


def evaluate(predictions: list[dict]) -> dict:
    """Compare prediction accuracy with profile vs without profile.

    Args:
        predictions: List of dicts from emotion_predictor.predict().
            Each dict must have: predicted_emotion, predicted_intensity,
            actual_emotion, actual_intensity, with_profile (bool).

    Returns:
        Dict with:
          - accuracy_with_profile: float
          - accuracy_without_profile: float
          - improvement: float (accuracy_with - accuracy_without)
          - per_emotion_accuracy: dict mapping emotion -> {"with": float, "without": float}
          - mean_intensity_error_with: float
          - mean_intensity_error_without: float
          - n_predictions: int (per mode, i.e. total / 2)
    """
    with_preds = [p for p in predictions if p.get("with_profile", False)]
    without_preds = [p for p in predictions if not p.get("with_profile", False)]

    acc_with = _accuracy(with_preds)
    acc_without = _accuracy(without_preds)

    return {
        "accuracy_with_profile": acc_with,
        "accuracy_without_profile": acc_without,
        "improvement": round(acc_with - acc_without, 4),
        "per_emotion_accuracy": _per_emotion_accuracy(with_preds, without_preds),
        "mean_intensity_error_with": _mean_intensity_error(with_preds),
        "mean_intensity_error_without": _mean_intensity_error(without_preds),
        "n_predictions": len(with_preds),
    }


def _accuracy(preds: list[dict]) -> float:
    if not preds:
        return 0.0
    correct = sum(
        1 for p in preds if p["predicted_emotion"] == p["actual_emotion"]
    )
    return round(correct / len(preds), 4)


def _mean_intensity_error(preds: list[dict]) -> float:
    if not preds:
        return 0.0
    errors = [abs(p["predicted_intensity"] - p["actual_intensity"]) for p in preds]
    return round(sum(errors) / len(errors), 4)


def _per_emotion_accuracy(
    with_preds: list[dict], without_preds: list[dict]
) -> dict:
    result = {}
    all_emotions = set()
    for p in with_preds + without_preds:
        all_emotions.add(p.get("actual_emotion", "neutral"))

    def emotion_acc(preds: list[dict], emotion: str) -> float:
        relevant = [p for p in preds if p["actual_emotion"] == emotion]
        if not relevant:
            return 0.0
        correct = sum(1 for p in relevant if p["predicted_emotion"] == emotion)
        return round(correct / len(relevant), 4)

    for emotion in sorted(all_emotions):
        result[emotion] = {
            "with": emotion_acc(with_preds, emotion),
            "without": emotion_acc(without_preds, emotion),
        }
    return result
