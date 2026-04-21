"""Encode a GlobalEmotionalProfile as a fixed-length numerical vector.

Responsibility: convert a GlobalEmotionalProfile dict to a numpy array of
shape (20,). No LLM calls.

Vector layout (20 values):
  [0:6]   HEXACO dimension scores (order: config hexaco.dimensions)
  [6:12]  CSI dimension scores (order: config csi.dimensions)
  [12:19] Emotion distribution (order: config meld.emotion_labels, 7 values)
  [19]    Dominant emotion index (0-6)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

_VECTOR_LENGTH = 20


def encode(profile: dict, config: dict = None) -> np.ndarray:
    """Convert a GlobalEmotionalProfile to a (20,) float64 numpy array.

    Vector layout:
      [0:6]   HEXACO dimension scores
      [6:12]  CSI dimension scores
      [12:19] Emotion distribution (7 labels, normalized)
      [19]    Dominant emotion index (float)

    Args:
        profile: GlobalEmotionalProfile dict with hexaco_scores, csi_scores,
                 emotion_distribution, dominant_emotion.
        config: Full parsed config dict. When provided, uses config dimension order;
                otherwise falls back to profile key order.

    Returns:
        numpy.ndarray of shape (20,) with dtype float64.
    """
    if config is not None:
        hexaco_dims = [d["name"] for d in config["hexaco"]["dimensions"]]
        csi_dims = [d["name"] for d in config["csi"]["dimensions"]]
        emotion_labels = config["meld"]["emotion_labels"]
    else:
        hexaco_dims = list(profile.get("hexaco_scores", {}).keys())
        csi_dims = list(profile.get("csi_scores", {}).keys())
        emotion_labels = [
            "neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"
        ]

    hexaco_scores = profile.get("hexaco_scores", {})
    csi_scores = profile.get("csi_scores", {})
    emotion_dist = profile.get("emotion_distribution", {})
    dominant_emotion = profile.get("dominant_emotion", "neutral")

    hexaco_vec = np.array(
        [float(hexaco_scores.get(dim, 3.0)) for dim in hexaco_dims],
        dtype=np.float64,
    )
    csi_vec = np.array(
        [float(csi_scores.get(dim, 3.0)) for dim in csi_dims],
        dtype=np.float64,
    )
    emotion_vec = np.array(
        [float(emotion_dist.get(label, 0.0)) for label in emotion_labels],
        dtype=np.float64,
    )
    dominant_idx = float(
        emotion_labels.index(dominant_emotion)
        if dominant_emotion in emotion_labels else 0
    )

    vector = np.concatenate([hexaco_vec, csi_vec, emotion_vec, [dominant_idx]])
    assert vector.shape == (_VECTOR_LENGTH,), (
        f"Profile vector shape mismatch: {vector.shape} != ({_VECTOR_LENGTH},)"
    )
    return vector
