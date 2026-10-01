"""Emotion detection using the Skills Network Watson NLP EmotionPredict service."""

from __future__ import annotations

from typing import Any, Dict

import requests


WATSON_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
WATSON_MODEL_ID = "emotion_aggregated-workflow_lang_en_stock"
EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")


def _extract_emotion_scores(payload: Any) -> Dict[str, float]:
    """Read emotion scores from the Watson EmotionPredict response."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Missing Watson emotion predictions")
        return _extract_emotion_scores(payload[0])

    if not isinstance(payload, dict):
        raise ValueError("Unexpected Watson emotion response")

    if "emotionPredictions" in payload:
        return _extract_emotion_scores(payload["emotionPredictions"])

    for key in ("emotion", "emotion_prediction", "prediction"):
        value = payload.get(key)
        if isinstance(value, dict):
            scores = {
                emotion_key: float(value[emotion_key])
                for emotion_key in EMOTION_KEYS
                if emotion_key in value
            }
            if scores:
                return scores

    if any(key in payload for key in EMOTION_KEYS):
        scores = {
            emotion_key: float(payload[emotion_key])
            for emotion_key in EMOTION_KEYS
            if emotion_key in payload
        }
        if scores:
            return scores

    raise ValueError("Watson emotion data not found in response")


def emotion_detector(text_to_analyse: Any) -> Dict[str, Any]:
    """Send valid text to the Watson NLP EmotionPredict endpoint."""
    if text_to_analyse is None or not str(text_to_analyse).strip():
        return {"error": "Please provide text to analyze."}

    text = str(text_to_analyse).strip()
    headers = {"grpc-metadata-mm-model-id": WATSON_MODEL_ID}
    payload = {"raw_document": {"text": text}}

    try:
        response = requests.post(
            WATSON_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        response_payload = response.json()
        scores = _extract_emotion_scores(response_payload)
    except requests.RequestException:
        return {"error": "Watson NLP service is unavailable."}
    except (ValueError, TypeError, KeyError, AttributeError):
        return {"error": "Invalid response from Watson NLP service."}

    normalized_scores = {
        emotion_key: float(scores.get(emotion_key, 0.0))
        for emotion_key in EMOTION_KEYS
    }
    normalized_scores["dominant_emotion"] = max(
        normalized_scores,
        key=normalized_scores.get,
    )
    return normalized_scores
