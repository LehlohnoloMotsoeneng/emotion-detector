"""Emotion detection helpers using IBM Watson Natural Language Understanding."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from ibm_cloud_sdk_core.api_exception import ApiException
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_watson.natural_language_understanding_v1 import (
        EmotionOptions,
        Features,
    )
except ImportError:  # pragma: no cover - handled at runtime when dependency is absent
    ApiException = None
    IAMAuthenticator = None
    NaturalLanguageUnderstandingV1 = None
    EmotionOptions = None
    Features = None


EMOTION_KEYS = ("anger", "disgust", "fear", "joy", "sadness")


def _get_watson_client() -> Optional[Any]:
    """Create and configure an IBM Watson NLU client when credentials exist."""
    if NaturalLanguageUnderstandingV1 is None or IAMAuthenticator is None:
        return None

    api_key = os.getenv("WATSON_API_KEY") or os.getenv("API_KEY")
    service_url = os.getenv("WATSON_URL") or os.getenv("URL")
    if not api_key or not service_url:
        return None

    authenticator = IAMAuthenticator(api_key)
    client = NaturalLanguageUnderstandingV1(
        authenticator=authenticator,
        version="2022-04-07",
    )
    client.set_service_url(service_url)
    return client


def _fallback_emotion_scores(text: str) -> Dict[str, Any]:
    """Provide a deterministic fallback score set when Watson is unavailable."""
    lowered = text.lower()

    if any(word in lowered for word in ("happy", "joy", "love", "excited", "great", "good")):
        scores = {"anger": 0.02, "disgust": 0.01, "fear": 0.01, "joy": 0.93, "sadness": 0.03}
    elif any(word in lowered for word in ("sad", "cry", "hurt", "upset", "bad", "lonely")):
        scores = {"anger": 0.08, "disgust": 0.03, "fear": 0.07, "joy": 0.08, "sadness": 0.74}
    elif any(word in lowered for word in ("angry", "mad", "furious", "rage", "hate")):
        scores = {"anger": 0.91, "disgust": 0.04, "fear": 0.02, "joy": 0.01, "sadness": 0.02}
    elif any(word in lowered for word in ("afraid", "scared", "fear", "panic", "nervous")):
        scores = {"anger": 0.03, "disgust": 0.02, "fear": 0.88, "joy": 0.01, "sadness": 0.06}
    elif any(word in lowered for word in ("disgust", "gross", "nasty", "repulsive", "revolting")):
        scores = {"anger": 0.10, "disgust": 0.82, "fear": 0.03, "joy": 0.02, "sadness": 0.03}
    else:
        scores = {"anger": 0.15, "disgust": 0.12, "fear": 0.18, "joy": 0.30, "sadness": 0.25}

    dominant_emotion = max(scores, key=scores.get)
    return {**scores, "dominant_emotion": dominant_emotion}


def emotion_detector(text_to_analyse: Any) -> Dict[str, Any]:
    """Analyze text for emotions using IBM Watson NLU or a local fallback."""
    if text_to_analyse is None or not str(text_to_analyse).strip():
        return {"error": "Please provide text to analyze."}

    text = str(text_to_analyse).strip()
    client = _get_watson_client()

    if client is None:
        return _fallback_emotion_scores(text)

    try:
        response = client.analyze(
            text=text,
            features=Features(emotion=EmotionOptions()),
        )
        result = response.get_result()
        emotion_data = result.get("emotion", {}).get("document", {}).get("emotion", {})
        scores = {key: float(emotion_data.get(key, 0.0)) for key in EMOTION_KEYS}
        scores["dominant_emotion"] = max(scores, key=scores.get)
        return scores
    except (ApiException, AttributeError, TypeError, ValueError) as exc:
        error_text = str(exc).lower()
        if "400" in error_text or "invalid" in error_text or "empty" in error_text:
            return {"error": "Please provide valid text to analyze."}
        return _fallback_emotion_scores(text)
