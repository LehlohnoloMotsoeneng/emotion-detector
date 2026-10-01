"""Tests for the Watson emotion detector."""

import unittest
from unittest.mock import Mock, patch

import requests

from emotion_detection import emotion_detector
from server import app


class TestEmotionDetector(unittest.TestCase):
    """Validate the Watson request and parsed emotion output."""

    def _mock_response(self, emotion_scores):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "emotionPredictions": [{"emotion": emotion_scores}]
        }
        mock_response.raise_for_status.return_value = None
        return mock_response

    @patch("emotion_detection.requests.post")
    def test_happy_message_has_joy_as_dominant_emotion(self, mock_post):
        """Happy text should resolve to joy when the Watson response is parsed."""
        mock_post.return_value = self._mock_response({
            "anger": 0.02,
            "disgust": 0.01,
            "fear": 0.01,
            "joy": 0.93,
            "sadness": 0.03,
        })

        result = emotion_detector("I am very happy today")

        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "joy")
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["grpc-metadata-mm-model-id"],
            "emotion_aggregated-workflow_lang_en_stock",
        )

    @patch("emotion_detection.requests.post")
    def test_sad_message_has_sadness_as_dominant_emotion(self, mock_post):
        """Sad text should resolve to sadness when the Watson response is parsed."""
        mock_post.return_value = self._mock_response({
            "anger": 0.05,
            "disgust": 0.02,
            "fear": 0.04,
            "joy": 0.08,
            "sadness": 0.81,
        })

        result = emotion_detector("I feel very sad and disappointed")

        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "sadness")

    @patch("emotion_detection.requests.post")
    def test_angry_message_has_anger_as_dominant_emotion(self, mock_post):
        """Angry text should resolve to anger when the Watson response is parsed."""
        mock_post.return_value = self._mock_response({
            "anger": 0.90,
            "disgust": 0.04,
            "fear": 0.02,
            "joy": 0.01,
            "sadness": 0.03,
        })

        result = emotion_detector("I am angry and enraged")

        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "anger")

    @patch("emotion_detection.requests.post")
    def test_watson_connection_failure_returns_service_error(self, mock_post):
        """A network timeout should be reported as a Watson service error."""
        mock_post.side_effect = requests.exceptions.Timeout("timed out")

        result = emotion_detector("I am very happy today")

        self.assertEqual(result["error"], "Watson NLP service is unavailable.")

    @patch("emotion_detection.requests.post")
    def test_invalid_watson_response_returns_service_error(self, mock_post):
        """Malformed Watson output should not be treated as invalid user input."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"unexpected": "response"}
        mock_post.return_value = mock_response

        result = emotion_detector("I am very happy today")

        self.assertEqual(result["error"], "Invalid response from Watson NLP service.")

    def test_blank_input_returns_error(self):
        """Blank input should produce an error response without contacting Watson."""
        result = emotion_detector("   ")
        self.assertIn("error", result)

    def test_flask_blank_input_returns_http_400(self):
        """Blank input through Flask should return 400."""
        client = app.test_client()
        response = client.get("/emotionDetector?text=%20%20")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Please provide text to analyze.")

    @patch("emotion_detection.requests.post")
    def test_flask_service_error_returns_http_503(self, mock_post):
        """A Watson service failure should become HTTP 503 instead of a text validation error."""
        mock_post.side_effect = requests.exceptions.Timeout("timed out")
        client = app.test_client()
        response = client.get("/emotionDetector?text=I%20am%20very%20happy%20today")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["error"], "Watson NLP service is unavailable.")


if __name__ == "__main__":
    unittest.main()
