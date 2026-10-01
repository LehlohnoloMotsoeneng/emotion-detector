"""Tests for the emotion detector."""

import unittest

from emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test the reaction scores and errors."""

    def test_happy_message_has_joy_as_dominant_emotion(self):
        """Happy text should resolve to joy."""
        result = emotion_detector("I am very happy today")
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_sad_message_has_sadness_as_dominant_emotion(self):
        """Sad text should resolve to sadness."""
        result = emotion_detector("I feel very sad and disappointed")
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_angry_message_has_anger_as_dominant_emotion(self):
        """Angry text should resolve to anger."""
        result = emotion_detector("I am angry and enraged")
        self.assertIn("dominant_emotion", result)
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_blank_input_returns_error(self):
        """Blank input should produce an error response."""
        result = emotion_detector("   ")
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
