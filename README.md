# Emotion Detector

This project implements a simple web application that analyzes the emotional tone of user-provided text using IBM Watson Natural Language Understanding.

## Features

- Detects emotions such as anger, disgust, fear, joy, and sadness
- Returns the dominant emotion for each submission
- Exposes a Flask route for browser and API-based use
- Includes unit tests and a simple package structure for reuse

## Project structure

- `emotion_detection.py` - core function for emotion analysis
- `EmotionDetection/` - Python package wrapper
- `server.py` - Flask application and UI routes
- `test_emotion_detection.py` - unit tests
- `requirements.txt` - project dependencies

## Local setup

1. Create a virtual environment if desired.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Set the Watson credentials as environment variables:
   `WATSON_API_KEY` and `WATSON_URL`
4. Run the app:
   `python server.py`

## Example usage

```python
from emotion_detection import emotion_detector

result = emotion_detector("I am very happy today")
print(result)
```

## Flask endpoint

- `GET /emotionDetector?text=I%20am%20very%20happy`

The endpoint returns a JSON object containing the emotion scores and the dominant emotion.
