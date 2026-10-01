"""Flask application for emotion analysis."""

from __future__ import annotations

from flask import Flask, jsonify, request

from emotion_detection import emotion_detector

app = Flask(__name__)


@app.route("/")
def index():
    """Render the simple text-analysis form used for browser testing."""
    return """
    <html>
      <head><title>Emotion Detector</title></head>
      <body>
        <h1>Emotion Detector</h1>
        <form action="/emotionDetector" method="get">
          <label for="text">Text:</label>
          <textarea id="text" name="text" rows="5" cols="60"></textarea><br>
          <button type="submit">Analyze</button>
        </form>
      </body>
    </html>
    """


@app.route("/emotionDetector", methods=["GET", "POST"])
def emotion_endpoint():
    """Accept a text string, analyze it, and return an emotion dictionary."""
    payload = request.args.get("text") or request.form.get("text") or ""
    if not payload or not str(payload).strip():
        return jsonify({"error": "Please provide text to analyze."}), 400

    result = emotion_detector(payload)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
