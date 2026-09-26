"""Flask entry point for the MindEasy wellness-support prototype."""

import logging
import os
import re

from flask import Flask, jsonify, render_template, request, session

from src.predict import predict_sentiment
from src.companion import CompanionError, get_companion_response
from src.response_engine import get_supportive_content

app = Flask(__name__, static_folder="public", static_url_path="/public")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-only-secret")
logging.basicConfig(level=logging.INFO)

SAFETY_PATTERN = re.compile(r"\b(kill myself|hurt myself|harm myself|end my life|suicid(?:e|al)|self harm|don't want to live|do not want to live)\b", re.I)


@app.get("/")
def index():
	return render_template("index.html")


@app.post("/predict")
def predict():
	try:
		payload = request.get_json(silent=True) or {}
		text = payload.get("text", "")
		if not isinstance(text, str) or not text.strip():
			return jsonify({"error": "Please write a few words before submitting."}), 400
		if len(text) > 2000:
			return jsonify({"error": "Please keep your message under 2,000 characters."}), 400
		safety = bool(SAFETY_PATTERN.search(text))
		sentiment, confidence = predict_sentiment(text)
		resource_state = session.get("resource_state", {})
		content = get_supportive_content(sentiment, text, safety, resource_state)
		session["resource_state"] = content.pop("resource_state")
		history = session.get("conversation", [])
		if not safety:
			companion = get_companion_response(text, sentiment, history)
			content.update({"acknowledgement": "MindEasy companion", "message": companion, "follow_up": "", "suggestion": ""})
			history.extend([{"role": "user", "text": text}, {"role": "mindeasy", "text": companion}])
		else:
			history.extend([{"role": "user", "text": text}, {"role": "mindeasy", "text": content["message"]}])
		session["conversation"] = history[-12:]
		return jsonify({"sentiment": sentiment, "confidence": confidence, **content, "conversation": session["conversation"]})
	except FileNotFoundError as error:
		logging.error(error)
		return jsonify({"error": "The model is not ready yet. Please run the training command."}), 503
	except CompanionError as error:
		logging.error("Companion error: %s", error)
		return jsonify({"error": str(error)}), 502
	except Exception:
		logging.exception("Prediction failed")
		return jsonify({"error": "Something went wrong while reflecting on that message."}), 500


@app.post("/reset")
def reset_checkin():
	session.pop("conversation", None)
	session.pop("resource_state", None)
	return jsonify({"conversation": []})


if __name__ == "__main__":
	app.run(debug=True)
