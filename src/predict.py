"""Sentiment prediction using Hugging Face Inference API."""

import os
from huggingface_hub import InferenceClient

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Initialize HF InferenceClient using environment token
client = InferenceClient(
    token=os.environ.get("HF_TOKEN"),
    model=MODEL_NAME,
)


def predict_sentiment(text: str) -> tuple[str, float | None]:
    """Return a sentiment label and model confidence."""

    if not text or not text.strip():
        return "neutral", None

    try:
        # Calls Hugging Face Inference API over HTTP (uses ~0 local RAM)
        results = client.text_classification(text.strip())

        # Response structure: [{'label': 'positive', 'score': 0.95}, ...]
        if isinstance(results, list) and len(results) > 0:
            top_result = results[0]
            label = str(top_result["label"]).lower()
            confidence = float(top_result["score"])
        else:
            return "neutral", 0.50

        # Normalize labels in case the model returns LABEL_0/LABEL_1/LABEL_2
        label_mapping = {
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }

        label = label_mapping.get(label, label)

        # Make sure the application only receives the three expected labels
        if label not in {"positive", "neutral", "negative"}:
            raise ValueError(f"Unexpected sentiment label from model: {label}")

        return label, confidence

    except Exception:
        # Fallback graceful prediction if API fails or token isn't provided yet
        return "neutral", 0.50