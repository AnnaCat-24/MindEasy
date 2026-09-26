"""Sentiment prediction using a pretrained RoBERTa model."""

from transformers import pipeline


# Pretrained sentiment model.
# It predicts: negative, neutral, or positive.
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# Load the model once when this module is imported.
# This is much faster than loading it for every prediction.
sentiment_model = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
)


def predict_sentiment(text: str) -> tuple[str, float | None]:
    """Return a sentiment label and model confidence."""

    if not text or not text.strip():
        return "neutral", None

    result = sentiment_model(text.strip(), truncation=True)[0]

    label = str(result["label"]).lower()
    confidence = float(result["score"])

    # Normalize labels in case the model returns LABEL_0/LABEL_1/LABEL_2.
    label_mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
    }

    label = label_mapping.get(label, label)

    # Make sure the application only receives the three labels
    # that MindEasy already expects.
    if label not in {"positive", "neutral", "negative"}:
        raise ValueError(f"Unexpected sentiment label from model: {label}")

    return label, confidence