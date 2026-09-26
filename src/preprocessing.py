"""Shared, lightweight text preprocessing for training and inference."""

import re


def preprocess_text(text: str) -> str:
	"""Normalize text while preserving negation words and useful emphasis."""
	if not isinstance(text, str):
		return ""
	text = text.lower().strip()
	# Keep apostrophes so contractions such as don't and can't remain meaningful.
	text = re.sub(r"[^a-z0-9\s']", " ", text)
	return re.sub(r"\s+", " ", text).strip()
