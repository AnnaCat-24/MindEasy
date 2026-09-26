"""Train, compare, evaluate, and persist the MindEasy sentiment model."""

from pathlib import Path
import sys
import struct
import zlib

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
							 precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing import preprocess_text


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"
EVALUATION_DIR = ROOT / "evaluation"


def save_confusion_png(matrix, path: Path) -> None:
	"""Write a tiny dependency-free heatmap so evaluation is reproducible everywhere."""
	size, cell = 180, 60
	pixels = bytearray()
	maximum = max(1, int(matrix.max()))
	for y in range(size):
		pixels.append(0)
		for x in range(size):
			row, column = min(y // cell, 2), min(x // cell, 2)
			value = int(matrix[row, column])
			shade = 245 - int(150 * value / maximum)
			pixels.extend((shade, min(255, shade + 10), min(255, shade + 5), 255))
	def chunk(kind, data):
		return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
	raw = zlib.compress(bytes(pixels))
	png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
	path.write_bytes(png)


def train() -> None:
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
	data = pd.read_csv(DATA_PATH).dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
	if set(data["label"]) != {"positive", "neutral", "negative"}:
		raise ValueError("Dataset must contain positive, neutral, and negative labels.")
	data["text"] = data["text"].map(preprocess_text)
	train_text, test_text, train_labels, test_labels = train_test_split(
		data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
	)
	vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
	train_features = vectorizer.fit_transform(train_text)
	test_features = vectorizer.transform(test_text)
	candidates = {
		"logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
		"multinomial_naive_bayes": MultinomialNB(),
		"linear_svm": LinearSVC(random_state=42),
	}
	scores = {}
	for name, candidate in candidates.items():
		candidate.fit(train_features, train_labels)
		predicted = candidate.predict(test_features)
		precision, recall, f1, _ = precision_recall_fscore_support(
			test_labels, predicted, average="weighted", zero_division=0
		)
		scores[name] = {"accuracy": accuracy_score(test_labels, predicted), "precision": precision, "recall": recall, "f1": f1}
	best_name = max(scores, key=lambda name: scores[name]["f1"])
	final_model = candidates[best_name]
	final_predictions = final_model.predict(test_features)
	MODEL_DIR.mkdir(exist_ok=True)
	EVALUATION_DIR.mkdir(exist_ok=True)
	joblib.dump(final_model, MODEL_DIR / "model.pkl")
	joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
	matrix = confusion_matrix(test_labels, final_predictions, labels=["positive", "neutral", "negative"])
	report = ["MindEasy sentiment evaluation", "=" * 30, f"Dataset size: {len(data)}", f"Class distribution: {data['label'].value_counts().to_dict()}", f"Train/test size: {len(train_text)}/{len(test_text)}", f"Selected model: {best_name}", "", "Candidate model scores:"]
	report.extend(f"{name}: accuracy={value['accuracy']:.3f}, precision={value['precision']:.3f}, recall={value['recall']:.3f}, f1={value['f1']:.3f}" for name, value in scores.items())
	report.extend(["", "Classification report:", classification_report(test_labels, final_predictions, zero_division=0), "Confusion matrix labels: positive, neutral, negative", str(matrix), "", "Manual error analysis: short, sarcastic, mixed, and context-dependent messages can be difficult for TF-IDF models. Results are sentiment labels, not clinical assessments."])
	(EVALUATION_DIR / "evaluation_report.txt").write_text("\n".join(report), encoding="utf-8")
	try:
		import matplotlib.pyplot as plt
		from sklearn.metrics import ConfusionMatrixDisplay
		ConfusionMatrixDisplay(matrix, display_labels=["positive", "neutral", "negative"]).plot(cmap="GnBu")
		plt.tight_layout()
		plt.savefig(EVALUATION_DIR / "confusion_matrix.png", dpi=160)
		plt.close()
	except ImportError:
		save_confusion_png(matrix, EVALUATION_DIR / "confusion_matrix.png")
		print("matplotlib is unavailable; wrote a dependency-free confusion matrix image.")
	print("\n".join(report))


if __name__ == "__main__":
	train()
