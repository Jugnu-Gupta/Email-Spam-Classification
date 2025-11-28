"""PySpark job entry point that loads a saved model and classifies an email payload."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
from pyspark.sql import SparkSession

_MODEL_CACHE: Dict[str, object] = {}


def load_email_text(input_path: Path) -> str:
    if not input_path.exists():
        raise FileNotFoundError(f"Payload file not found: {input_path}")
    return input_path.read_text(encoding="utf-8")


def load_model(model_path: Path):
    cached = _MODEL_CACHE.get(str(model_path))
    if cached is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        cached = joblib.load(model_path)
        _MODEL_CACHE[str(model_path)] = cached
    return cached


def classify_email(model, email_text: str) -> Dict[str, object]:
    prediction = model.predict([email_text])[0]
    probabilities = model.predict_proba([email_text])[0]
    confidence = float(max(probabilities))
    return {
        "label": str(prediction),
        "confidence": round(confidence, 4),
    }


def main() -> None:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Payload path and model path are required"}), flush=True)
        return

    payload_path = Path(sys.argv[1])
    model_path = Path("D:/Big Data/spark_jobs/spam_classifier_tfidf_logreg.joblib")

    spark = SparkSession.builder.appName("EmailClassifier").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        email_text = load_email_text(payload_path)
        model = load_model(model_path)
        result = classify_email(model, email_text)
        print(json.dumps(result), flush=True)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}), flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
