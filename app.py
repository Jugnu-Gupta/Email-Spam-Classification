from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, template_folder="templates", static_folder="static")

SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://192.168.0.175:7077")
# SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "spark://10.74.77.162:7077")
SPARK_SUBMIT_BIN = os.getenv("SPARK_SUBMIT_BIN", "spark-submit.cmd" if os.name == "nt" else "spark-submit")
PYTHON_PATH = os.getenv("SPARK_PYTHON", r"C:\\Users\\jugnu\\Big_Data\\.venv\\Scripts\\python.exe")
SPARK_JOB_PATH = str((Path(__file__).parent / "spark_jobs" / "classify_job.py").resolve())

MODEL_PATH = "D:/Big Data/spark_jobs/spam_classifier_tfidf_logreg.joblib"

@app.route("/", methods=["GET"])
def index() -> str:
    return render_template("index.html")


def run_spark_classification(email_text: str) -> Dict[str, Any]:
    """Submit the email text to the Spark cluster and return its response."""

    payload_path: str | None = None
    result: subprocess.CompletedProcess[str] | None = None
    try:
        spark_submit_path: str | None
        configured_path = Path(SPARK_SUBMIT_BIN)
        if configured_path.exists():
            spark_submit_path = str(configured_path)
        else:
            spark_submit_path = shutil.which(SPARK_SUBMIT_BIN)

        if not spark_submit_path:
            return {
                "error": "spark-submit not found",
                "details": (
                    "Set SPARK_SUBMIT_BIN to the full path of spark-submit or add it to the PATH."
                ),
                "status_code": 500,
            }

        if not Path(PYTHON_PATH).exists():
            return {
                "error": "Python interpreter for Spark not found",
                "details": (
                    "Update SPARK_PYTHON to point at the Python installed on both driver and executors."
                ),
                "status_code": 500,
            }

        model_path = Path(MODEL_PATH)
        if not model_path.exists():
            return {
                "error": "Model artifact not found",
                "details": f"Expected model at {model_path}",
                "status_code": 500,
            }

        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".txt") as handle:
            handle.write(email_text)
            payload_path = handle.name

        command = [
            spark_submit_path,
            "--master",
            SPARK_MASTER_URL,
            "--conf",
            f"spark.pyspark.python={PYTHON_PATH}",
            "--conf",
            f"spark.pyspark.driver.python={PYTHON_PATH}",
            "--files",
            str(model_path),
            SPARK_JOB_PATH,
            payload_path,
            model_path.name,
        ]
        command = [str(part) for part in command]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=False,
            check=False,
        )
    finally:
        if payload_path:
            try:
                Path(payload_path).unlink(missing_ok=True)
            except OSError:
                pass

    if result is None:
        return {
            "error": "Failed to invoke spark-submit",
            "details": "spark-submit did not return a result.",
            "status_code": 500,
        }

    if result.returncode != 0:
        return {
            "error": "Spark job failed",
            "details": result.stderr.strip() or "Spark returned non-zero status.",
            "stdout": result.stdout.strip(),
            "command": " ".join(command),
            "status_code": 502,
        }

    try:
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        if not output_lines:
            raise ValueError("No output from Spark job")
        payload = json.loads(output_lines[-1])
        if not isinstance(payload, dict):
            raise ValueError("Unexpected JSON payload")
        return payload
    except json.JSONDecodeError as exc:
        return {
            "error": "Invalid JSON response from Spark",
            "raw": result.stdout,
            "details": str(exc),
            "command": " ".join(command),
            "status_code": 502,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": "Failed to parse Spark output",
            "raw": result.stdout,
            "details": str(exc),
            "command": " ".join(command),
            "status_code": 502,
        }


@app.route("/api/classify", methods=["POST"])
def api_classify() -> Any:
    payload = request.get_json(silent=True) or {}
    email_text = (payload.get("emailText") or "").strip()

    if not email_text:
        return jsonify({"error": "Email text is required."}), 400

    result = run_spark_classification(email_text)

    if isinstance(result, dict) and result.get("error"):
        status_code = int(result.get("status_code") or 500)
        return jsonify(result), status_code

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

