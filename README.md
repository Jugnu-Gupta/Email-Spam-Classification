# Email Spam Classification

Interactive web interface backed by a Flask API and a distributed Spark inference job for classifying email bodies as spam or not. The UI accepts raw email text, the backend submits the work to a Spark cluster, and the cluster loads a saved TF‑IDF logistic regression model (`spam_classifier_tfidf_logreg.joblib`) to produce a label and confidence score.

## Tech Stack

-   **Frontend:** Vanilla JavaScript, HTML, CSS
-   **Backend API:** Flask (Python)
-   **Distributed inference:** Apache Spark (spark-submit + PySpark)
-   **Model artifact:** `spam_classifier_tfidf_logreg.joblib` trained offline with scikit-learn and stored alongside the project

## Local Development

1. **Clone & enter the project**

    ```cmd
    git clone https://github.com/Jugnu-Gupta/Email-Spam-Classification.git
    cd Email-Spam-Classification
    ```

2. **Create / activate virtual environment** (example on Windows PowerShell):

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

3. **Install dependencies**

    ```powershell
    pip install -r requirements.txt
    ```

4. **Model placement**

    - Place `spam_classifier_tfidf_logreg.joblib` either in the project root or inside `spark_jobs/`.
    - Alternatively set `SPARK_MODEL_PATH` to an absolute path if the artifact lives elsewhere.

5. **Configure Spark connection**
   Set environment variables as needed (examples shown for PowerShell):

    ```powershell
    $env:SPARK_MASTER_URL = "spark://<master-ip>:7077"
    $env:SPARK_SUBMIT_BIN = "C:\spark\bin\spark-submit.cmd"  # if not on PATH
    $env:SPARK_PYTHON = "C:\Users\jugnu\Big_Data\.venv\Scripts\python.exe"
    $env:SPARK_MODEL_PATH = "D:\Big Data\spam_classifier_tfidf_logreg.joblib"
    ```

6. **Run the Flask server**
    ```powershell
    python app.py
    ```
    Visit `http://localhost:5000` (or the host IP) to access the UI.

## Spark Workflow

1. **Client request** – The browser posts email text to `/api/classify`.
2. **Flask API** – Writes the payload to a temporary file, ensures the model exists, and shells out to `spark-submit`:
    ```
    spark-submit --master <SPARK_MASTER_URL> \
    				 --conf spark.pyspark.python=<SPARK_PYTHON> \
    				 --conf spark.pyspark.driver.python=<SPARK_PYTHON> \
    				 --files <MODEL_PATH> \
    				 spark_jobs/classify_job.py <payload_file> <model_filename>
    ```
3. **Spark driver (classify_job.py)**
    - Runs on the cluster, loads the model via joblib.
    - Reads the email content from the temp file.
    - Executes `predict` and `predict_proba` with the logistic regression model.
4. **Result emission** – The Spark job prints a JSON object `{"label": "spam", "confidence": 0.87}` to stdout.
5. **Response to client** – Flask parses the JSON and forwards it to the frontend, which renders the result card.

### Spark Cluster Notes

-   **Master node** coordinates executors; set via `SPARK_MASTER_URL`.
-   **Worker nodes** must have Python, joblib, and scikit-learn available (the `--files` flag distributes the model file, but libraries must already exist on each node).
-   The `SPARK_PYTHON` configuration ensures both driver and executors use the same interpreter (usually the virtualenv python).
-   Logs from `spark-submit` are captured and surfaced in the UI if the job fails, aiding debugging.

## Repository Scripts

-   `app.py` – Flask entry point, Spark invocation logic, and environment resolution.
-   `templates/index.html`, `static/app.js`, `static/styles.css` – Frontend UI and interactivity.
-   `spark_jobs/classify_job.py` – Spark driver script that loads the model and performs inference.

## License
