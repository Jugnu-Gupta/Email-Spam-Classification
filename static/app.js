const classifyBtn = document.getElementById("classifyBtn");
const emailTextEl = document.getElementById("emailText");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const resultLabelEl = document.getElementById("resultLabel");
const resultConfidenceEl = document.getElementById("resultConfidence");
const resultMessageEl = document.getElementById("resultMessage");
const sampleButtons = document.querySelectorAll("[data-sample]");
const defaultButtonLabel = classifyBtn.textContent;

function setLoading(isLoading) {
    classifyBtn.disabled = isLoading;
    classifyBtn.dataset.loading = isLoading ? "true" : "false";
    classifyBtn.setAttribute("aria-busy", isLoading ? "true" : "false");
    classifyBtn.textContent = isLoading ? "Classifying..." : defaultButtonLabel;
}

function setStatus(message, type = "info") {
    if (!message) {
        statusEl.hidden = true;
        statusEl.textContent = "";
        delete statusEl.dataset.state;
        return;
    }

    statusEl.hidden = false;
    statusEl.textContent = message;
    statusEl.dataset.state = type;
}

function formatConfidence(value) {
    if (!Number.isFinite(value)) {
        return "N/A";
    }
    const normalized = value > 1 ? value : value * 100;
    return `${Math.round(normalized)}%`;
}

function getResultMessage(state) {
    if (state === "spam") {
        return "Treat this message with caution. Route it for review before trusting links or attachments.";
    }
    if (state === "not-spam") {
        return "No major spam indicators detected. Still verify unexpected requests before responding.";
    }
    return "Model response received. Review the message contents before taking action.";
}

function updateResult(payload) {
    const rawLabel = String(payload.label ?? "").trim().toLowerCase();
    let datasetState = "unknown";

    if (rawLabel.includes("spam") || rawLabel.includes("phish") || rawLabel.includes("junk")) {
        datasetState = "spam";
    } else if (rawLabel.length > 0) {
        datasetState = "not-spam";
    }

    const displayLabel = datasetState === "spam"
        ? "Spam"
        : datasetState === "not-spam"
            ? "Not Spam"
            : payload.label || "Unknown";

    resultLabelEl.textContent = displayLabel;
    resultEl.dataset.state = datasetState;
    resultConfidenceEl.textContent = formatConfidence(payload.confidence);

    if (resultMessageEl) {
        resultMessageEl.textContent = getResultMessage(datasetState);
    }

    resultEl.hidden = false;
    resultEl.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function classifyEmail() {
    const emailText = emailTextEl.value.trim();

    if (!emailText) {
        setStatus("Please add email content before running a classification.", "warning");
        resultEl.hidden = true;
        return;
    }

    setLoading(true);
    setStatus("Running classification...", "info");
    resultEl.hidden = true;

    try {
        const response = await fetch("/api/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ emailText })
        });

        if (!response.ok) {
            const errorPayload = await response.json().catch(() => ({}));
            const message = errorPayload.error || "An unexpected error occurred.";
            setStatus(message, "error");
            return;
        }

        const payload = await response.json().catch(() => null);

        if (!payload) {
            setStatus("The service returned an unreadable response.", "error");
            resultEl.hidden = true;
            return;
        }

        if (payload.error) {
            const pieces = [payload.error, payload.details, payload.stdout, payload.command]
                .filter(Boolean)
                .map((part) => (typeof part === "string" ? part : JSON.stringify(part)));
            setStatus(pieces.join(" — "), "error");
            resultEl.hidden = true;
            return;
        }

        updateResult(payload);
        setStatus("Classification complete.", "success");
    } catch (error) {
        console.error("Failed to classify email", error);
        setStatus("Unable to reach the classification service.", "error");
    } finally {
        setLoading(false);
    }
}

classifyBtn.addEventListener("click", classifyEmail);

emailTextEl.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "enter") {
        event.preventDefault();
        classifyEmail();
    }
});

sampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
        const sampleText = button.getAttribute("data-sample") || "";
        emailTextEl.value = sampleText;
        emailTextEl.focus();
        setStatus(`Loaded sample: ${button.textContent}`, "info");
        resultEl.hidden = true;
    });
});

