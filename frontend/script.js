
"use strict";

// =============================================
// FraudLens - Frontend / FastAPI integration
// =============================================

// 1. API configuration
const API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_THRESHOLD = 0.5;

// 2. Get references to HTML elements
const transactionForm = document.getElementById("transaction-form");
const analyzeButton = document.getElementById("analyze-button");
const sampleButton = document.getElementById("sample-button");

const apiStatus = document.getElementById("api-status");
const apiStatusIndicator = document.querySelector(".status-indicator");
const thresholdDisplay = document.getElementById("threshold-display");

const formError = document.getElementById("form-error");

const resultPlaceholder = document.getElementById("result-placeholder");
const resultContent = document.getElementById("result-content");
const resultStatus = document.getElementById("result-status");
const resultStatusIcon = document.getElementById("result-status-icon");
const resultLabel = document.getElementById("result-label");

const probabilityValue = document.getElementById("probability-value");
const probabilityTrack = document.getElementById("probability-track");
const probabilityBar = document.getElementById("probability-bar");

const predictionValue = document.getElementById("prediction-value");
const resultThreshold = document.getElementById("result-threshold");
const resultExplanation = document.getElementById("result-explanation");

// 3. Display the configured decision threshold
thresholdDisplay.textContent = `${(DEFAULT_THRESHOLD * 100).toFixed(0)}%`;

// 4. Check whether the FastAPI server is reachable
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (data.status === "ok" && data.model_loaded === true) {
            apiStatus.textContent = "Connected";
            apiStatusIndicator.classList.add("connected");
            apiStatusIndicator.classList.remove("disconnected");
        } else {
            apiStatus.textContent = "Model unavailable";
            apiStatusIndicator.classList.add("disconnected");
            apiStatusIndicator.classList.remove("connected");
        }
    } catch (error) {
        apiStatus.textContent = "Disconnected";
        apiStatusIndicator.classList.add("disconnected");
        apiStatusIndicator.classList.remove("connected");

        console.error("API health check failed:", error);
    }
}

// 5. Show a form error safely
function showFormError(message) {
    formError.textContent = message;
    formError.hidden = false;
}

function clearFormError() {
    formError.textContent = "";
    formError.hidden = true;
}

// 6. Read and validate the transaction form
function getTransactionData() {
    const formData = new FormData(transactionForm);

    const transaction = {
        type: formData.get("type"),
        amount: Number(formData.get("amount")),
        oldbalanceOrg: Number(formData.get("oldbalanceOrg")),
        newbalanceOrig: Number(formData.get("newbalanceOrig")),
        oldbalanceDest: Number(formData.get("oldbalanceDest")),
        newbalanceDest: Number(formData.get("newbalanceDest")),
        balanceDiffOrig: Number(formData.get("balanceDiffOrig")),
        balanceDiffDest: Number(formData.get("balanceDiffDest"))
    };

    const numericFields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "balanceDiffOrig",
        "balanceDiffDest"
    ];

    for (const field of numericFields) {
        if (!Number.isFinite(transaction[field])) {
            throw new Error(`Please enter a valid number for ${field}.`);
        }
    }

    const nonNegativeFields = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ];

    for (const field of nonNegativeFields) {
        if (transaction[field] < 0) {
            throw new Error(`${field} cannot be negative.`);
        }
    }

    return transaction;
}

// 7. Send the transaction to the FastAPI prediction endpoint
async function predictTransaction(transaction) {
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        body: JSON.stringify(transaction)
    });

    const data = await response.json();

    if (!response.ok) {
        const detail = Array.isArray(data.detail)
            ? data.detail.map((item) => item.msg).join(" ")
            : "The API could not process the transaction.";

        throw new Error(detail || `Request failed with HTTP ${response.status}.`);
    }

    return data;
}

// 8. Display the prediction returned by the API
function displayPrediction(data) {
    const isFraud = data.prediction === 1;
    const probability = Number(data.fraud_probability);
    const threshold = Number(data.threshold);

    if (
        !Number.isFinite(probability) ||
        probability < 0 ||
        probability > 1 ||
        !Number.isFinite(threshold) ||
        threshold < 0 ||
        threshold > 1 ||
        (data.prediction !== 0 && data.prediction !== 1)
    ) {
        throw new Error("The API returned an invalid prediction response.");
    }

    const percentage = probability * 100;
    const label = isFraud ? "Fraudulent" : "Legitimate";

    // Switch from the empty state to the results panel.
    resultPlaceholder.hidden = true;
    resultContent.hidden = false;

    // Classification badge.
    resultStatus.classList.toggle("fraud", isFraud);
    resultStatus.classList.toggle("legitimate", !isFraud);

    resultStatusIcon.textContent = isFraud ? "!" : "✓";
    resultLabel.textContent = label;

    // Estimated probability.
    probabilityValue.textContent = `${percentage.toFixed(2)}%`;
    probabilityBar.style.width = `${percentage}%`;
    probabilityBar.classList.toggle("fraud", isFraud);
    probabilityBar.classList.toggle("legitimate", !isFraud);

    probabilityTrack.setAttribute("aria-valuenow", percentage.toFixed(2));

    // Model output details.
    predictionValue.textContent = String(data.prediction);
    resultThreshold.textContent = `${(threshold * 100).toFixed(0)}%`;

    // Explain the classification.
    resultExplanation.textContent = isFraud
        ? `The estimated fraud probability (${percentage.toFixed(2)}%) is at or above the ${(threshold * 100).toFixed(0)}% decision threshold. The model classifies this transaction as potentially fraudulent.`
        : `The estimated fraud probability (${percentage.toFixed(2)}%) is below the ${(threshold * 100).toFixed(0)}% decision threshold. The model classifies this transaction as legitimate.`;
}

// 9. Handle form submission
transactionForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    clearFormError();
    analyzeButton.disabled = true;
    analyzeButton.querySelector("span:first-child").textContent = "Analyzing...";

    try {
        const transaction = getTransactionData();
        const prediction = await predictTransaction(transaction);

        displayPrediction(prediction);
    } catch (error) {
        showFormError(
            `${error.message} Make sure the FastAPI server is running at ${API_BASE_URL}.`
        );

        console.error("Transaction prediction failed:", error);
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.querySelector("span:first-child").textContent =
            "Analyze transaction";
    }
});

// 10. Fill the form with a sample transaction
sampleButton.addEventListener("click", () => {
    transactionForm.elements.type.value = "TRANSFER";
    transactionForm.elements.amount.value = "5000";
    transactionForm.elements.oldbalanceOrg.value = "5000";
    transactionForm.elements.newbalanceOrig.value = "0";
    transactionForm.elements.oldbalanceDest.value = "1000";
    transactionForm.elements.newbalanceDest.value = "6000";
    transactionForm.elements.balanceDiffOrig.value = "5000";
    transactionForm.elements.balanceDiffDest.value = "-5000";

    clearFormError();
});

// 11. Check API status when the page loads
checkApiHealth();
