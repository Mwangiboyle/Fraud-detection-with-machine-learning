const FEATURES = ["Time", ...Array.from({ length: 28 }, (_, index) => `V${index + 1}`), "Amount"];
const API_ENDPOINTS = ["/api/predict", "/.netlify/functions/predict"];
const MODEL_DESCRIPTIONS = {
  logistic: "Linear model that estimates fraud probability from a weighted sum of the scaled features.",
  knn: "Finds the 3 nearest historical transactions using Manhattan distance and votes with distance weighting.",
  tree: "Decision tree that splits the scaled features into fraud and legitimate leaves.",
};
const METRIC_LABELS = [
  ["Accuracy", "accuracy"],
  ["Precision", "precision"],
  ["Recall", "recall"],
  ["F1 score", "f1"],
  ["ROC AUC", "rocAuc"],
];

const state = {
  metadata: null,
  activeSample: null,
  endpoint: API_ENDPOINTS[0],
};

const elements = {
  form: document.querySelector("#prediction-form"),
  modelSelect: document.querySelector("#model-select"),
  modelDescription: document.querySelector("#model-description"),
  sampleSelect: document.querySelector("#sample-select"),
  sampleNote: document.querySelector("#sample-note"),
  featureGrid: document.querySelector("#feature-grid"),
  resetButton: document.querySelector("#reset-button"),
  predictButton: document.querySelector("#predict-button"),
  buttonLabel: document.querySelector("#button-label"),
  spinner: document.querySelector("#spinner"),
  errorMessage: document.querySelector("#error-message"),
  resultCard: document.querySelector(".result-card"),
  result: document.querySelector("#result"),
  resultPlaceholder: document.querySelector("#result-placeholder"),
  resultModel: document.querySelector("#result-model"),
  verdict: document.querySelector("#verdict"),
  probabilityValue: document.querySelector("#probability-value"),
  probabilityFill: document.querySelector("#probability-fill"),
  resultCopy: document.querySelector("#result-copy"),
  metricsList: document.querySelector("#metrics-list"),
};

function buildFeatureInputs() {
  const fragment = document.createDocumentFragment();
  for (const feature of FEATURES.slice(1, -1)) {
    const group = document.createElement("div");
    group.className = "field-group";

    const label = document.createElement("label");
    label.htmlFor = feature;
    label.textContent = feature;

    const input = document.createElement("input");
    input.id = feature;
    input.name = feature;
    input.type = "number";
    input.step = "any";
    input.inputMode = "decimal";
    input.value = "0";
    input.defaultValue = "0";
    input.required = true;

    group.append(label, input);
    fragment.append(group);
  }
  elements.featureGrid.append(fragment);
}

async function apiRequest(options = {}) {
  let lastError = null;
  for (const endpoint of API_ENDPOINTS) {
    try {
      const response = await fetch(endpoint, options);
      if (response.status === 404) {
        continue;
      }
      state.endpoint = endpoint;
      return response;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError ?? new Error("Could not reach the prediction service.");
}

function formatPercent(value) {
  const percentage = value * 100;
  if (percentage > 0 && percentage < 0.01) {
    return "<0.01%";
  }
  return `${percentage.toFixed(percentage >= 99.995 ? 0 : 2)}%`;
}

function updateModelDescription() {
  const info = state.metadata?.modelInfo?.[elements.modelSelect.value];
  elements.modelDescription.textContent =
    info?.description ?? MODEL_DESCRIPTIONS[elements.modelSelect.value];
}

function populateSamples() {
  const samples = state.metadata?.samples ?? [];
  if (samples.length === 0) {
    elements.sampleSelect.disabled = true;
    return;
  }

  const legitimate = document.createElement("optgroup");
  legitimate.label = "Legitimate transactions";
  const fraudulent = document.createElement("optgroup");
  fraudulent.label = "Fraudulent transactions";

  for (const sample of samples) {
    const option = document.createElement("option");
    option.value = sample.id;
    option.textContent = sample.title;
    (sample.trueClass === 1 ? fraudulent : legitimate).append(option);
  }

  elements.sampleSelect.append(legitimate, fraudulent);
  elements.sampleSelect.disabled = legitimate.children.length + fraudulent.children.length === 0;
}

function applySample(sampleId) {
  const sample = state.metadata?.samples?.find((item) => item.id === sampleId);
  if (!sample) {
    state.activeSample = null;
    elements.sampleNote.hidden = true;
    return;
  }

  for (const feature of FEATURES) {
    document.getElementById(feature).value = String(sample.features[feature]);
  }

  state.activeSample = sample;
  elements.sampleNote.textContent = `${sample.title}. True label: ${sample.label}.`;
  elements.sampleNote.hidden = false;
  resetResult(
    "Example loaded",
    "Run the analysis to see how the selected model classifies this transaction."
  );
}

function resetResult(title = "Ready when you are", copy = "Run an analysis to see the selected model's verdict and fraud probability.") {
  elements.result.hidden = true;
  elements.resultPlaceholder.hidden = false;
  elements.resultPlaceholder.querySelector("h2").textContent = title;
  elements.resultPlaceholder.querySelector("p").textContent = copy;
  elements.probabilityFill.style.width = "0%";
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.hidden = false;
}

function clearError() {
  elements.errorMessage.hidden = true;
  elements.errorMessage.textContent = "";
}

function setLoading(isLoading) {
  elements.predictButton.disabled = isLoading;
  elements.spinner.hidden = !isLoading;
  elements.buttonLabel.textContent = isLoading ? "Analyzing..." : "Analyze transaction";
}

function collectFeatures() {
  const features = {};
  for (const feature of FEATURES) {
    const input = document.getElementById(feature);
    const rawValue = input.value.trim();
    if (rawValue === "") {
      throw new Error(`${feature} is required.`);
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) {
      throw new Error(`${feature} must be a valid number.`);
    }
    features[feature] = value;
  }
  return features;
}

function renderMetrics(metrics) {
  elements.metricsList.replaceChildren();
  for (const [label, key] of METRIC_LABELS) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    term.textContent = label;
    const value = document.createElement("dd");
    value.textContent = formatPercent(metrics[key]);
    row.append(term, value);
    elements.metricsList.append(row);
  }
}

function renderResult(result) {
  elements.resultPlaceholder.hidden = true;
  elements.result.hidden = false;

  elements.resultModel.textContent = result.modelName;
  elements.verdict.textContent = result.label;
  elements.verdict.className = `verdict ${result.prediction === 1 ? "fraud" : "legit"}`;
  elements.probabilityValue.textContent = formatPercent(result.fraudProbability);
  elements.probabilityFill.style.width = `${Math.min(100, Math.max(0, result.fraudProbability * 100))}%`;
  const outcome = result.prediction === 1 ? "fraudulent" : "legitimate";
  elements.resultCopy.textContent = state.activeSample
    ? `Preset true label: ${state.activeSample.label}. This model classifies the transaction as ${outcome}.`
    : `This model classifies the transaction as ${outcome}.`;
  renderMetrics(result.metrics);

  if (window.matchMedia("(max-width: 900px)").matches) {
    elements.resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  clearError();

  let features;
  try {
    features = collectFeatures();
  } catch (error) {
    showError(error.message);
    return;
  }

  setLoading(true);
  try {
    const response = await apiRequest({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: elements.modelSelect.value, features }),
    });

    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.error ?? `Prediction failed with status ${response.status}.`);
    }
    renderResult(data);
  } catch (error) {
    const message =
      error.message === "Failed to fetch"
        ? "Could not reach the prediction service. Run the site with `netlify dev` locally."
        : error.message;
    showError(message);
  } finally {
    setLoading(false);
  }
}

function handleReset() {
  elements.form.reset();
  state.activeSample = null;
  elements.sampleSelect.value = "";
  elements.sampleNote.hidden = true;
  clearError();
  updateModelDescription();
  resetResult();
}

function handleModelChange() {
  updateModelDescription();
  if (!elements.result.hidden) {
    resetResult(
      "Model changed",
      "Analyze the transaction again to see results from the newly selected model."
    );
  }
}

function handleFormInput(event) {
  if (!event.target.matches("input")) {
    return;
  }
  if (state.activeSample) {
    state.activeSample = null;
    elements.sampleSelect.value = "";
    elements.sampleNote.hidden = true;
  }
  if (!elements.result.hidden) {
    resetResult(
      "Values changed",
      "Analyze the transaction again to see a result for the updated values."
    );
  }
}

async function loadMetadata() {
  try {
    const response = await apiRequest();
    if (!response.ok) {
      throw new Error(`Metadata request failed with status ${response.status}.`);
    }
    state.metadata = await response.json();
    populateSamples();
    updateModelDescription();
  } catch (error) {
    console.error(error);
    elements.sampleSelect.disabled = true;
    elements.sampleSelect.options[0].textContent = "Examples unavailable while the backend is offline";
    updateModelDescription();
  }
}

buildFeatureInputs();
updateModelDescription();
loadMetadata();

elements.form.addEventListener("submit", handleSubmit);
elements.form.addEventListener("input", handleFormInput);
elements.resetButton.addEventListener("click", handleReset);
elements.modelSelect.addEventListener("change", handleModelChange);
elements.sampleSelect.addEventListener("change", (event) => applySample(event.target.value));
