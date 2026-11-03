import modelData from "./model-data.mjs";

export const FEATURES = modelData.features;
export const THRESHOLD = 0.5;

export class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

function finiteNumber(value, feature) {
  if (value === "" || value === null || value === undefined || typeof value === "boolean") {
    throw new ValidationError(`${feature} must be a number.`);
  }
  const number = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(number)) {
    throw new ValidationError(`${feature} must be a finite number.`);
  }
  return number;
}

function normalizeFeatures(input) {
  if (Array.isArray(input)) {
    if (input.length !== FEATURES.length) {
      throw new ValidationError(`Expected ${FEATURES.length} feature values.`);
    }
    return input.map((value, index) => finiteNumber(value, FEATURES[index]));
  }

  if (!input || typeof input !== "object") {
    throw new ValidationError("features must be an object or an array.");
  }

  return FEATURES.map((feature) => {
    if (!(feature in input)) {
      throw new ValidationError(`Missing feature: ${feature}.`);
    }
    return finiteNumber(input[feature], feature);
  });
}

function scaleFeatures(values) {
  return values.map(
    (value, index) => (value - modelData.scaler.mean[index]) / modelData.scaler.scale[index]
  );
}

function sigmoid(value) {
  if (value >= 0) {
    return 1 / (1 + Math.exp(-value));
  }
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function logisticProbability(scaled) {
  const model = modelData.models.logistic;
  let score = model.intercept;
  for (let index = 0; index < scaled.length; index += 1) {
    score += model.coef[index] * scaled[index];
  }
  return sigmoid(score);
}

function treeProbability(scaled) {
  const model = modelData.models.tree;
  let node = 0;
  while (model.childrenLeft[node] !== -1) {
    const feature = model.feature[node];
    node =
      scaled[feature] <= model.threshold[node] ? model.childrenLeft[node] : model.childrenRight[node];
  }
  const [legitimate, fraud] = model.value[node];
  const total = legitimate + fraud;
  return total === 0 ? 0 : fraud / total;
}

function knnProbability(scaled) {
  const model = modelData.models.knn;
  const { fitX, labels, nNeighbors } = model;
  const distances = new Float64Array(fitX.length);

  for (let row = 0; row < fitX.length; row += 1) {
    const trainingRow = fitX[row];
    let distance = 0;
    for (let column = 0; column < scaled.length; column += 1) {
      distance += Math.abs(scaled[column] - trainingRow[column]);
    }
    distances[row] = distance;
  }

  const order = Array.from({ length: fitX.length }, (_, index) => index);
  order.sort((left, right) => distances[left] - distances[right]);
  const neighbors = order.slice(0, nNeighbors);
  const hasExactMatch = neighbors.some((index) => distances[index] === 0);

  let legitimateWeight = 0;
  let fraudWeight = 0;
  for (const index of neighbors) {
    const distance = distances[index];
    const weight = hasExactMatch ? (distance === 0 ? 1 : 0) : 1 / distance;
    if (labels[index] === 1) {
      fraudWeight += weight;
    } else {
      legitimateWeight += weight;
    }
  }

  const total = legitimateWeight + fraudWeight;
  return total === 0 ? 0 : fraudWeight / total;
}

export function getMetadata() {
  return {
    features: FEATURES,
    threshold: THRESHOLD,
    modelInfo: modelData.modelInfo,
    metrics: modelData.metrics,
    samples: modelData.samples.map(({ expected, ...sample }) => sample),
  };
}

export function predict(modelId, input) {
  if (!Object.hasOwn(modelData.models, modelId)) {
    throw new ValidationError(`Unknown model: ${modelId}.`);
  }

  const values = normalizeFeatures(input);
  const scaled = scaleFeatures(values);
  const probabilityByModel = {
    logistic: logisticProbability,
    knn: knnProbability,
    tree: treeProbability,
  };
  const fraudProbability = Math.min(
    1,
    Math.max(0, probabilityByModel[modelId](scaled))
  );
  const prediction = fraudProbability > THRESHOLD ? 1 : 0;

  return {
    model: modelId,
    modelName: modelData.modelInfo[modelId].name,
    prediction,
    label: prediction === 1 ? "Fraud" : "Legitimate",
    fraudProbability,
    legitimateProbability: 1 - fraudProbability,
    threshold: THRESHOLD,
    metrics: modelData.metrics[modelId],
  };
}
