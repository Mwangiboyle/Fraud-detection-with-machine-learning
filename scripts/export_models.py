import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.tree._tree as tree_module
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "netlify" / "functions" / "lib"
OUTPUT_PATH = OUTPUT_DIR / "model-data.mjs"
PARITY_PATH = ROOT / "scripts" / "parity-data.json"

FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODEL_NAMES = {
    "logistic": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "tree": "Decision Tree",
}
MODEL_FILES = {
    "logistic": "Logistic_Regression.joblib",
    "knn": "K-Nearest_Neighbors.joblib",
    "tree": "Decision_Tree.joblib",
}
MODEL_DESCRIPTIONS = {
    "logistic": "Linear model that estimates fraud probability from a weighted sum of the scaled features.",
    "knn": "Finds the 3 nearest historical transactions using Manhattan distance and votes with distance weighting.",
    "tree": "Decision tree that splits the scaled features into fraud and legitimate leaves.",
}
SAMPLES = [
    (71744, "Legitimate transaction", "Clear legitimate example"),
    (80880, "Legitimate transaction", "Borderline legitimate example"),
    (541, "Fraudulent transaction", "Clear fraud example"),
    (4920, "Fraudulent transaction", "Fraud example with high confidence"),
    (623, "Fraudulent transaction", "Fraud caught by KNN and the tree"),
    (10498, "Fraudulent transaction", "Hard case: Logistic Regression misses it"),
]

_original_check = tree_module._check_node_ndarray


def _check_node_ndarray(node_ndarray, expected_dtype):
    if node_ndarray.dtype.names and "missing_go_to_left" not in node_ndarray.dtype.names:
        upgraded = np.zeros(node_ndarray.shape, dtype=expected_dtype)
        for name in node_ndarray.dtype.names:
            upgraded[name] = node_ndarray[name]
        node_ndarray = upgraded
    return _original_check(node_ndarray, expected_dtype)


tree_module._check_node_ndarray = _check_node_ndarray


def load_model(name):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def round_list(values, digits=8):
    return [round(float(value), digits) for value in values]


def build_scaler():
    data = pd.read_csv(DATA_PATH)
    sampled = pd.concat(
        [
            data[data["class"] == label].sample(n=4000, replace=True, random_state=1)
            for label in (0, 1)
        ]
    ).reset_index(drop=True)
    features = sampled.drop("class", axis=1)
    labels = sampled["class"]
    resampled_features, resampled_labels = SMOTE(random_state=42).fit_resample(features, labels)
    scaler = StandardScaler().fit(resampled_features)
    return data, scaler, resampled_features, resampled_labels


def evaluate(model, x_test, y_test):
    predictions = model.predict(x_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
        "precision": round(float(precision_score(y_test, predictions)), 6),
        "recall": round(float(recall_score(y_test, predictions)), 6),
        "f1": round(float(f1_score(y_test, predictions)), 6),
        "rocAuc": round(float(roc_auc_score(y_test, predictions)), 6),
    }


def model_probabilities(name, model, scaled):
    if name == "tree":
        leaves = model.apply(scaled)
        values = model.tree_.value[leaves][:, 0, :]
        return values / values.sum(axis=1, keepdims=True)
    return model.predict_proba(scaled)


def export():
    data, scaler, resampled_features, resampled_labels = build_scaler()
    scaled = scaler.transform(resampled_features)
    x_train, x_test, y_train, y_test = train_test_split(
        scaled, resampled_labels, test_size=0.3, random_state=42
    )

    loaded = {name: load_model(name) for name in MODEL_NAMES}
    for model in loaded.values():
        assert list(model.classes_) == [0, 1], "Expected class order [0, 1]."
    assert loaded["knn"].metric == "manhattan", "KNN metric changed; update inference.mjs."
    assert loaded["knn"].weights == "distance", "KNN weights changed; update inference.mjs."
    if not np.allclose(
        np.sort(x_train, axis=0), np.sort(loaded["knn"]._fit_X, axis=0), atol=1e-8
    ):
        raise RuntimeError(
            "The reproduced scaler does not match the KNN training data. "
            "Check the pandas/NumPy versions before exporting."
        )
    metrics = {
        name: evaluate(model, x_test, y_test) for name, model in loaded.items()
    }

    full_scaled = scaler.transform(data[FEATURES])
    parity_rows = data.sample(n=20, random_state=123)
    parity_scaled = scaler.transform(parity_rows[FEATURES])
    parity_expected = []
    for row_index, row, scaled_row in zip(
        parity_rows.index, parity_rows.itertuples(index=False), parity_scaled
    ):
        features = {feature: round(float(getattr(row, feature)), 8) for feature in FEATURES}
        expected = {
            "index": int(row_index),
            "trueClass": int(row._asdict()["_30"]),
            "features": features,
        }
        expected["models"] = {}
        for name, model in loaded.items():
            probabilities = model_probabilities(name, model, scaled_row.reshape(1, -1))[0]
            expected["models"][name] = {
                "prediction": int(model.predict(scaled_row.reshape(1, -1))[0]),
                "fraudProbability": round(float(probabilities[1]), 8),
            }
        parity_expected.append(expected)

    samples = []
    for index, label, title in SAMPLES:
        row = data.loc[index]
        expected = {}
        for name, model in loaded.items():
            probabilities = model_probabilities(name, model, full_scaled[index].reshape(1, -1))[0]
            expected[name] = {
                "prediction": int(model.predict(full_scaled[index].reshape(1, -1))[0]),
                "fraudProbability": round(float(probabilities[1]), 8),
            }
        samples.append(
            {
                "id": f"sample-{index}",
                "index": int(index),
                "label": label,
                "title": title,
                "trueClass": int(row["class"]),
                "features": {feature: round(float(row[feature]), 8) for feature in FEATURES},
                "expected": expected,
            }
        )

    logistic = loaded["logistic"]
    knn = loaded["knn"]
    tree = loaded["tree"]

    model_data = {
        "features": FEATURES,
        "modelInfo": {
            name: {"name": MODEL_NAMES[name], "description": MODEL_DESCRIPTIONS[name]}
            for name in MODEL_NAMES
        },
        "scaler": {
            "mean": round_list(scaler.mean_),
            "scale": round_list(scaler.scale_),
        },
        "models": {
            "logistic": {
                "coef": round_list(logistic.coef_[0]),
                "intercept": round(float(logistic.intercept_[0]), 8),
                "classes": [int(value) for value in logistic.classes_],
            },
            "knn": {
                "nNeighbors": int(knn.n_neighbors),
                "metric": knn.metric,
                "weights": knn.weights,
                "fitX": [round_list(row) for row in knn._fit_X],
                "labels": [int(value) for value in knn._y],
                "classes": [int(value) for value in knn.classes_],
            },
            "tree": {
                "childrenLeft": [int(value) for value in tree.tree_.children_left],
                "childrenRight": [int(value) for value in tree.tree_.children_right],
                "feature": [int(value) for value in tree.tree_.feature],
                "threshold": round_list(tree.tree_.threshold),
                "value": [
                    [int(class_zero), int(class_one)]
                    for class_zero, class_one in tree.tree_.value[:, 0, :]
                ],
                "classes": [int(value) for value in tree.classes_],
            },
        },
        "metrics": metrics,
        "samples": samples,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        "export default " + json.dumps(model_data, separators=(",", ":")) + ";\n"
    )
    PARITY_PATH.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "scaler": model_data["scaler"],
                "rows": parity_expected,
                "samples": samples,
            },
            indent=2,
        )
    )

    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Wrote {PARITY_PATH}")
    for name, values in metrics.items():
        print(name, values)


if __name__ == "__main__":
    export()
