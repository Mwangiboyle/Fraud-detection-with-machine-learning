# Credit Card Fraud Detection

Live demo: [fraud-detection-ml-demo.netlify.app](https://fraud-detection-ml-demo.netlify.app)

This project uses machine learning to detect fraudulent credit card transactions. It includes three pre-trained models, a one-page web app deployed with Netlify, and the original Streamlit app for local experimentation.

## Live demo architecture

The Netlify site is a single page with a bundled backend:

- `public/` contains the one-page HTML, CSS, and JavaScript interface.
- `netlify/functions/predict.mjs` is a Netlify Function that runs the selected model.
- `netlify/functions/lib/model-data.mjs` contains the model parameters exported from the scikit-learn models.

Netlify Functions do not run Python, so `scripts/export_models.py` converts the trained scikit-learn models into a compact JavaScript artifact. Inference is then performed by the Node function during a request. The frontend and backend deploy together in one Netlify release.

## Models

The following models were trained in `Model_Creation_Notebook.ipynb`:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree

Test-set performance:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 94.88% | 97.10% | 92.46% | 94.72% | 94.86% |
| K-Nearest Neighbors | 98.88% | 98.02% | 99.75% | 98.88% | 98.88% |
| Decision Tree | 99.04% | 98.35% | 99.75% | 99.04% | 99.05% |

## Run the Netlify site locally

Install the Netlify CLI if needed:

```sh
npm install -g netlify-cli
```

Start the site and the function together:

```sh
netlify dev
```

Open `http://localhost:8888`.

The page loads example transactions from the backend, lets you choose a model, edit any of the 30 features, and returns a fraud/legitimate verdict with a fraud probability.

## Regenerate the model artifact

The generated artifact is committed, so deploying does not require Python. Regenerate it only after changing the models or preprocessing:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/export_models.py
```

The export script reproduces the notebook preprocessing, exports the scaler and model parameters, adds preset transactions, and writes:

- `netlify/functions/lib/model-data.mjs`
- `scripts/parity-data.json`

Verify that the JavaScript inference matches scikit-learn:

```sh
node scripts/parity_test.mjs
```

## Deploy to Netlify

### Option 1: Netlify CLI

```sh
netlify login
netlify init
netlify deploy --prod
```

### Option 2: Connect the Git repository

Import the repository in the Netlify dashboard. Netlify reads `netlify.toml`, publishes `public/`, and bundles `netlify/functions/` automatically. No separate backend deployment is needed.

## Streamlit app (local only)

The original Streamlit interface is still available for local use:

```sh
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501`.

## Project structure

```text
data/                         Dataset used by the notebook and export script
model/                        Original scikit-learn model files
public/                       Netlify one-page frontend
netlify/functions/            Netlify Function API and inference engine
netlify/functions/lib/        Exported model artifact
scripts/export_models.py      Converts sklearn models to JavaScript data
scripts/parity_test.mjs       Checks JavaScript inference against sklearn
app.py                        Original Streamlit app
Model_Creation_Notebook.ipynb Training and evaluation notebook
netlify.toml                  Netlify build, functions, and header config
```
