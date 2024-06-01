# Credit Card Fraud Detection

This project uses machine learning algorithms to detect fraudulent credit card transactions. It includes pre-trained models and a Streamlit app for user interaction.

## Project Description

The goal of this project is to identify fraudulent credit card transactions using various machine learning models. The dataset used for training is highly imbalanced, with a small percentage of transactions being fraudulent. The project includes data preprocessing, model training, and a web interface for predictions.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```sh
    git https://github.com/Mwangiboyle/Fraud-detection-with-machine-learning.git
    cd Fraud-detection-with-machine-learning
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your virtual environment is activated:
    ```sh
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to use the app.

## Folder Structure

│
├── data/ # Contains the dataset

├── models/ # Contains the saved models

├── venv/ # Virtual environment directory (excluded from Git)

├── notebook.ipynb # Jupyter notebook for exploration and analysis

├── readme.md # Project description and instructions

├── requirements.txt # List of dependencies

├── .gitignore # Git ignore file

└── app.py # Streamlit app script

## Models

The `models` directory contains three pre-trained models used for predicting fraudulent transactions. Each model was trained using different machine learning algorithms to compare their performance.

## Streamlit App

The Streamlit app provides an interactive interface for users to input transaction details and get predictions on whether the transaction is fraudulent. The app also displays some basic statistics and visualizations of the dataset.

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss the changes you would like to make.
