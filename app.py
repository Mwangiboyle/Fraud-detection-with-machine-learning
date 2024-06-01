import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# load models
log_reg_model = joblib.load('model/Logistic_Regression.joblib')
knn_model = joblib.load('model/K-Nearest_Neighbors.joblib')
dt_model = joblib.load('model/Decision_Tree.joblib')

# Title for the application

st.title("Crediit Card Fraud Detection")

# input fields fo user

features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
user_input = {}
for feature in features:
    user_input[feature] = st.number_input(feature, value=0.0)

# Choose model in selectbox
model = st.sidebar.selectbox('Select Model From Here', ('Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree'))

# predict button

if st.button('Predict'):
    # create dataframe with user input

    input_data = pd.DataFrame(user_input, index=[0])

    # load the appropriate model
    if model == 'Logistic Regression':
        model = log_reg_model
    elif model == 'K-Nearest Neighbors':
        model = knn_model
    elif model == 'Decision Tree':
        model = dt_model

    # scale the input_data
    scaler = StandardScaler()
    scaler.fit(input_data)
    input_data = scaler.transform(input_data)

    # make prediction
    prediction = model.predict(input_data)

    # print prediction
    if prediction == 0:
        st.success('The transaction is not a fraud')
    else:
        st.error('The transaction is a fraud')
