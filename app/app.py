import streamlit as st
import pandas as pd
import joblib
import os

# Get the folder this script lives in, so file paths work regardless of 
# where the app is launched from (local machine vs EC2 server)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and column list
model = joblib.load(os.path.join(BASE_DIR, 'house_price_model.pkl'))
model_columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))

# App title
st.title("House Price Prediction")
st.write("Enter house details to estimate the sale price.")

# --- User inputs (key fields based on our top features) ---
overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=0, value=1500)
garage_cars = st.number_input("Garage Capacity (cars)", min_value=0, max_value=5, value=2)
garage_area = st.number_input("Garage Area (sq ft)", min_value=0, value=400)
total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, value=800)
year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2000)
central_air = st.selectbox("Central Air Conditioning", ["Yes", "No"])

# --- When user clicks Predict ---
if st.button("Predict Sale Price"):
    # Build a single-row input matching training data format
    input_dict = {col: 0 for col in model_columns}  # start with all zeros

    # Fill in numeric values (only if column exists in training data)
    for col, val in [
        ('OverallQual', overall_qual),
        ('GrLivArea', gr_liv_area),
        ('GarageCars', garage_cars),
        ('GarageArea', garage_area),
        ('TotalBsmtSF', total_bsmt_sf),
        ('YearBuilt', year_built),
    ]:
        if col in input_dict:
            input_dict[col] = val

    # Handle CentralAir (one-hot encoded as CentralAir_Y)
    if central_air == "Yes" and 'CentralAir_Y' in input_dict:
        input_dict['CentralAir_Y'] = 1

    # Convert to DataFrame in correct column order
    input_df = pd.DataFrame([input_dict])[model_columns]

    # Predict (model outputs log-transformed price, so convert back)
    import numpy as np
    prediction_log = model.predict(input_df)[0]
    prediction_actual = np.expm1(prediction_log)

    st.success(f"🏠 Estimated Sale Price: ${prediction_actual:,.0f}")