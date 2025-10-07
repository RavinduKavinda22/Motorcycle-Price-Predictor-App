import streamlit as st
import pandas as pd
import numpy as np
import joblib
from custom_models import EnsembleRegressor 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["XGBOOST_ENABLE_STRICT_VERSION_CHECK"] = "0"


# =============================
# Load Models & Encoders
# =============================
ensemble_model = joblib.load("Ensemble_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Motorcycle Price Predictor", layout="wide")
st.title("Motorcycle Price Prediction (Sri Lanka)")

# =============================
# Mapping Rules
# =============================
brand_model_map = {
    "Bajaj": ["CT-100", "Discover 125"],
    "Honda": ["DIO"],
    "TVS": ["Ntorq 125"],
    "Yamaha": ["FZ Version 2", "RAY ZR"]
}

# Year ranges per model
model_year_ranges = {
    "CT-100": (2004, 2025),
    "Discover 125": (2005, 2025),
    "DIO": (2012, 2025),
    "Ntorq 125": (2018, 2025),
    "FZ Version 2": (2014, 2025),
    "RAY ZR": (2016, 2025),
}

# Engine & start type rules
model_specs = {
    "CT-100": {"cc": [100], "start_types": ["Kick / Electric", "Kick"]},
    "Discover 125": {"cc": [125], "start_types": ["Kick / Electric", "Kick"]},
    "DIO": {"cc": [110], "start_types": ["Kick / Electric"]},
    "Ntorq 125": {"cc": [125], "start_types": ["Kick / Electric"]},
    "FZ Version 2": {"cc": [149], "start_types": ["Electric"]},
    "RAY ZR": {"cc": [113, 125], "start_types": ["Kick / Electric"]},
}

# =============================
# User Input Form
# =============================
st.sidebar.header("Enter Motorcycle Details")

brand = st.sidebar.selectbox("Brand", list(brand_model_map.keys()))
model = st.sidebar.selectbox("Model", brand_model_map[brand])

condition = st.sidebar.selectbox("Condition", label_encoders["Condition"].classes_)

# If Brand New → lock YOM=2025 & Mileage=0
if condition == "Brand New":
    yom = 2025
    mileage = 0
    st.sidebar.text_input("Year of Manufacture", value="2025", disabled=True)
    st.sidebar.text_input("Mileage (km)", value="0", disabled=True)
else:
    min_year, max_year = model_year_ranges[model]
    yom = st.sidebar.number_input(
        "Year of Manufacture", 
        min_value=min_year, 
        max_value=max_year, 
        value=min_year, 
        step=1
    )
    mileage = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=300000, value=45000, step=500)

# Engine CC logic
if model == "RAY ZR":
    if 2016 <= yom <= 2018:
        default_cc = 113
    elif 2019 <= yom <= 2025:
        default_cc = 125
    else:
        default_cc = 113
    engine_cc = st.sidebar.selectbox(
        "Engine (cc)", 
        model_specs["RAY ZR"]["cc"],
        index=model_specs["RAY ZR"]["cc"].index(default_cc)
    )
else:
    engine_cc = st.sidebar.selectbox("Engine (cc)", model_specs[model]["cc"])

# Start Type selection
start_type = st.sidebar.selectbox("Start Type", model_specs[model]["start_types"])

# =============================
# Prediction
# =============================
if st.sidebar.button("Predict Price"):
    user_input = {
        "Brand": brand,
        "Model": model,
        "Engine (cc)": engine_cc,
        "YOM": yom,
        "Mileage": mileage,
        "Condition": condition,
        "Start Type": start_type
    }
    input_df = pd.DataFrame([user_input])

    # Encode categorical columns
    for col in ["Brand", "Model", "Condition", "Start Type"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Predict
    y_pred_log = ensemble_model.predict(input_df)
    predicted_price = np.expm1(y_pred_log)[0]

    # Show Results
    st.success(f"💰 Predicted Price: Rs. {predicted_price:,.0f}")
    st.info(f"👉 Lakh: {predicted_price/100000:.2f} Lakh")
    
    # Disclaimer
    st.caption("⚠️ This is only an **estimated price** assuming the motorcycle is in good condition. The price may vary depending on the real condition of the motorcycle.")

# =============================
# Chatbot Button (Always Visible)
# =============================
st.markdown("---")
st.markdown("###  Need more help about motorcycles?")
st.markdown(
    "<a href='https://sri-lanaka-motorcycle-chatbot.streamlit.app/' target='_blank'>"
    "<button style='background-color:#4CAF50; color:white; padding:10px 20px; "
    "border:none; border-radius:8px; font-size:16px; cursor:pointer;'>"
    "🤖 Open Sri Lanka Motorcycle Chatbot</button></a>",
    unsafe_allow_html=True
)

# =============================
# About Section
# =============================
st.markdown("---")
st.subheader("ℹ️ About this App")
st.markdown("""
This **Motorcycle Price Predictor (Sri Lanka)** is built using **Machine Learning** models 
to estimate used motorcycle prices based on **brand, model, year, mileage, condition, Engine cc and start type**.  

- Data collected through web scraping from **Riyasewana.lk** and **Ikman.lk**, covering a wide range of motorcycles and scooters listed between **March 2025 and September 2025**.  
- Powered by an **Ensemble Regressor** trained on Sri Lankan market data.  
- Evaluated multiple regression models, including **Random Forest, Extra Trees, Gradient Boosting, and XGBoost**, before selecting the final ensemble approach.  
- Supports popular models such as **Bajaj CT-100, Discover 125, Honda Dio, TVS Ntorq, Yamaha FZ V2, and Yamaha Ray ZR**.  
- Integrated with a [Motorcycle Chatbot](https://sri-lanaka-motorcycle-chatbot.streamlit.app/) for additional guidance and motorcycle information.  
- Designed and developed as part of a **final year research project** aimed at improving **transparency in Sri Lanka's used motorcycle market**.    

**Developer:** M R K Karunathilaka  
**Version:** 1.0.0
""")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>© 2025 Motorcycle Price Predictor – Research Project</p>", unsafe_allow_html=True)
