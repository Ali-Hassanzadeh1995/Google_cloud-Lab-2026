import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://obesity-api-744193306580.europe-west1.run.app")

st.set_page_config(page_title="Obesity Prediction", page_icon="⚖️", layout="centered")

st.title("🍕⚖️🍎 Obesity Level Prediction ")
st.write("Choose the user's information and get an automatic prediction.")

with st.form("obesity_form"):
    Gender = st.selectbox("Gender", ["Female", "Male"])

    Age = st.slider("Age", 10, 100, 28, step=1)
    Height = st.number_input(
        "Height (meters)", min_value=1.00, max_value=2.50, value=1.62, step=0.01
    )
    Weight = st.number_input(
        "Weight (kg)", min_value=20.0, max_value=250.0, value=64.0, step=1.0
    )

    family_history_with_overweight = st.selectbox(
        "Family history with overweight?", ["yes", "no"]
    )

    FAVC = st.selectbox("Frequent consumption of high caloric food?", ["yes", "no"])

    FCVC = st.slider(
        "Frequency of vegetables consumption (per weak)",
        min_value=0,
        max_value=7,
        value=2,
        step=1,
    )

    NCP = st.slider(
        "Number of main meals per day",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
    )

    CAEC = st.selectbox(
        "Food between meals", ["no", "Sometimes", "Frequently", "Always"]
    )

    SMOKE = st.selectbox("Do you smoke?", ["yes", "no"])

    CH2O = st.slider(
        "Daily water consumption (liters)",
        min_value=1.0,
        max_value=4.0,
        value=2.0,
        step=0.1,
    )

    SCC = st.selectbox("Do you monitor calories?", ["yes", "no"])

    FAF = st.slider(
        "Physical activity frequency (hours/day)",
        min_value=0.0,
        max_value=24.0,
        value=1.0,
        step=0.1,
    )

    TUE = st.slider(
        "Time using technology devices (hours/day)",
        min_value=0.0,
        max_value=24.0,
        value=1.0,
        step=0.1,
    )

    CALC = st.selectbox(
        "Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"]
    )

    MTRANS = st.selectbox(
        "Transportation",
        ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"],
    )

    submit = st.form_submit_button("Predict")

if submit:
    payload = {
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "family_history_with_overweight": family_history_with_overweight,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            st.success(f"Predicted obesity level: {result['prediction']}")

            if "probabilities" in result:
                st.subheader("Prediction probabilities")
                st.bar_chart(result["probabilities"])

        else:
            st.error(f"API error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to the prediction API.")
        st.write(e)
