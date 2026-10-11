"""
Heart Disease Prediction - Streamlit Web Application
====================================================
Teaching demo (AIC354 - Machine Learning Fundamentals, Assignment 2, Task 2).
Loads the saved scikit-learn pipeline (trained on the Kaggle Heart Failure
Prediction dataset) and predicts from exactly four inputs.
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "heart_disease_model.pkl"

CHEST_PAIN_LABELS = {
    "TA": "Typical Angina (TA)",
    "ATA": "Atypical Angina (ATA)",
    "NAP": "Non-Anginal Pain (NAP)",
    "ASY": "Asymptomatic (ASY)",
}
SEX_LABELS = {"M": "Male", "F": "Female"}


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


try:
    artifact = load_artifact()
    model = artifact["pipeline"]
    FEATURES = artifact["features"]
except FileNotFoundError:
    st.error(f"Model file `{MODEL_PATH.name}` was not found. Run `python train_model.py` first.")
    st.stop()
except Exception as exc:  # e.g. scikit-learn version mismatch
    st.error(
        "The saved model could not be loaded. It was probably trained with a different "
        f"scikit-learn version than the one installed. Re-run `python train_model.py`.\n\nDetails: {exc}"
    )
    st.stop()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    """
    Teaching demo built for **AIC354 — Machine Learning Fundamentals
    (Assignment 2, Task 2)**. Enter a patient's details and the trained
    **Support Vector Classifier (SVC)** predicts whether heart disease is indicated.
    Trained on the Kaggle *Heart Failure Prediction* dataset (918 records).

    **Input attributes:** Age, Sex, Chest Pain Type, Max Heart Rate
    """
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=28, max_value=77, value=54, step=1,
                          help="Training data covers ages 28–77")
    sex = st.selectbox("Sex", options=list(SEX_LABELS), format_func=SEX_LABELS.get)
with col2:
    chest_pain = st.selectbox("Chest Pain Type", options=list(CHEST_PAIN_LABELS),
                              format_func=CHEST_PAIN_LABELS.get, index=3)
    max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=202, value=140, step=1,
                             help="Training data covers 60–202 bpm")

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame(
        [{"Age": age, "Sex": sex, "ChestPainType": chest_pain, "MaxHR": max_hr}]
    )[FEATURES]
    try:
        prediction = int(model.predict(user_input)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ Heart Disease Indicated")
        st.markdown("The model predicts that heart disease **may be present** based on the provided attributes.")
    else:
        st.success("### ✅ No Heart Disease Indicated")
        st.markdown("The model predicts that heart disease is **not indicated** based on the provided attributes.")

    with st.expander("See the input sent to the model"):
        st.dataframe(user_input, hide_index=True)

st.divider()
st.caption(
    "⚕️ **Disclaimer:** Educational model for a university assignment. It is NOT a medical "
    "diagnostic tool and has not been clinically validated. It uses only four attributes and a "
    "small public dataset (Kaggle Heart Failure Prediction, 918 records) and must never be used for "
    "clinical decisions. Consult a qualified healthcare professional for medical advice."
)
