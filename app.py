import streamlit as st
import pandas as pd
import joblib

# ---- Page setup (must be the first Streamlit command) ----
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")


# ---- Load the saved files once and keep them cached ----
@st.cache_resource
def load_artifacts():
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scalar.pkl')
    columns = joblib.load('columns.pkl')   # exact column order used in training
    return model, scaler, columns


model, scaler, columns = load_artifacts()


# ---- Sidebar: info about the app ----
with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts the risk of heart disease using a "
        "Logistic Regression model trained on 918 patient records."
    )
    st.write(f"**Model:** {type(model).__name__}")
    st.warning("Educational project only — not real medical advice.")


st.title("Heart Disease Prediction ❤️")
st.write("Fill in the patient details below, then click **Predict**.")

# ---- Collect inputs, split across two columns ----
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Info")
    age = st.slider("Age", 1, 100, 50)
    sex = st.radio("Sex", ["M", "F"], horizontal=True)
    chest_pain = st.selectbox(
        "Chest Pain Type", ["ASY", "ATA", "NAP", "TA"],
        help="ASY=Asymptomatic, ATA=Atypical Angina, NAP=Non-Anginal Pain, TA=Typical Angina",
    )
    resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 60, 220, 130)
    cholesterol = st.slider("Cholesterol (mg/dl)", 50, 700, 240)
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

with col2:
    st.subheader("Test Results")
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.radio(
        "Exercise Induced Angina", ["N", "Y"], horizontal=True,
        format_func=lambda x: "Yes" if x == "Y" else "No",
    )
    oldpeak = st.slider("Oldpeak (ST depression)", -3.0, 7.0, 1.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])


# ---- Predict ----
st.divider()
if st.button("Predict", type="primary", use_container_width=True):
    # Build a one-row DataFrame with the RAW values
    raw = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    # Encode exactly like training (get_dummies + drop_first), then
    # reindex to the saved training columns so order/shape always match.
    encoded = pd.get_dummies(raw, drop_first=True)
    encoded = encoded.reindex(columns=columns, fill_value=0)

    # Scale, then predict
    scaled = scaler.transform(encoded)
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    # ---- Show the result ----
    res_col, gauge_col = st.columns(2)

    with res_col:
        if prediction == 1:
            st.error("### ⚠️ High Risk of Heart Disease")
        else:
            st.success("### ✅ Low Risk of Heart Disease")
        st.metric("Risk Probability", f"{probability:.1%}")

    with gauge_col:
        st.write("**Risk level**")
        st.progress(float(probability))
        if probability < 0.4:
            st.caption("🟢 Low risk range")
        elif probability < 0.7:
            st.caption("🟡 Moderate risk range")
        else:
            st.caption("🔴 High risk range")

    # ---- Show what was entered ----
    with st.expander("See the values you entered"):
        st.dataframe(raw.T.rename(columns={0: "Value"}))
