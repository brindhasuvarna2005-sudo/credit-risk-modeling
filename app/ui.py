import streamlit as st
import joblib
import pandas as pd
import shap
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------------
# FEATURE NAME MAPPING
# -------------------------------
FEATURE_NAMES = {
    "AMT_INCOME_TOTAL": "Total Income",
    "AMT_CREDIT": "Loan Amount",
    "AMT_ANNUITY": "Loan Annuity",
    "AGE_YEARS": "Age",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "CREDIT_TO_ANNUITY_RATIO": "Credit-to-Annuity Ratio",
    "ANNUITY_TO_INCOME_RATIO": "Annuity-to-Income Ratio"
}

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "xgb_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

THRESHOLD = 0.2

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.title("Applicant Details")

income = st.sidebar.number_input("Total Income", value=500000)
credit = st.sidebar.number_input("Loan Amount", value=1000000)
annuity = st.sidebar.number_input("Annuity", value=30000)
age = st.sidebar.number_input("Age", value=35)
ext2 = st.sidebar.number_input("External Score 2", value=0.5)
ext3 = st.sidebar.number_input("External Score 3", value=0.5)
ratio = st.sidebar.number_input("Credit/Annuity Ratio", value=30)
income_ratio = st.sidebar.number_input("Annuity/Income Ratio", value=0.1)

predict_btn = st.sidebar.button("Predict Risk")

# -------------------------------
# HEADER
# -------------------------------
st.title("Credit Risk Scoring Dashboard")
st.write("AI-powered loan default prediction system")

# -------------------------------
# PREDICTION (no FastAPI)
# -------------------------------
if predict_btn:

    input_data = {
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AGE_YEARS": age,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "CREDIT_TO_ANNUITY_RATIO": ratio,
        "ANNUITY_TO_INCOME_RATIO": income_ratio
    }

    # ✅ Direct model call — no FastAPI needed
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[:, 1][0]
    pred = int(prob > THRESHOLD)

    # -------------------------------
    # RISK LEVEL
    # -------------------------------
    if prob < 0.2:
        risk_level, color = "Low Risk", "green"
    elif prob < 0.5:
        risk_level, color = "Medium Risk", "orange"
    else:
        risk_level, color = "High Risk", "red"

    col1, col2 = st.columns(2)
    col1.metric("Default Probability", f"{prob*100:.1f}%")
    col2.metric("Risk Level", risk_level)

    # -------------------------------
    # GAUGE
    # -------------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Default Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # SHAP EXPLANATION
    # -------------------------------
    st.subheader("Model Explanation")

    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    shap_df = pd.DataFrame({
        "feature": df.columns,
        "impact": shap_values.values[0]
    })

    shap_df["feature"] = shap_df["feature"].map(FEATURE_NAMES).fillna(shap_df["feature"])
    shap_df = shap_df.sort_values(by="impact", key=abs, ascending=False).head(8)

    fig_shap = go.Figure(go.Bar(
        x=shap_df["impact"],
        y=shap_df["feature"],
        orientation='h'
    ))
    fig_shap.update_layout(title="Top Factors Affecting Risk")
    st.plotly_chart(fig_shap, use_container_width=True)

    # -------------------------------
    # HUMAN-READABLE INSIGHTS
    # -------------------------------
    st.subheader("Key Insights")

    for _, row in shap_df.head(3).iterrows():
        direction = "increases" if row["impact"] > 0 else "reduces"
        st.write(f"{row['feature']} {direction} the risk")

    # -------------------------------
    # INPUT SUMMARY
    # -------------------------------
    st.subheader("Input Summary")
    readable_input = {FEATURE_NAMES.get(k, k): v for k, v in input_data.items()}
    st.json(readable_input)