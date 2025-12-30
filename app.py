
import streamlit as st
import joblib
import pandas as pd
import requests

model = joblib.load('lightgbm_best_model.pkl')
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction App")

st.sidebar.header("Input Loan Applicant Details")
income = st.sidebar.number_input("Annual Income", min_value=0)
dti = st.sidebar.slider("Debt-to-Income Ratio (DTI) %", min_value=0.0, max_value=100.0, step=0.1)
loan_amount= st.sidebar.number_input("Loan Amount", min_value=0)
interest_rate = st.sidebar.slider("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

x = pd.DataFrame({
    'annual_inc': [income],
    'dti': [dti],
    'loan_amnt': [loan_amount],
    'int_rate': [interest_rate]
})

# Store prediction in session state
if "pd_pred" not in st.session_state:
    st.session_state.pd_pred = None
    st.session_state.band = None

# Run PD model
if st.button("Predict Default Probability"):
    pd_pred = model.predict_proba(x)[0, 1]
    st.session_state.pd_pred = pd_pred

    # Risk band logic
    if pd_pred >= 0.3:
        band = "Reject"
        st.error("High Risk - Review / Reject")
    elif pd_pred >= 0.2:
        band = "QA Review"
        st.warning("Medium Risk - QA")
    elif pd_pred >= 0.05:
        band = "Approve with Caution"
        st.info("Low to Medium Risk - Approve with Caution")
    else:
        band = "Approve"
        st.success("Low Risk - Approve")

    st.session_state.band = band
    st.metric("Probability of Default", f"{pd_pred:.2%}")

# RAG Button
if st.session_state.pd_pred is not None:
    if st.button("Generate Credit Recommendation"):
        payload = {
            "PD": float(st.session_state.pd_pred),
            "income": income,
            "dti": dti,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "risk_band": st.session_state.band
        }

        r = requests.post("http://127.0.0.1:8000/rag_credit", json=payload)

        st.subheader("AI Credit Recommendation")

        if r.status_code == 200:
            data = r.json()
            st.write(data.get("recommendation", "No recommendation returned"))
        else:
            st.error(f"RAG server error: {r.text}")





