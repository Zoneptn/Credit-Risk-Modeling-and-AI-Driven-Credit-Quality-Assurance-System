import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Credit QA", layout="wide")

st.title("LLM-Powered Credit Quality Assurance")


# Load data
qa_cases = pd.read_csv("result_table/qa_cases.csv")
ai_decisions = pd.read_csv("result_table/ai_decision.csv")


ai_decisions = ai_decisions.reset_index(drop=True)

# Merge AI output back to cases
df = qa_cases.reset_index().merge(ai_decisions, on="case_id", how="left")
df.index = df.index +1

st.subheader("QA Case List")
st.dataframe(df[["case_id", "PD", "Action", "ai_decision"]])
st.caption(
    "⚠️ Note: This dashboard shows a sample run using only 10 customer cases for demonstration purposes."
)

# Select loan
selected_id = st.selectbox("Select a Loan ID", df["case_id"])

case = df[df["case_id"] == selected_id].iloc[0]

st.subheader("Loan Details")
d1, d2, d3 = st.columns(3)
d1.metric("Loan Amount", f"${case.loan_amnt:,.0f}")
d2.metric("Debt-to-Income", f"{case.dti:.1%}")
d3.metric("Loan-to-Income", f"{case.loan_to_income:.1%}")

d4, d5, d6 = st.columns(3)
d4.metric("Revolving Utilization", f"{case.revol_util:.1f}%")
d5.metric("Employment Length", f"{case.emp_length:.1f} yrs")
d6.metric("Income Verified", case.verification_status)

st.subheader("Loan Risk Summary")
col1, col2, col3,col4 = st.columns(4)
col1.metric("Probability of Default", f"{case.PD:.2%}")
col2.metric("Policy Segment", case.Action)
col3.metric("AI Decision", case.ai_decision)
col4.metric("Expected Loss", f"${case.expected_loss:,.0f}")

st.subheader("Version Info")
a1,a2,a3 = st.columns(3)
a1.metric("Model Version", case.model_version)
a2.metric("LangGraph Version", case.langgraph_version)
a3.metric(
    "Timestamp",
    pd.to_datetime(case.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
)


st.markdown("### Top Risk Drivers")
st.write(case.top_risk_factors)

st.markdown("### Policy Assessment")
st.write(case.policy_status)

st.markdown("### Recommended Next Step")
st.write(case.next_step)

st.markdown("### AI Explanation")
st.write(case.ai_reason)











