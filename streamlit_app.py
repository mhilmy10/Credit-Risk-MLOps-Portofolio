import os
import io
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# In Docker Compose, the FastAPI service is reachable via its service name.
# Locally (running both with `streamlit run` / `uvicorn` on your machine), it's localhost.
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Prediction")
st.caption("MLOps portfolio project — FastAPI backend + Streamlit frontend")


def render_gauge(score: int):
    if score >= 700:
        color = "#2ecc71"
    elif score >= 600:
        color = "#f1c40f"
    else:
        color = "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credit Score"},
        gauge={
            "axis": {"range": [300, 850]},
            "bar": {"color": color},
            "steps": [
                {"range": [300, 600], "color": "#fadbd8"},
                {"range": [600, 700], "color": "#fdebd0"},
                {"range": [700, 850], "color": "#d5f5e3"},
            ],
        }
    ))
    fig.update_layout(height=300, margin=dict(t=50, b=10, l=30, r=30))
    return fig


def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/", timeout=5)
        return resp.ok, resp.json() if resp.ok else None
    except requests.exceptions.RequestException:
        return False, None


healthy, health_data = check_api_health()
with st.sidebar:
    st.subheader("API Status")
    if healthy and health_data and health_data.get("model_loaded"):
        st.success(f"Connected to API\n\n{API_URL}")
    elif healthy:
        st.warning("API is up but model is not loaded.")
    else:
        st.error(f"Cannot reach API at {API_URL}")

tab_single, tab_batch = st.tabs(["🔎 Single Prediction", "📁 Batch Prediction (CSV)"])

# SINGLE PREDICTION
with tab_single:
    st.subheader("Applicant Information")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
            person_income = st.number_input("Annual Income", min_value=0, value=55000, step=1000)
            person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            person_emp_length = st.number_input("Employment Length (years)", min_value=0.0, value=3.0, step=0.5)

        with col2:
            loan_intent = st.selectbox("Loan Intent", [
                "PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
            ])
            loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000, step=500)
            loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, value=11.5, step=0.1)

        with col3:
            loan_percent_income = st.number_input(
                "Loan as % of Income", min_value=0.0, max_value=1.0, value=0.18, step=0.01
            )
            cb_person_default_on_file = st.selectbox("Has Defaulted Before?", ["N", "Y"])
            cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=4)

        submitted = st.form_submit_button("Predict", use_container_width=True, type="primary")

    if submitted:
        payload = {
            "person_age": int(person_age),
            "person_income": float(person_income),
            "person_home_ownership": person_home_ownership,
            "person_emp_length": float(person_emp_length),
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": float(loan_amnt),
            "loan_int_rate": float(loan_int_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": int(cb_person_cred_hist_length),
        }
        try:
            with st.spinner("Scoring applicant..."):
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.ok:
                result = resp.json()
                label = result["label"]
                prob = result["probability_default"]
                score = result["credit_score"]

                res_col1, res_col2 = st.columns([1, 1])
                with res_col1:
                    if label == 1:
                        st.error(f"### ⚠️ High Risk (label={label})")
                    else:
                        st.success(f"### ✅ Low Risk (label={label})")
                    st.metric("Credit Score", score, help="Scorecard-style score, 300-850 (higher = lower risk)")
                    st.metric("Probability of Default", f"{prob * 100:.2f}%")

                with res_col2:
                    st.plotly_chart(render_gauge(score), use_container_width=True)
            else:
                st.error(f"API error ({resp.status_code}): {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach API: {e}")

# ---------------------------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload CSV for Batch Prediction")
    st.caption(
        "CSV must contain columns: person_age, person_income, person_home_ownership, "
        "person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, "
        "loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        preview_df = pd.read_csv(uploaded_file)
        st.write("Preview:")
        st.dataframe(preview_df.head(), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            try:
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                with st.spinner("Scoring batch..."):
                    resp = requests.post(f"{API_URL}/predict-batch", files=files, timeout=60)

                if resp.ok:
                    data = resp.json()
                    results_df = pd.DataFrame(data["results"])

                    st.success(f"Processed {data['total_records']} records")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Records", data["total_records"])
                    m2.metric("High Risk (1)", int((results_df["label"] == 1).sum()))
                    m3.metric("Low Risk (0)", int((results_df["label"] == 0).sum()))
                    m4.metric("Avg Credit Score", int(results_df["risk_score"].mean()))

                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.write("Risk Label Distribution")
                        st.bar_chart(results_df["label"].value_counts())
                    with chart_col2:
                        st.write("Credit Score Distribution")
                        st.bar_chart(results_df["risk_score"])

                    st.write("Detailed Results:")
                    st.dataframe(results_df, use_container_width=True)

                    # Offer downloadable CSV (re-request the CSV-formatted endpoint)
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    csv_resp = requests.post(f"{API_URL}/predict-batch-csv", files=files, timeout=60)
                    if csv_resp.ok:
                        st.download_button(
                            "⬇️ Download Results as CSV",
                            data=csv_resp.content,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error(f"API error ({resp.status_code}): {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach API: {e}")