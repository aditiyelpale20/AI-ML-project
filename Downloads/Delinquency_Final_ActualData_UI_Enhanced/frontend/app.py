
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent.parent
API_URL = "http://127.0.0.1:8000"
META_PATH = BASE / "backend" / "model_metadata.json"
DATA_PATH = BASE / "backend" / "data" / "Delinquency_prediction_dataset.csv"

META = json.loads(META_PATH.read_text(encoding="utf-8"))
DATA = pd.read_csv(DATA_PATH)

st.set_page_config(
    page_title="Delinquency Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Theme -----------------------------
st.markdown("""
<style>
:root {
    --ink:#172033;
    --muted:#6b7280;
    --line:#e7ebf2;
    --surface:#ffffff;
    --soft:#f6f8fc;
    --accent:#4f46e5;
}
.block-container {max-width: 1450px; padding: 1.4rem 2rem 3rem;}
[data-testid="stSidebar"] {border-right:1px solid #e8ecf3;}
.hero {
    padding: 1.8rem 2rem;
    border-radius: 24px;
    background: linear-gradient(135deg,#111827 0%,#27345d 58%,#4f46e5 100%);
    color:white;
    margin-bottom:1.25rem;
    box-shadow:0 14px 40px rgba(17,24,39,.14);
}
.hero h1 {font-size:2.35rem; margin:0 0 .35rem; letter-spacing:-.03em;}
.hero p {margin:0; color:#dbe3f4; font-size:1rem;}
.section-title {font-size:1.35rem; font-weight:750; color:var(--ink); margin:.35rem 0 .85rem;}
.kpi {
    background:var(--surface); border:1px solid var(--line); border-radius:18px;
    padding:1.05rem 1.2rem; box-shadow:0 7px 22px rgba(15,23,42,.045);
}
.kpi-label {font-size:.78rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em;}
.kpi-value {font-size:1.65rem; font-weight:800; color:var(--ink); margin-top:.15rem;}
.kpi-note {font-size:.78rem; color:var(--muted);}
.insight {
    padding:1rem 1.1rem; background:#f7f8ff; border:1px solid #e7e9ff;
    border-radius:16px; color:#30354b;
}
.result-card {
    border:1px solid var(--line); border-radius:20px; padding:1.25rem;
    background:#fff; box-shadow:0 8px 28px rgba(15,23,42,.05);
}
.risk-high {background:#fff1f2; border:1px solid #fecdd3; color:#9f1239;}
.risk-mid {background:#fffbeb; border:1px solid #fde68a; color:#92400e;}
.risk-low {background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46;}
.small {color:var(--muted); font-size:.82rem;}
div[data-testid="stMetric"] {
    background:white; border:1px solid var(--line); padding:1rem;
    border-radius:16px; box-shadow:0 5px 18px rgba(15,23,42,.04);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------- Data helpers -----------------------------
status_order = ["On-time", "Late", "Missed"]

def pct(n, d):
    return f"{(n / d * 100):.1f}%" if d else "0.0%"

month6 = DATA["Month_6"].value_counts().reindex(status_order, fill_value=0)
missed_share = month6["Missed"] / len(DATA)
avg_score = DATA["Credit_Score"].mean()
avg_util = DATA["Credit_Utilization"].mean()
avg_income = DATA["Income"].mean()

# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.markdown("## 💳 Delinquency AI")
    st.caption("Analytics + prediction workspace")
    st.divider()

    st.markdown("### System")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        api_ok = health.ok
    except Exception:
        api_ok = False

    if api_ok:
        st.success("API connected")
    else:
        st.warning("API offline — analytics still available")

    st.divider()
    st.markdown("### Model")
    st.write("**Random Forest Classifier**")
    st.metric("Training records", f"{META['rows']:,}")
    st.metric("Test accuracy", f"{META['accuracy']:.1%}")
    st.caption("Accuracy is the saved hold-out test result from the current model artifact.")

    st.divider()
    st.caption("Data source")
    st.code("backend/data/Delinquency_prediction_dataset.csv", language="text")

# ----------------------------- Hero -----------------------------
st.markdown("""
<div class="hero">
  <h1>💳 Customer Delinquency Intelligence</h1>
  <p>Explore the real customer dataset, understand payment behaviour, and predict the next payment status through the FastAPI ML service.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Navigation -----------------------------
tab_overview, tab_predict, tab_data = st.tabs(
    ["📊 Dashboard", "🔮 Predict Customer", "🗂️ Dataset Explorer"]
)

# ============================= DASHBOARD =============================
with tab_overview:
    st.markdown('<div class="section-title">Portfolio overview</div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Customers</div><div class="kpi-value">{len(DATA):,}</div><div class="kpi-note">records in original dataset</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Missed in Month 6</div><div class="kpi-value">{month6["Missed"]:,}</div><div class="kpi-note">{missed_share:.1%} of customers</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Avg credit score</div><div class="kpi-value">{avg_score:,.0f}</div><div class="kpi-note">dataset average</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="kpi"><div class="kpi-label">Avg utilization</div><div class="kpi-value">{avg_util:.1%}</div><div class="kpi-note">credit utilization</div></div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1, 1.15])

    with left:
        fig = px.pie(
            values=month6.values,
            names=month6.index,
            hole=.58,
            title="Month 6 payment-status mix",
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10), legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        loc = DATA["Location"].value_counts().sort_values(ascending=True).reset_index()
        loc.columns = ["Location", "Customers"]
        fig = px.bar(
            loc, x="Customers", y="Location", orientation="h",
            title="Customers by location", text="Customers"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10), yaxis_title="", xaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        emp = DATA["Employment_Status"].value_counts().reset_index()
        emp.columns = ["Employment Status", "Customers"]
        fig = px.bar(
            emp, x="Employment Status", y="Customers",
            title="Customer mix by employment status", text="Customers"
        )
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=10), xaxis_title="", yaxis_title="Customers")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            DATA,
            x="Credit_Utilization",
            y="Missed_Payments",
            color="Month_6",
            hover_data=["Customer_ID", "Credit_Score", "Income", "Location"],
            category_orders={"Month_6": status_order},
            title="Utilization vs missed payments",
            labels={
                "Credit_Utilization":"Credit utilization",
                "Missed_Payments":"Missed payments",
                "Month_6":"Month 6 status"
            }
        )
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Payment behaviour across six months")
    month_cols = [f"Month_{i}" for i in range(1, 7)]
    trend = pd.DataFrame({
        "Month": [f"Month {i}" for i in range(1, 7)],
        "On-time": [(DATA[c] == "On-time").sum() for c in month_cols],
        "Late": [(DATA[c] == "Late").sum() for c in month_cols],
        "Missed": [(DATA[c] == "Missed").sum() for c in month_cols],
    })
    fig = px.line(
        trend, x="Month", y=["On-time", "Late", "Missed"],
        markers=True, title="Payment status trend"
    )
    fig.update_layout(height=390, margin=dict(l=10,r=10,t=55,b=10), yaxis_title="Customers", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight"><b>Dataset snapshot:</b> average income is ₹{avg_income:,.0f}, '
        f'average credit score is {avg_score:,.0f}, and {missed_share:.1%} of customers have a '
        f'<b>Missed</b> Month 6 payment status.</div>',
        unsafe_allow_html=True
    )

# ============================= PREDICTION =============================
with tab_predict:
    st.markdown('<div class="section-title">Customer prediction</div>', unsafe_allow_html=True)
    st.caption("Enter the same feature set used by the trained model. The prediction is returned by FastAPI.")

    with st.form("prediction_form"):
        st.markdown("#### 1. Financial profile")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            age = st.number_input("Age", 18, 100, 35)
        with c2:
            income = st.number_input("Income (₹)", 0.0, 10_000_000.0, 75_000.0, step=1_000.0)
        with c3:
            credit_score = st.number_input("Credit score", 300.0, 850.0, 650.0)
        with c4:
            loan = st.number_input("Loan balance (₹)", 0.0, 10_000_000.0, 40_000.0, step=1_000.0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            utilization = st.slider("Credit utilization", 0.0, 1.0, 0.35, 0.01)
        with c2:
            dti = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.30, 0.01)
        with c3:
            missed = st.number_input("Missed payments", 0, 30, 1)
        with c4:
            tenure = st.number_input("Account tenure (months)", 0, 120, 12)

        st.markdown("#### 2. Customer profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            employment = st.selectbox("Employment status", META["employment_options"])
        with c2:
            card = st.selectbox("Credit card type", META["card_options"])
        with c3:
            location = st.selectbox("Location", META["location_options"])

        st.markdown("#### 3. Previous payment history")
        cols = st.columns(5)
        months = []
        for i, col in enumerate(cols, 1):
            with col:
                months.append(st.selectbox(
                    f"Month {i}", META["payment_options"],
                    key=f"pred_month_{i}"
                ))

        submitted = st.form_submit_button(
            "🔮 Predict Month 6 Payment Status",
            type="primary",
            use_container_width=True
        )

    if submitted:
        payload = {
            "age": age, "income": income, "credit_score": credit_score,
            "credit_utilization": utilization, "missed_payments": missed,
            "delinquent_account": 1 if missed > 0 else 0,
            "loan_balance": loan, "debt_to_income_ratio": dti,
            "employment_status": employment, "account_tenure": tenure,
            "credit_card_type": card, "location": location,
            "month_1": months[0], "month_2": months[1], "month_3": months[2],
            "month_4": months[3], "month_5": months[4],
        }

        with st.spinner("Calling the prediction API..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
                r.raise_for_status()
                result = r.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                st.divider()
                st.markdown("### Prediction result")

                if prediction == "Missed":
                    css, icon, text = "risk-high", "🔴", "Higher delinquency signal"
                elif prediction == "Late":
                    css, icon, text = "risk-mid", "🟠", "Moderate payment-risk signal"
                else:
                    css, icon, text = "risk-low", "🟢", "On-time payment signal"

                a, b = st.columns([1, 1.2])
                with a:
                    st.markdown(
                        f'<div class="result-card {css}">'
                        f'<div class="small">MODEL PREDICTION</div>'
                        f'<h1 style="margin:.15rem 0">{icon} {prediction}</h1>'
                        f'<b>{text}</b>'
                        f'<p style="margin:.65rem 0 0">Confidence: <b>{confidence:.1%}</b></p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                with b:
                    probs = pd.DataFrame({
                        "Status": list(result["probabilities"].keys()),
                        "Probability": [v * 100 for v in result["probabilities"].values()]
                    })
                    fig = px.bar(
                        probs, x="Status", y="Probability", text="Probability",
                        title="Class probabilities"
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(
                        height=300, margin=dict(l=10,r=10,t=55,b=10),
                        yaxis_title="Probability (%)", xaxis_title=""
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Model factors")
                factors = pd.DataFrame(result["top_model_factors"])
                factors = factors.rename(columns={"feature":"Feature", "importance":"Importance"})
                factors["Importance"] = factors["Importance"] * 100
                fig = px.bar(
                    factors.sort_values("Importance"),
                    x="Importance", y="Feature", orientation="h",
                    text="Importance", title="Top model feature importances"
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(
                    height=390, margin=dict(l=10,r=10,t=55,b=10),
                    xaxis_title="Importance (%)", yaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    '<div class="insight"><b>Important:</b> feature importance shows which '
                    'variables the Random Forest used most across the model; it does not by '
                    'itself prove that a feature caused the predicted outcome.</div>',
                    unsafe_allow_html=True
                )

            except requests.exceptions.ConnectionError:
                st.error("FastAPI is not running. Start the backend first with the backend command.")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ============================= DATA EXPLORER =============================
with tab_data:
    st.markdown('<div class="section-title">Dataset explorer</div>', unsafe_allow_html=True)
    st.caption("This view uses the original 500-row dataset bundled with the project.")

    c1, c2, c3 = st.columns(3)
    with c1:
        status_filter = st.multiselect("Month 6 status", status_order, default=status_order)
    with c2:
        locations = sorted(DATA["Location"].dropna().unique())
        location_filter = st.multiselect("Location", locations, default=locations)
    with c3:
        min_score, max_score = float(DATA["Credit_Score"].min()), float(DATA["Credit_Score"].max())
        score_range = st.slider("Credit score range", min_score, max_score, (min_score, max_score))

    filtered = DATA[
        DATA["Month_6"].isin(status_filter)
        & DATA["Location"].isin(location_filter)
        & DATA["Credit_Score"].between(score_range[0], score_range[1])
    ]

    a, b, c = st.columns(3)
    a.metric("Filtered customers", f"{len(filtered):,}")
    b.metric("Missed share", pct((filtered["Month_6"] == "Missed").sum(), len(filtered)))
    c.metric("Average income", f"₹{filtered['Income'].mean():,.0f}" if len(filtered) else "—")

    st.dataframe(filtered, use_container_width=True, height=460)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name="filtered_delinquency_customers.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    '<div class="small">Architecture: User → Streamlit dashboard → FastAPI → Random Forest → JSON response → visual analytics.</div>',
    unsafe_allow_html=True
)
