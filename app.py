"""Generative AI-Powered Business Intelligence Dashboard - Customer Churn Intelligence."""
import os
import json
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from insight_engine import InsightEngine

st.set_page_config(page_title="GenAI BI Dashboard - Churn Intelligence", layout="wide")
BASE = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE, "outputs", "cleaned_telco.csv"))


@st.cache_data
def load_risk():
    p = os.path.join(BASE, "outputs", "risk_watchlist.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_metrics():
    p = os.path.join(BASE, "outputs", "metrics.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_model():
    p = os.path.join(BASE, "outputs", "churn_model.pkl")
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


df = load_data()
risk_df = load_risk()
metrics = load_metrics()
model = load_model()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Churn Analytics", "Risk Watchlist", "AI Assistant"])
st.sidebar.markdown("---")
st.sidebar.caption("GenAI-Powered BI Dashboard\nIBM SkillsBuild Internship 2026")

# ---------------------------------------------------------------- Overview
if page == "Overview":
    st.title("Customer Churn Intelligence - Overview")
    churn_rate = df["ChurnFlag"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{churn_rate:.1%}")
    c3.metric("Avg. monthly revenue", f"${df['MonthlyCharges'].mean():.2f}")
    rev_risk = metrics.get("revenue_at_risk_annual", 0)
    c4.metric("Annual revenue at risk", f"${rev_risk:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        by_contract = df.groupby("Contract")["ChurnFlag"].mean().reset_index()
        by_contract["Churn rate"] = by_contract["ChurnFlag"] * 100
        fig = px.bar(by_contract, x="Contract", y="Churn rate", color="Contract",
                     title="Churn rate by contract type")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x="tenure", color="Churn", nbins=30, barmode="overlay",
                           title="Tenure distribution by churn status",
                           color_discrete_map={"No": "#2ca02c", "Yes": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model performance")
    comp = metrics.get("model_comparison", [])
    if comp:
        st.dataframe(pd.DataFrame(comp).round(4), use_container_width=True)
    best = metrics.get("best_test_metrics", {})
    if best:
        st.caption("Tuned XGBoost (test set): " +
                   " | ".join(f"{k}={v:.4f}" for k, v in best.items()))

# ---------------------------------------------------------------- Churn Analytics
elif page == "Churn Analytics":
    st.title("Churn Analytics - Segment Deep-Dive")
    dim = st.selectbox("Segment by", ["Contract", "InternetService", "PaymentMethod",
                                      "tenure_band", "gender", "SeniorCitizen"])
    seg = df.groupby(dim)["ChurnFlag"].agg(["mean", "count"]).reset_index()
    seg.columns = [dim, "Churn rate", "Customers"]
    seg["Churn rate"] = seg["Churn rate"] * 100
    fig = px.bar(seg.sort_values("Churn rate", ascending=False), x=dim, y="Churn rate",
                 color="Churn rate", color_continuous_scale="Reds",
                 hover_data=["Customers"], title=f"Churn rate by {dim}")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                  title="Monthly charges by churn status",
                  color_discrete_map={"No": "#2ca02c", "Yes": "#d62728"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top churn drivers (SHAP)")
    drivers = metrics.get("top_churn_drivers", [])
    if drivers:
        st.write(pd.DataFrame({"Rank": range(1, len(drivers[:10]) + 1),
                               "Feature": drivers[:10]}).to_html(index=False),
                 unsafe_allow_html=True)

# ---------------------------------------------------------------- Risk Watchlist
elif page == "Risk Watchlist":
    st.title("Risk Watchlist - Customers Most Likely to Churn")
    if risk_df is None:
        st.warning("Watchlist not found. Run the notebook first to generate it.")
    else:
        st.caption(f"{len(risk_df)} customers flagged | "
                   f"${risk_df['MonthlyCharges'].sum() * 12:,.0f} annual revenue at stake")
        show = risk_df.copy()
        show["churn_risk"] = (show["churn_risk"] * 100).round(1).astype(str) + "%"
        st.dataframe(show.head(50), use_container_width=True)
        st.download_button("Download full watchlist (CSV)",
                           risk_df.to_csv(index=False),
                           file_name="risk_watchlist.csv", mime="text/csv")

# ---------------------------------------------------------------- AI Assistant
else:
    st.title("AI Assistant - Ask Your Data Anything")
    drivers = metrics.get("top_churn_drivers", [])
    engine = InsightEngine(df, risk_df=risk_df, top_drivers=drivers)

    st.markdown("**Try:** *Which contract type has the highest churn?* · "
                "*Why do customers churn?* · *How much revenue is at risk?*")
    q = st.text_input("Ask a question about the customer data")
    if st.button("Ask") and q.strip():
        with st.spinner("Analysing..."):
            st.markdown(engine.answer(q))

    st.markdown("---")
    if st.button("Generate executive summary"):
        with st.spinner("Generating briefing..."):
            summary = engine.executive_summary()
            polished = engine.enhance_with_llm(summary)
            st.text_area("Executive briefing", polished, height=380)
            if polished == summary:
                st.caption("Tip: set the OPENAI_API_KEY environment variable to enable LLM-polished briefings.")
            st.download_button("Download briefing", polished,
                               file_name="executive_summary.txt")
