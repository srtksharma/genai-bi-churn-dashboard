# GenAI BI Dashboard — Customer Churn Intelligence

An interactive Streamlit dashboard for customer churn analytics on the IBM Telco dataset (7,043 customers), built for the IBM SkillsBuild Data Analytics with AI Internship 2026.

## Live demo
https://genai-bi-churn-dashboard-xfxzsxy9pnqp8b8hu6ioau.streamlit.app

## Pages
- **Overview** — KPIs: churn rate, customers, revenue at risk
- **Churn Analytics** — churn by contract, tenure, payment method + SHAP drivers
- **Risk Watchlist** — top-200 at-risk customers, filterable and downloadable
- **AI Assistant** — natural-language Q&A over the project metrics (works without an API key)

## Results
- Overall churn rate: 26.5%
- Tuned XGBoost: ROC-AUC 0.845, F1 0.579
- Month-to-month contract churn: 42.7% vs 2.8% on two-year contracts
- Top-200 watchlist annual revenue at risk: ~$203,047

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a public GitHub repo.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub → **New app**.
3. Select the repo, branch `main`, main file `app.py` → **Deploy**.
4. In **Advanced settings**, choose Python 3.12.
