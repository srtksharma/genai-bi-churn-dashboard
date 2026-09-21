import os
import pandas as pd
import numpy as np


class InsightEngine:
    # Natural-language analytics assistant over the Telco churn dataset.
    # Works fully offline on pre-computed aggregates; optionally polishes
    # summaries with an LLM when OPENAI_API_KEY is set.

    def __init__(self, df, risk_df=None, top_drivers=None):
        self.df = df.copy()
        self.risk_df = risk_df
        self.top_drivers = top_drivers or []
        d = self.df
        self.m = {}
        self.m['n_customers'] = len(d)
        self.m['churn_rate'] = float(d['ChurnFlag'].mean())
        self.m['avg_monthly'] = float(d['MonthlyCharges'].mean())
        self.m['avg_tenure'] = float(d['tenure'].mean())
        self.m['by_contract'] = d.groupby('Contract')['ChurnFlag'].mean().sort_values(ascending=False)
        self.m['by_internet'] = d.groupby('InternetService')['ChurnFlag'].mean().sort_values(ascending=False)
        self.m['by_payment'] = d.groupby('PaymentMethod')['ChurnFlag'].mean().sort_values(ascending=False)
        self.m['by_tenure_band'] = d.groupby('tenure_band', observed=True)['ChurnFlag'].mean().sort_values(ascending=False)
        self.m['avg_monthly_churned'] = float(d.loc[d['ChurnFlag'] == 1, 'MonthlyCharges'].mean())
        self.m['avg_monthly_retained'] = float(d.loc[d['ChurnFlag'] == 0, 'MonthlyCharges'].mean())

    def _pct(self, x):
        return '%.1f%%' % (x * 100)

    def _series_to_text(self, s, title):
        lines = [title]
        for k, v in s.items():
            lines.append('  - %s: %s' % (k, self._pct(v)))
        return chr(10).join(lines)

    def answer(self, question):
        q = question.lower()
        m = self.m
        if 'contract' in q:
            return self._series_to_text(m['by_contract'], 'Churn rate by contract type:')
        if any(w in q for w in ['internet', 'dsl', 'fiber']):
            return self._series_to_text(m['by_internet'], 'Churn rate by internet service:')
        if any(w in q for w in ['payment', 'billing']):
            return self._series_to_text(m['by_payment'], 'Churn rate by payment method:')
        if any(w in q for w in ['tenure', 'new customer', 'loyal']):
            return self._series_to_text(m['by_tenure_band'], 'Churn rate by tenure band (months):')
        if any(w in q for w in ['why', 'driver', 'reason', 'cause', 'important', 'predictor']):
            if self.top_drivers:
                lines = ['Top drivers of churn (from SHAP explainability analysis):']
                for i, f in enumerate(self.top_drivers[:8], 1):
                    lines.append('  %d. %s' % (i, f))
                return chr(10).join(lines)
            return 'Driver analysis becomes available after the SHAP step runs.'
        if any(w in q for w in ['risk', 'watchlist', 'likely to churn', 'at risk']):
            n = len(self.risk_df) if self.risk_df is not None else 0
            seg = m['by_contract'].index[0]
            return ('The predictive model flags %d customers as high churn risk. The riskiest segment is '
                    '%s contracts at %s churn.' % (n, seg, self._pct(m['by_contract'].iloc[0])))
        if any(w in q for w in ['revenue', 'money', 'loss', 'at stake']):
            at_risk = float(self.risk_df['MonthlyCharges'].sum() * 12) if self.risk_df is not None else 0.0
            uplift = (m['avg_monthly_churned'] - m['avg_monthly_retained']) / m['avg_monthly_retained']
            return ('High-risk customers represent about $%s in annual recurring revenue at stake. '
                    'Churned customers paid %s more per month on average than retained ones.'
                    % ('{:,.0f}'.format(at_risk), self._pct(uplift)))
        if any(w in q for w in ['churn rate', 'how many churn', 'overall', 'total churn']):
            n_churn = int((self.df['ChurnFlag'] == 1).sum())
            return ('Overall churn rate is %s (%d of %d customers).'
                    % (self._pct(m['churn_rate']), n_churn, m['n_customers']))
        return ('I can answer questions about churn by contract, internet service, payment method, '
                'tenure, revenue at risk, churn drivers and the high-risk watchlist. '
                'Try: "Which contract type has the highest churn?"')

    def executive_summary(self):
        m = self.m
        L = []
        L.append('EXECUTIVE SUMMARY - CUSTOMER CHURN INTELLIGENCE')
        L.append('')
        L.append('Overall churn stands at %s across %d customers, with average tenure of %.1f months '
                 'and average monthly revenue of $%.2f per customer.'
                 % (self._pct(m['churn_rate']), m['n_customers'], m['avg_tenure'], m['avg_monthly']))
        L.append('')
        L.append('Key findings:')
        L.append('1. Contract type is the strongest lever: %s customers churn at %s, far above longer '
                 'commitments. Migrating customers to annual contracts is the single biggest retention opportunity.'
                 % (m['by_contract'].index[0], self._pct(m['by_contract'].iloc[0])))
        L.append('2. %s is the riskiest payment method at %s churn - the billing experience may be a contributing factor.'
                 % (m['by_payment'].index[0], self._pct(m['by_payment'].iloc[0])))
        L.append('3. New customers are the most vulnerable: the %s-month tenure band churns at %s. '
                 'The first year needs a stronger onboarding programme.'
                 % (m['by_tenure_band'].index[0], self._pct(m['by_tenure_band'].iloc[0])))
        if self.top_drivers:
            L.append('4. The predictive model identifies %s as the strongest signals of churn.'
                     % ', '.join(self.top_drivers[:3]))
        if self.risk_df is not None:
            L.append('5. %d customers are flagged high-risk by the model and listed on the Risk Watchlist for proactive outreach.'
                     % len(self.risk_df))
        L.append('')
        L.append('Recommended actions: incentivise annual contracts, review the electronic-check billing journey, '
                 'and launch a first-90-days onboarding programme for new customers.')
        return chr(10).join(L)

    def enhance_with_llm(self, text):
        # Optional: polish the summary with an LLM when an API key is available.
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return text
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': 'You are a business analyst. Rewrite the briefing below in polished executive language without changing any numbers.'},
                    {'role': 'user', 'content': text}],
                temperature=0.3)
            return resp.choices[0].message.content
        except Exception as e:
            return text + chr(10) + chr(10) + '[LLM enhancement skipped: %s]' % e
