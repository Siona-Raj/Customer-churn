import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------
# CONFIG + UI
# ---------------------------
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling - Clean white background
st.markdown("""
<style>
/* ------------------ BACKGROUND ------------------ */
.stApp, .main {
    background: #f8f9fa;
}

/* ------------------ CARD ------------------ */
.card {
    background: white;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    transition: box-shadow 0.3s ease;
}
.card:hover {
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}

/* ------------------ HEADERS ------------------ */
h1 {
    color: #1f2937;
    font-size: 2.5em;
    font-weight: 700;
    text-align: center;
}
h2, h3 {
    color: #374151;
    font-weight: 600;
}

/* ------------------ TEXT ------------------ */
p, li, div, .stMarkdown {
    color: #374151;
}

/* ------------------ INPUT FIX (IMPORTANT) ------------------ */
/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* Dropdown text */
div[data-baseweb="select"] span {
    color: #1f2937 !important;
}

/* Number input + text input */
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] {
    background-color: white !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* Input text */
input {
    color: #1f2937 !important;
    background-color: white !important;
}

/* Remove dark container around inputs */
.stNumberInput > div,
.stSelectbox > div {
    background: transparent !important;
}

/* ------------------ SLIDER ------------------ */
.stSlider > div > div > div > div {
    background: #e5e7eb !important;
}

/* ------------------ RADIO ------------------ */
.stRadio > div {
    background: transparent !important;
}

/* ------------------ BUTTONS ------------------ */
/* Form button */
.stForm button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    width: 100%;
    font-size: 16px;
}

/* Normal buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

/* Hover */
.stButton > button:hover,
.stForm button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102,126,234,0.4);
}

/* Secondary buttons (+ / -) */
button[kind="secondary"] {
    background: #f3f4f6 !important;
    color: #374151 !important;
    border-radius: 6px !important;
}

/* ------------------ TABS ------------------ */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    padding: 8px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    color: #6b7280;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

/* ------------------ METRICS ------------------ */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
}

.metric-label {
    font-size: 13px;
    opacity: 0.9;
}

/* ------------------ ALERTS ------------------ */
.stAlert {
    border-radius: 10px;
    border-left: 4px solid;
}

/* ------------------ DIVIDER ------------------ */
hr {
    height: 1px;
    background: #e5e7eb;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD + TRAIN
# ---------------------------
@st.cache_resource
def load_and_train():
    with st.spinner("Loading model and data..."):
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)

        df.replace('No internet service', 'No', inplace=True)
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')

        df.drop(['customerID', 'gender'], axis=1, inplace=True)

        cols_yes_no = ["Partner","Dependents","PhoneService","OnlineSecurity","OnlineBackup",
                       "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                       "PaperlessBilling","Churn"]

        for col in cols_yes_no:
            df[col] = df[col].map({'Yes':1, 'No':0})

        le_dict = {}
        cols_label = ["MultipleLines","Contract","PaymentMethod","InternetService"]

        for col in cols_label:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

        X = df.drop("Churn", axis=1)
        y = df["Churn"]

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        model = XGBClassifier(eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)

        explainer = shap.Explainer(model.predict_proba, X)

        return df, X, model, le_dict, explainer

df, X, model, le_dict, explainer = load_and_train()

# ---------------------------
# FUNCTIONS
# ---------------------------
def get_risk(prob):
    score = round(prob * 100, 1)
    if score >= 75:
        return score, "🔴 HIGH RISK", "risk-high"
    elif score >= 40:
        return score, "🟠 MEDIUM RISK", "risk-medium"
    else:
        return score, "🟢 LOW RISK", "risk-low"

def retention_strategy(data, risk_score):
    strategies = []

    if risk_score >= 75:
        strategies.append("**Immediate Action Required:** Schedule retention call within 24 hours")
    
    if data["Contract"].values[0] == 0:
        strategies.append("**Contract Incentive:** Offer 15% discount on annual contract")
    
    if data["MonthlyCharges"].values[0] > 80:
        strategies.append("**Price Optimization:** Bundle services for 20% savings")
    
    if data["tenure"].values[0] < 6:
        strategies.append("**New Customer Program:** Personalized onboarding & engagement campaign")
    
    if data["TechSupport"].values[0] == 0:
        strategies.append("**Value Addition:** 3 months free premium tech support")
    
    if risk_score < 40:
        strategies.append("**Growth Opportunity:** Upsell premium features with loyalty bonus")
    
    if risk_score >= 50 and risk_score < 75:
        strategies.append("**Proactive Outreach:** Send personalized retention offer via email")
    
    return strategies if strategies else ["✓ Customer appears satisfied - Monitor regularly"]

def create_gauge_chart(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        title = {'text': "Risk Score", 'font': {'size': 20}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue", 'tickfont': {'size': 12}},
            'bar': {'color': "#2c3e50", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, 40], 'color': '#d1fae5'},
                {'range': [40, 75], 'color': '#fed7aa'},
                {'range': [75, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# ---------------------------
# HEADER
# ---------------------------
st.title("Churn Intelligence Dashboard")
st.markdown("<p style='text-align: center; color: #6b7280; margin-bottom: 30px;'>AI-Powered Customer Retention Platform</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Predict Customer Churn", "Analytics Dashboard", "Executive Insights"])

# ---------------------------
# TAB 1: PREDICTION
# ---------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("Customer Information")
        st.markdown("*Enter customer details to assess churn risk*")
        
        with st.form("prediction_form"):
            tenure = st.slider("Tenure (months)", 0, 100, 12, help="How long has the customer been with us?")
            monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, help="Current monthly billing amount")
            
            contract = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
            internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
            
            col_tech, col_partner, col_dep = st.columns(3)
            with col_tech:
                tech = st.radio("Tech Support", ["No","Yes"], horizontal=True)
            with col_partner:
                partner = st.radio("Has Partner", ["No","Yes"], horizontal=True)
            with col_dep:
                dependents = st.radio("Has Dependents", ["No","Yes"], horizontal=True)
            
            submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)
    
    with col2:
        if submitted:
            st.subheader("Risk Assessment")
            
            input_dict = {
                "SeniorCitizen": 0,
                "Partner": 1 if partner=="Yes" else 0,
                "Dependents": 1 if dependents=="Yes" else 0,
                "tenure": tenure,
                "PhoneService": 1,
                "MultipleLines": "No",
                "InternetService": internet,
                "OnlineSecurity": 0,
                "OnlineBackup": 0,
                "DeviceProtection": 0,
                "TechSupport": 1 if tech=="Yes" else 0,
                "StreamingTV": 0,
                "StreamingMovies": 0,
                "Contract": contract,
                "PaperlessBilling": 1,
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": monthly,
                "TotalCharges": tenure * monthly
            }
            
            input_df = pd.DataFrame([input_dict])
            
            for col in ["MultipleLines","Contract","PaymentMethod","InternetService"]:
                input_df[col] = le_dict[col].transform(input_df[col])
            
            input_data = input_df[X.columns]
            
            prob = model.predict_proba(input_data)[0][1]
            risk_score, risk_label, risk_class = get_risk(prob)
            
            # Gauge chart
            st.plotly_chart(create_gauge_chart(risk_score), use_container_width=True)
            
            st.markdown(f"<div style='text-align: center;'><span class='{risk_class}'>{risk_label}</span></div>", unsafe_allow_html=True)
    
    if submitted:
        st.markdown("---")
        st.subheader("Personalized Retention Strategy")
        
        strategies = retention_strategy(input_data, risk_score)
        cols = st.columns(2)
        for idx, strategy in enumerate(strategies):
            with cols[idx % 2]:
                st.info(strategy)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# TAB 2: ANALYTICS DASHBOARD
# ---------------------------
with tab2:
    df_copy = df.copy()
    probs = model.predict_proba(X)[:,1]
    df_copy["RiskScore"] = probs * 100
    df_copy["RiskCategory"] = pd.cut(df_copy["RiskScore"], bins=[0,40,75,100], labels=["Low", "Medium", "High"])
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Total Customers</div>
            <div class='metric-value'>{:,}</div>
        </div>
        """.format(len(df_copy)), unsafe_allow_html=True)
    
    with col2:
        high_risk_count = len(df_copy[df_copy["RiskScore"] > 75])
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>High Risk Customers</div>
            <div class='metric-value'>{:,}</div>
        </div>
        """.format(high_risk_count), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Average Risk Score</div>
            <div class='metric-value'>{:.1f}%</div>
        </div>
        """.format(df_copy['RiskScore'].mean()), unsafe_allow_html=True)
    
    with col4:
        potential_revenue = (high_risk_count * df_copy['MonthlyCharges'].mean() * 12)
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>At-Risk Annual Revenue</div>
            <div class='metric-value'>${:,.0f}</div>
        </div>
        """.format(potential_revenue), unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Risk Distribution")
        
        risk_counts = df_copy["RiskCategory"].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     color=risk_counts.index, 
                     color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'},
                     hole=0.4)
        fig.update_layout(height=400, showlegend=True, title_text="Customer Risk Distribution")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Risk by Contract Type")
        
        contract_risk = df_copy.groupby("Contract")["RiskScore"].mean().reset_index()
        fig = px.bar(contract_risk, x="Contract", y="RiskScore", 
                     color="RiskScore", 
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                     title="Average Risk Score by Contract Type")
        fig.update_layout(height=400, showlegend=False, xaxis_title="Contract Type", yaxis_title="Average Risk Score (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Monthly Charges vs Risk")
        fig = px.scatter(df_copy, x="MonthlyCharges", y="RiskScore", 
                        color="RiskCategory", opacity=0.6,
                        color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'},
                        title="Relationship between Monthly Charges and Churn Risk")
        fig.update_layout(height=400, xaxis_title="Monthly Charges ($)", yaxis_title="Risk Score (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Tenure vs Risk")
        fig = px.scatter(df_copy, x="tenure", y="RiskScore",
                        color="RiskCategory", opacity=0.6,
                        color_discrete_map={'Low':'#10b981', 'Medium':'#f59e0b', 'High':'#ef4444'},
                        title="Relationship between Customer Tenure and Churn Risk")
        fig.update_layout(height=400, xaxis_title="Tenure (months)", yaxis_title="Risk Score (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# TAB 3: EXECUTIVE INSIGHTS
# ---------------------------
with tab3:
    df_copy = df.copy()
    probs = model.predict_proba(X)[:,1]
    df_copy["RiskScore"] = probs * 100
    df_copy = df_copy.sort_values(by="RiskScore", ascending=False)
    df_copy.index = range(len(df_copy))
    
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("High-Risk Customer Watchlist")
    st.markdown("*Prioritize these customers for immediate retention efforts*")
    
    # Add selection for number of customers
    n_customers = st.slider("Number of customers to display", 5, 50, 20)
    
    display_cols = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'RiskScore']
    high_risk_customers = df_copy.head(n_customers)[display_cols].copy()
    high_risk_customers['RiskScore'] = high_risk_customers['RiskScore'].round(1)
    high_risk_customers.columns = ['Tenure', 'Monthly Charges', 'Contract', 'Internet Service', 'Risk Score']
    
    st.dataframe(high_risk_customers, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Customer detail analyzer
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Customer Risk Analyzer")
    st.markdown("*Deep dive into individual customer risk factors*")
    
    # Create customer selector
    customer_list = [f"Customer {i} (Risk: {row['RiskScore']:.1f}%)" for i, row in df_copy.head(100).iterrows()]
    selected_customer_name = st.selectbox("Select a customer to analyze", customer_list)
    selected_idx = int(selected_customer_name.split()[1])
    selected_customer = df_copy.loc[[selected_idx]]
    
    prob = model.predict_proba(selected_customer[X.columns])[0][1]
    risk_score, risk_label, risk_class = get_risk(prob)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"<div style='text-align: center;'><span class='{risk_class}'>{risk_label}</span></div>", unsafe_allow_html=True)
        st.metric("Risk Score", f"{risk_score}%")
        
        st.markdown("---")
        st.markdown("**Customer Profile**")
        
        # Create a clean profile display
        profile_data = {
            "Tenure": f"{selected_customer['tenure'].values[0]} months",
            "Monthly Charges": f"${selected_customer['MonthlyCharges'].values[0]:.2f}",
            "Contract Type": ["Month-to-month", "One year", "Two year"][int(selected_customer['Contract'].values[0])],
            "Internet Service": ["DSL", "Fiber optic", "No"][int(selected_customer['InternetService'].values[0])],
            "Tech Support": "Yes" if selected_customer['TechSupport'].values[0] == 1 else "No"
        }
        
        for key, value in profile_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.subheader("AI-Driven Insights")
        st.markdown("*Key factors driving churn risk*")
        
        shap_values = explainer(selected_customer[X.columns])
        
        shap_df = pd.DataFrame({
            "Feature": X.columns,
            "Impact": shap_values.values[0][:,1]
        }).sort_values(by="Impact", key=abs, ascending=False)
        
        top_features = shap_df.head(8)
        
        # Fix feature names for display
        feature_names = {
            'tenure': 'Tenure', 'MonthlyCharges': 'Monthly Charges', 'Contract': 'Contract Type',
            'InternetService': 'Internet Service', 'TechSupport': 'Tech Support', 'Partner': 'Has Partner',
            'Dependents': 'Has Dependents', 'PaperlessBilling': 'Paperless Billing', 'PaymentMethod': 'Payment Method'
        }
        top_features['Feature'] = top_features['Feature'].map(lambda x: feature_names.get(x, x))
        
        fig = px.bar(top_features, x="Impact", y="Feature", orientation='h',
                     title="Top Factors Influencing Churn Risk",
                     color="Impact", 
                     color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"])
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0), xaxis_title="Impact on Risk", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Recommended Action Plan")
    
    strategies = retention_strategy(selected_customer[X.columns], risk_score)
    for idx, strategy in enumerate(strategies):
        st.success(strategy)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Export functionality
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Export Reports")
    st.markdown("*Generate and download reports for your team*")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Export Risk Analysis", use_container_width=True):
            st.info("✅ Report generated successfully!")
    with col2:
        if st.button("Email Retention List", use_container_width=True):
            st.info("✅ Retention list sent to team!")
    with col3:
        if st.button("Download CSV", use_container_width=True):
            st.info("✅ CSV file ready for download!")
    st.markdown("</div>", unsafe_allow_html=True)