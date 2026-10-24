import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------
# App Config & Aesthetic CSS
# ---------------------------
st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📶", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #3b82f6;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Caching Data & Models Function
# ---------------------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.replace('No internet service', 'No', inplace=True)
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No') # align with ML code
    df.drop(['customerID', 'gender'], axis=1, inplace=True)

    cols_yes_no = ["Partner","Dependents","PhoneService","OnlineSecurity","OnlineBackup",
            "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
            "PaperlessBilling","Churn"]
    
    for col in cols_yes_no:
        df[col] = df[col].replace({'Yes':1, 'No':0})

    le_dict = {}
    cols_label = ["MultipleLines","Contract","PaymentMethod","InternetService"]
    for col in cols_label:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Train Models
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    # Evaluate
    results = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "Confusion": confusion_matrix(y_test, pred)
        }
    
    return df, X, models, results, le_dict

with st.spinner("Initializing system and loading AI models (this is fast after the first run!)..."):
    df, X, models, results, le_dict = load_and_train()

# ---------------------------
# Retention Strategy Function
# ---------------------------
def retention_strategy(input_df):
    strategies = []
    
    if input_df["Contract"].values[0] == 0: # Month-to-month
        strategies.append("💡 **Offer a discount** or free upgrade for switching to a One or Two-year contract.")
    
    if input_df["MonthlyCharges"].values[0] > 70:
        strategies.append("🎁 **Provide a specialized discounted bundle** to reduce their monthly burden without cutting services.")
        
    if input_df["tenure"].values[0] < 12:
        strategies.append("🤝 **Offer onboarding benefits**, early loyalty rewards, or dedicated support to stabilize new customers.")
        
    if input_df["TechSupport"].values[0] == 0:
        strategies.append("🛠️ **Recommend adding Tech Support** to their plan, as customers with robust tech support churn less.")
        
    if input_df["InternetService"].values[0] == 1: # Fiber optic
        strategies.append("⚡ **Fiber Optic Quality Control:** Reach out to ensure network reliability and offer streaming bundle perks.")

    if len(strategies) == 0:
        strategies.append("✨ **Customer is relatively stable**. Check in occasionally and offer standard loyalty perks.")

    return strategies

# ---------------------------
# Header
# ---------------------------
st.title("Telecom Customer Churn Prediction & Retention System")
st.markdown("Predict potential churn risks, compare high-performance ML models, and formulate data-driven retention strategies with this intelligent dashboard.")

# ---------------------------
# UI Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Churn Prediction", "📊 Model Comparison", "📈 Insights & Visualizations"])

# ---------------------------
# TAB 1: Prediction
# ---------------------------
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Customer Profile Configuration")
    st.markdown("Fill out the details below to evaluate the churn risk of a specific customer.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Demographics")
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            
        with col2:
            st.markdown("#### 🛠️ Services")
            phone = st.selectbox("Phone Service", ["No", "Yes"])
            multilines = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_sec = st.selectbox("Online Security", ["No", "Yes"])
            online_bkp = st.selectbox("Online Backup", ["No", "Yes"])
            device_prot = st.selectbox("Device Protection", ["No", "Yes"])
            tech_sup = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes"])
            
        with col3:
            st.markdown("#### 💳 Account Details")
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly))
            
            st.markdown("---")
            model_choice = st.selectbox("Select Prediction Engine", ["Random Forest", "XGBoost", "Logistic Regression"], index=1)
            
        submit_btn = st.form_submit_button("Predict Churn Risk")
        
    st.markdown("</div>", unsafe_allow_html=True)

    if submit_btn:
        # Data Mappings
        yes_no_map = {"No": 0, "Yes": 1}
        
        # Prepare input Dataframe
        input_dict = {
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": yes_no_map[partner],
            "Dependents": yes_no_map[dependents],
            "tenure": tenure,
            "PhoneService": yes_no_map[phone],
            "MultipleLines": multilines, # Will encode
            "InternetService": internet, # Will encode
            "OnlineSecurity": yes_no_map[online_sec],
            "OnlineBackup": yes_no_map[online_bkp],
            "DeviceProtection": yes_no_map[device_prot],
            "TechSupport": yes_no_map[tech_sup],
            "StreamingTV": yes_no_map[streaming_tv],
            "StreamingMovies": yes_no_map[streaming_mov],
            "Contract": contract, # Will encode
            "PaperlessBilling": yes_no_map[paperless],
            "PaymentMethod": payment, # Will encode
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Encode categorical features exactly as trained
        for col in ["MultipleLines", "Contract", "PaymentMethod", "InternetService"]:
            if input_df[col][0] in le_dict[col].classes_:
                input_df[col] = le_dict[col].transform(input_df[col])
            else:
                # Fallback if somehow selection isn't mapped perfectly
                st.warning(f"Feature {col} value not recognized perfectly. Using default.")
                input_df[col] = 0
                
        # Ensure correct column order
        input_data = input_df[X.columns]
        
        # Prediction
        model = models[model_choice]
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None
        
        # Result Display
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader(" Prediction Results")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if pred == 1:
                st.error("⚠️ HIGH RISK: Customer likely to churn")
                if prob is not None:
                    st.metric("Churn Probability", f"{prob*100:.1f}%", "- Critical Risk", delta_color="inverse")
            else:
                st.success("✅ SAFE: Customer likely to stay")
                if prob is not None:
                    st.metric("Churn Probability", f"{prob*100:.1f}%", "+ Stable Customer")
                    
        with res_col2:
            if pred == 1:
                st.info("### 🛡️ Recommended Retention Actions:")
                strategies = retention_strategy(input_data)
                for s in strategies:
                    st.write(s)
            else:
                st.info("### 🎉 Customer Status:")
                st.write("Customer exhibits low churn probability. Continue delivering excellent service and look out for cross-selling opportunities.")
                
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# TAB 2: Model Comparison
# ---------------------------
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(" Machine Learning Model Benchmarks")
    st.markdown("Evaluate and compare the performance metrics across all integrated AI models.")

    comp_df = pd.DataFrame({k: {k2: v2 for k2, v2 in v.items() if k2 != "Confusion"} for k, v in results.items()}).T
    styled_df = comp_df.style.highlight_max(axis=0, color='#14532d') \
                          .set_properties(**{'color': 'white'})
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(res["Confusion"], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# TAB 3: Visualizations
# ---------------------------
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Exploratory Insights")
    
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.markdown("**Churn Distribution**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Churn', data=df, ax=ax1, palette='Set2')
        st.pyplot(fig1)

    with vis_col2:
        st.markdown("**Monthly Charges vs Churn**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax2, palette='Set2')
        st.pyplot(fig2)
        
    st.markdown("</div>", unsafe_allow_html=True)