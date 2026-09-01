import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Card Default Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Enhancement
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #212529;
    }
    .metric-sub {
        font-size: 12px;
        color: #28a745;
        font-weight: 500;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #1e293b;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    div.stButton > button:first-child {
        background-color: #1f77b4;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #145a8d;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown("<h1 style='text-align: center; color: #0f172a;'>💳 Credit Card Default Prediction & Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>An AI-driven dashboard for risk assessment and ML model evaluation</p>", unsafe_allow_html=True)
st.markdown("---")

# Navigation Tabs
tab1, tab2 = st.tabs(["📊 Model Leaderboard & Analytics", "🔮 Real-Time Risk Assessment"])

# ==================== TAB 1: LEADERBOARD & ANALYTICS ====================
with tab1:
    st.markdown("<div class='section-header'>🚀 Model Performance Summary</div>", unsafe_allow_html=True)
    
    # Custom HTML Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class='metric-card' style='border-left-color: #2563eb;'>
                <div class='metric-title'>Top Model</div>
                <div class='metric-value'>LightGBM</div>
                <div class='metric-sub'>Best Overall</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
            <div class='metric-card' style='border-left-color: #16a34a;'>
                <div class='metric-title'>Best ROC-AUC</div>
                <div class='metric-value'>0.7815</div>
                <div class='metric-sub'>+1.35% vs XGBoost</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
            <div class='metric-card' style='border-left-color: #0891b2;'>
                <div class='metric-title'>Accuracy</div>
                <div class='metric-value'>81.75%</div>
                <div class='metric-sub'>High Reliability</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class='metric-card' style='border-left-color: #9333ea;'>
                <div class='metric-title'>F1-Score</div>
                <div class='metric-value'>0.4651</div>
                <div class='metric-sub'>Optimal Threshold</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Comparison Table & ROC-AUC Chart
    col_left, col_right = st.columns([1, 1.2])
    
    with col_left:
        st.markdown("<div class='section-header'>📋 Model Leaderboard</div>", unsafe_allow_html=True)
        if os.path.exists("outputs/model_comparison.csv"):
            benchmark_data = pd.read_csv("outputs/model_comparison.csv")
        else:
            benchmark_data = pd.DataFrame({
                "Model": ["LightGBM", "XGBoost", "Random_Forest", "KNN"],
                "Accuracy": [0.8175, 0.8078, 0.8138, 0.7923],
                "F1-Score": [0.4651, 0.4517, 0.4606, 0.4290],
                "ROC-AUC": [0.7736, 0.7601, 0.7577, 0.7014]
            })

        # Highlight the best performing row completely
        def highlight_best_row(row):
            is_max = row['ROC-AUC'] == benchmark_data['ROC-AUC'].max()
            return ['background-color: #dcfce7; font-weight: bold; color: #166534;' if is_max else '' for _ in row]

        styled_df = benchmark_data.style.apply(highlight_best_row, axis=1).format({
            "Accuracy": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        })

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=210,
            hide_index=True
        )
        
    with col_right:
        st.markdown("<div class='section-header'>📈 ROC-AUC Score Comparison</div>", unsafe_allow_html=True)
        if os.path.exists("outputs/model_comparison_chart.png"):
            st.image("outputs/model_comparison_chart.png", use_container_width=True)
        else:
            st.info("Comparison chart image not found.")

    st.markdown("---")
    
    # Feature Importance & Confusion Matrix
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>🎯 Confusion Matrix (Best Model)</div>", unsafe_allow_html=True)
        if os.path.exists("outputs/confusion_matrix.png"):
            st.image("outputs/confusion_matrix.png", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")
            
    with c2:
        st.markdown("<div class='section-header'>⭐ Top 10 Feature Importance</div>", unsafe_allow_html=True)
        if os.path.exists("outputs/feature_importance.png"):
            st.image("outputs/feature_importance.png", use_container_width=True)
        else:
            st.info("Feature importance image not found.")

# ==================== TAB 2: REAL-TIME PREDICTION ====================
with tab2:
    st.markdown("<div class='section-header'>📋 Enter Client Profile Data</div>", unsafe_allow_html=True)
    
    try:
        preprocessor = joblib.load("preprocessor.pkl") if os.path.exists("preprocessor.pkl") else None
        model = joblib.load("models/best_model.pkl") if os.path.exists("models/best_model.pkl") else None

        col_dem, col_pay, col_fin = st.columns(3)
        
        with col_dem:
            st.subheader("👤 Demographic Profile")
            limit_bal = st.number_input("Limit Balance (LIMIT_BAL)", min_value=1000, max_value=1000000, value=50000, step=5000)
            sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = st.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: {1: "Grad School", 2: "University", 3: "High School", 4: "Others"}[x])
            marriage = st.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
            age = st.slider("Age", 18, 80, 30)

        with col_pay:
            st.subheader("⏱ Repayment History")
            pay_0 = st.selectbox("Sept Repayment Status (PAY_0)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_2 = st.selectbox("Aug Repayment Status (PAY_2)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_3 = st.selectbox("July Repayment Status (PAY_3)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            
            with st.expander("More Repayment Months (April - June)"):
                pay_4 = st.selectbox("June Repayment Status (PAY_4)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
                pay_5 = st.selectbox("May Repayment Status (PAY_5)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
                pay_6 = st.selectbox("April Repayment Status (PAY_6)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)

        with col_fin:
            st.subheader("💰 Recent Financials")
            bill_amt1 = st.number_input("Latest Bill Amount - Sept (BILL_AMT1)", value=20000.0, step=1000.0)
            pay_amt1 = st.number_input("Latest Paid Amount - Sept (PAY_AMT1)", value=1000.0, step=500.0)
            
            with st.expander("Previous Months Financials"):
                bill_amt2 = st.number_input("Bill Amount Aug (BILL_AMT2)", value=bill_amt1)
                pay_amt2 = st.number_input("Paid Amount Aug (PAY_AMT2)", value=pay_amt1)

        input_dict = {
            'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
            'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
            'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt1, 
            'BILL_AMT4': bill_amt1, 'BILL_AMT5': bill_amt1, 'BILL_AMT6': bill_amt1,
            'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt1, 
            'PAY_AMT4': pay_amt1, 'PAY_AMT5': pay_amt1, 'PAY_AMT6': pay_amt1
        }

        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🔮 Evaluate Default Risk", use_container_width=True):
            if model is None:
                st.error("Model file not found in 'models/best_model.pkl'")
            else:
                input_df = pd.DataFrame([input_dict])
                
                if preprocessor is not None:
                    processed_input = preprocessor.transform(input_df)
                else:
                    processed_input = input_df
                
                prediction = model.predict(processed_input)[0]
                probability = model.predict_proba(processed_input)[0][1]
                
                st.markdown("---")
                st.subheader("🎯 Risk Assessment Result")
                
                st.write(f"**Predicted Risk Probability:** `{probability * 100:.2f}%`")
                st.progress(float(probability))
                
                if prediction == 1 or probability > 0.5:
                    st.error(f"🚨 **HIGH RISK DETECTED**\nThe client is **likely to default** next month with a probability of **{probability*100:.2f}%**.")
                else:
                    st.success(f"✅ **LOW RISK CLIENT**\nThe client is **unlikely to default** next month. Default Probability: **{probability*100:.2f}%**.")

    except Exception as e:
        st.error(f"Error during execution: {e}")