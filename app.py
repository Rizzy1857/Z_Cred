import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import shap
import matplotlib.pyplot as plt

# Import custom modules
from generate_data import generate_synthetic_data
from model_pipeline import train_and_save_models, load_models, predict_default_probability
from local_db import (init_db, insert_applicant, get_applicants, update_applicant_scores, 
                     get_single_applicant, get_sync_status, sync_data_to_csv, log_consent)
from kfs_generator import generate_kfs_for_applicant
from shap_explain import generate_shap_explanations, explain_applicant

# Page configuration
st.set_page_config(
    page_title="Z-Score Credit Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'offline_mode' not in st.session_state:
    st.session_state.offline_mode = True
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = "AGENT001"

def load_models():
    """Load trained models using the model_pipeline module"""
    try:
        from model_pipeline import load_models as pipeline_load_models
        lr_model, xgb_model, metrics = pipeline_load_models()
        return lr_model, xgb_model, metrics
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, {}

def calculate_trust_scores(applicant_data):
    """Calculate trust scores according to Phase 2 Trust Bar Logic specification"""
    
    # Behavioral Trust = min(100, (on_time_payment_ratio * 100) - (avg_bill_delay_days * 2))
    behavioral_trust = min(100, max(0, 
        (applicant_data.get('on_time_payment_ratio', 0.5) * 100) - 
        (applicant_data.get('avg_bill_delay_days', 10) * 2)
    ))
    
    # Social Trust = community_endorsements * 20
    social_trust = min(100, applicant_data.get('community_endorsements', 0) * 20)
    
    # Digital Trace = (sim_card_tenure_months / 60 * 50) + (stable_location_ratio * 50)
    digital_trace = min(100, 
        (applicant_data.get('sim_card_tenure_months', 0) / 60 * 50) + 
        (applicant_data.get('stable_location_ratio', 0.5) * 50)
    )
    
    # Total Trust Score = average of the three segments
    total_trust_score = (behavioral_trust + social_trust + digital_trace) / 3
    
    return {
        'behavioral_trust': behavioral_trust,
        'social_trust': social_trust,
        'digital_trace': digital_trace,
        'total_trust_score': total_trust_score,
        'graduation_threshold_met': total_trust_score >= 70
    }

def predict_default_probability(applicant_data, lr_model=None, xgb_model=None):
    """Predict default probability using trained models"""
    if lr_model is None or xgb_model is None:
        from model_pipeline import predict_default_probability as pipeline_predict
        return pipeline_predict(applicant_data)
    
    # Legacy compatibility - prepare features
    features = pd.DataFrame([{
        'avg_bill_delay_days': applicant_data.get('avg_bill_delay_days', 0),
        'on_time_payment_ratio': applicant_data.get('on_time_payment_ratio', 0.8),
        'prev_loans_taken': applicant_data.get('prev_loans_taken', 0),
        'prev_loans_defaulted': applicant_data.get('prev_loans_defaulted', 0),
        'community_endorsements': applicant_data.get('community_endorsements', 2),
        'sim_card_tenure_months': applicant_data.get('sim_card_tenure_months', 12),
        'recharge_frequency_per_month': applicant_data.get('recharge_frequency_per_month', 10),
        'stable_location_ratio': applicant_data.get('stable_location_ratio', 0.8)
    }])
    
    lr_prob = lr_model.predict_proba(features)[0][1]
    xgb_prob = xgb_model.predict_proba(features)[0][1]
    
    return {
        'score_logistic': lr_prob,
        'score_xgb': xgb_prob,
        'average_score': (lr_prob + xgb_prob) / 2
    }

def main():
    st.title("🏦 Z-Score Credit Assessment Platform")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dashboard", 
        "👤 Applicant Onboarding", 
        "🎯 Risk Scoring", 
        "🔍 Explainability",
        "📄 KFS Generator",
        "📡 Offline Mode",
        "🗄️ Database"
    ])
    
    # Offline mode toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Connection Mode")
    offline_mode = st.sidebar.toggle("Offline Mode", value=st.session_state.offline_mode)
    st.session_state.offline_mode = offline_mode
    
    if offline_mode:
        st.sidebar.success("📴 Operating in Offline Mode")
        # Show sync status
        sync_status = get_sync_status()
        st.sidebar.metric("Unsynced Records", sync_status['unsynced_records'])
    else:
        st.sidebar.success("🌐 Online Mode")
        if st.sidebar.button("🔄 Sync Now"):
            synced_count = sync_data_to_csv()
            if synced_count > 0:
                st.sidebar.success(f"✅ Synced {synced_count} records")
            else:
                st.sidebar.info("📡 No records to sync")
    
    # Initialize database
    init_db()
    
    # Route to appropriate page based on Phase 2 specification
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "👤 Applicant Onboarding":
        onboarding_page()
    elif page == "🎯 Risk Scoring":
        risk_scoring_page()
    elif page == "🔍 Explainability":
        explainability_page()
    elif page == "📄 KFS Generator":
        kfs_generator_page()
    elif page == "📡 Offline Mode":
        offline_mode_page()
    elif page == "🗄️ Database":
        database_page()
    
    # Legacy page mappings
    elif page == "New Applicant Assessment":
        onboarding_page()
    elif page == "Applicant Database":
        database_page()
    elif page == "Model Management":
        model_management_page()
    elif page == "Analytics":
        analytics_page()

def dashboard_page():
    st.header("📊 Dashboard")
    
    # Check if data and models exist
    data_exists = os.path.exists('data/applicants.csv')
    models_exist = os.path.exists('models/lr_model.pkl') and os.path.exists('models/xgb_model.pkl')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if data_exists:
            st.success("✅ Data Available")
            df = pd.read_csv('data/applicants.csv')
            st.metric("Total Applicants", len(df))
        else:
            st.error("❌ No Data Found")
            st.metric("Total Applicants", 0)
    
    with col2:
        if models_exist:
            st.success("✅ Models Trained")
        else:
            st.error("❌ Models Not Trained")
    
    with col3:
        db_applicants = get_applicants()
        st.metric("Database Records", len(db_applicants))
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Generate Sample Data"):
            with st.spinner("Generating synthetic data..."):
                if not os.path.exists('data'):
                    os.makedirs('data')
                df = generate_synthetic_data()
                df.to_csv('data/applicants.csv', index=False)
                st.success("Sample data generated successfully!")
                st.rerun()
    
    with col2:
        if st.button("Train Models"):
            if data_exists:
                with st.spinner("Training models..."):
                    train_and_save_models()
                    st.success("Models trained successfully!")
                    st.rerun()
            else:
                st.error("Please generate data first!")
    
    with col3:
        if st.button("Reset System"):
            # Clear data and models
            if os.path.exists('data/applicants.csv'):
                os.remove('data/applicants.csv')
            if os.path.exists('models/lr_model.pkl'):
                os.remove('models/lr_model.pkl')
            if os.path.exists('models/xgb_model.pkl'):
                os.remove('models/xgb_model.pkl')
            st.success("System reset successfully!")
            st.rerun()

def assessment_page():
    """Legacy assessment page - redirects to onboarding"""
    st.info("🔄 This page has been updated. Redirecting to new Onboarding page...")
    onboarding_page()

def database_page():
    st.header("🗄️ Applicant Database")
    
    # Get all applicants
    df = get_applicants()
    
    if len(df) == 0:
        st.info("No applicants in database yet.")
        return
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applicants", len(df))
    with col2:
        approved = len(df[df['status'] == 'Approved'])
        st.metric("Approved", approved)
    with col3:
        rejected = len(df[df['status'] == 'Rejected'])
        st.metric("Rejected", rejected)
    with col4:
        under_review = len(df[df['status'] == 'Review Required'])
        st.metric("Under Review", under_review)
    
    # Filter options
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        status_filter = st.selectbox("Filter by Status", ["All", "Approved", "Rejected", "Review Required"])
    with col2:
        score_range = st.slider("Trust Score Range", 0.0, 100.0, (0.0, 100.0))
    
    # Apply filters
    filtered_df = df.copy()
    if status_filter != "All":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    if 'trust_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['trust_score'] >= score_range[0]) & 
            (filtered_df['trust_score'] <= score_range[1])
        ]
    
    # Display filtered data
    st.subheader("Applicant Records")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Individual applicant details
    if len(filtered_df) > 0:
        st.subheader("Individual Applicant Details")
        selected_id = st.selectbox("Select Applicant ID", filtered_df['applicant_id'].values)
        
        if st.button("View Details"):
            applicant = get_single_applicant(selected_id)
            if len(applicant) > 0:
                applicant_data = applicant.iloc[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Personal Information:**")
                    st.write(f"Name: {applicant_data.get('applicant_name', 'N/A')}")
                    st.write(f"Status: {applicant_data.get('status', 'N/A')}")
                    st.write(f"Consent Given: {'Yes' if applicant_data.get('consent_given', 0) else 'No'}")
                
                with col2:
                    st.write("**Risk Scores:**")
                    st.write(f"Logistic Regression PD: {applicant_data.get('score_logistic', 0):.2%}")
                    st.write(f"XGBoost PD: {applicant_data.get('score_xgb', 0):.2%}")
                    st.write(f"Trust Score: {applicant_data.get('trust_score', 0):.1f}")

def model_management_page():
    st.header("🤖 Model Management")
    
    # Model status
    lr_exists = os.path.exists('models/lr_model.pkl')
    xgb_exists = os.path.exists('models/xgb_model.pkl')
    data_exists = os.path.exists('data/applicants.csv')
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Status")
        st.write(f"Logistic Regression: {'✅ Available' if lr_exists else '❌ Not Found'}")
        st.write(f"XGBoost: {'✅ Available' if xgb_exists else '❌ Not Found'}")
        st.write(f"Training Data: {'✅ Available' if data_exists else '❌ Not Found'}")
    
    with col2:
        st.subheader("Quick Actions")
        
        if not data_exists:
            if st.button("Generate Training Data"):
                with st.spinner("Generating data..."):
                    if not os.path.exists('data'):
                        os.makedirs('data')
                    df = generate_synthetic_data()
                    df.to_csv('data/applicants.csv', index=False)
                    st.success("Training data generated!")
                    st.rerun()
        
        if data_exists and st.button("Train Models"):
            with st.spinner("Training models..."):
                try:
                    train_and_save_models()
                    st.success("Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
        
        if lr_exists and xgb_exists and st.button("Retrain Models"):
            with st.spinner("Retraining models..."):
                try:
                    train_and_save_models()
                    st.success("Models retrained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error retraining models: {str(e)}")
    
    # Training data overview
    if data_exists:
        st.subheader("Training Data Overview")
        df = pd.read_csv('data/applicants.csv')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            default_rate = df['default'].mean()
            st.metric("Default Rate", f"{default_rate:.2%}")
        with col3:
            missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
            st.metric("Missing Data Rate", f"{missing_rate:.2%}")
        
        # Feature distribution
        st.subheader("Feature Distributions")
        feature = st.selectbox("Select Feature", df.columns[1:-1])  # Exclude ID and target
        
        fig = px.histogram(df, x=feature, color='default', title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)

def analytics_page():
    st.header("📈 Analytics")
    
    # Get database data
    df = get_applicants()
    
    if len(df) == 0:
        st.info("No data available for analytics. Please assess some applicants first.")
        return
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", len(df))
    with col2:
        if 'status' in df.columns:
            approval_rate = len(df[df['status'] == 'Approved']) / len(df)
            st.metric("Approval Rate", f"{approval_rate:.1%}")
    with col3:
        if 'score_logistic' in df.columns:
            avg_pd = df['score_logistic'].mean()
            st.metric("Avg. Default Probability", f"{avg_pd:.2%}")
    with col4:
        if 'trust_score' in df.columns:
            avg_trust = df['trust_score'].mean()
            st.metric("Avg. Trust Score", f"{avg_trust:.1f}")
    
    # Status distribution
    if 'status' in df.columns:
        st.subheader("Application Status Distribution")
        status_counts = df['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index, title="Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Score distributions
    if 'score_logistic' in df.columns and 'score_xgb' in df.columns:
        st.subheader("Risk Score Distributions")
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='score_logistic', title="Logistic Regression Scores")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='score_xgb', title="XGBoost Scores")
            st.plotly_chart(fig, use_container_width=True)
    
    # Trust score breakdown
    trust_cols = ['behavioral_trust', 'social_trust', 'digital_trace']
    available_trust_cols = [col for col in trust_cols if col in df.columns]
    
    if available_trust_cols:
        st.subheader("Trust Score Analysis")
        trust_data = df[available_trust_cols].mean()
        
        fig = px.bar(x=trust_data.index, y=trust_data.values, title="Average Trust Scores by Component")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

# === NEW PHASE 2 PAGES ===

def onboarding_page():
    """Page 1 - Applicant Onboarding (Phase 2 spec)"""
    st.header("👤 Applicant Onboarding")
    st.markdown("*Offline-first onboarding for field agents*")
    
    # Agent information
    col1, col2 = st.columns(2)
    with col1:
        agent_id = st.text_input("Agent ID", value=st.session_state.current_agent)
        st.session_state.current_agent = agent_id
    with col2:
        location = st.text_input("Location", placeholder="Assessment location")
    
    # Consent notice (multilingual placeholder)
    st.subheader("📋 Data Processing Consent")
    st.info("**English:** I consent to the processing of my data for credit assessment.\n"
           "**हिंदी:** मैं क्रेडिट मूल्यांकन के लिए अपने डेटा के प्रसंस्करण की सहमति देता हूं।")
    
    # Onboarding form
    with st.form("onboarding_form"):
        st.subheader("Applicant Information")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", placeholder="Enter full name as per ID")
            phone = st.text_input("Mobile Number", placeholder="+91XXXXXXXXXX")
            consent = st.checkbox("I agree to data processing for credit assessment", value=False)
        
        with col2:
            id_type = st.selectbox("ID Type", ["Aadhaar", "PAN", "Voter ID", "Driving License"])
            id_number = st.text_input("ID Number", placeholder="Enter ID number")
            
        # Alternative data fields (as per dataset schema)
        st.subheader("📊 Alternative Data Assessment")
        
        col1, col2 = st.columns(2)
        with col1:
            avg_bill_delay = st.number_input("Average Bill Payment Delay (days)", 
                                           min_value=0, max_value=30, value=5,
                                           help="How many days late are bill payments on average?")
            payment_ratio = st.slider("On-time Payment Ratio", 0.0, 1.0, 0.8, 0.01,
                                    help="What percentage of bills are paid on time?")
            prev_loans = st.number_input("Previous Loans Taken", min_value=0, max_value=10, value=1)
            prev_defaults = st.number_input("Previous Defaults", min_value=0, max_value=5, value=0)
        
        with col2:
            endorsements = st.number_input("Community Endorsements", min_value=0, max_value=5, value=2,
                                         help="Number of SHG/NGO endorsements")
            sim_tenure = st.number_input("SIM Card Tenure (months)", min_value=0, max_value=60, value=24)
            recharge_freq = st.number_input("Monthly Recharges", min_value=0, max_value=20, value=8)
            location_stability = st.slider("Location Stability Ratio", 0.0, 1.0, 0.8, 0.01,
                                          help="Percentage of time at stable location")
        
        notes = st.text_area("Additional Notes", placeholder="Any additional observations...")
        
        submitted = st.form_submit_button("💾 Save Applicant Data", type="primary")
        
        if submitted:
            if not name:
                st.error("❌ Please enter applicant name")
            elif not consent:
                st.error("❌ Consent is required for data processing")
            else:
                # Prepare applicant data
                applicant_data = {
                    'name': name,
                    'phone': phone,
                    'id_type': id_type,
                    'id_number': id_number,
                    'avg_bill_delay_days': avg_bill_delay,
                    'on_time_payment_ratio': payment_ratio,
                    'prev_loans_taken': prev_loans,
                    'prev_loans_defaulted': prev_defaults,
                    'community_endorsements': endorsements,
                    'sim_card_tenure_months': sim_tenure,
                    'recharge_frequency_per_month': recharge_freq,
                    'stable_location_ratio': location_stability,
                    'consent': 1,
                    'notes': notes
                }
                
                # Insert into database
                applicant_id = insert_applicant(applicant_data, agent_id=agent_id, location=location)
                
                st.success(f"✅ Applicant {name} onboarded successfully! ID: {applicant_id}")
                st.info("📴 Data saved locally. Will sync when online.")
                
                # Show what's next
                st.markdown("### Next Steps:")
                st.markdown("1. 🎯 Go to **Risk Scoring** to assess this applicant")
                st.markdown("2. 🔍 View **Explainability** for model insights")
                st.markdown("3. 📄 Generate **KFS** if approved")

def risk_scoring_page():
    """Page 2 - Risk Scoring (Phase 2 spec)"""
    st.header("🎯 Risk Scoring")
    st.markdown("*Select applicant and display PD from both models*")
    
    # Get applicants
    applicants_df = get_applicants()
    
    if len(applicants_df) == 0:
        st.warning("⚠️ No applicants found. Please onboard applicants first.")
        return
    
    # Applicant selection
    unprocessed_applicants = applicants_df[applicants_df['data_processed'] != 1]
    
    if len(unprocessed_applicants) == 0:
        st.info("ℹ️ All applicants have been processed.")
        processed_applicants = applicants_df[applicants_df['data_processed'] == 1]
        selected_applicant = st.selectbox("View Processed Applicant", 
                                        processed_applicants['applicant_name'].values)
        applicant_data = processed_applicants[processed_applicants['applicant_name'] == selected_applicant].iloc[0]
        show_existing_scores = True
    else:
        selected_applicant = st.selectbox("Select Applicant for Risk Assessment", 
                                        unprocessed_applicants['applicant_name'].values)
        applicant_data = unprocessed_applicants[unprocessed_applicants['applicant_name'] == selected_applicant].iloc[0]
        show_existing_scores = False
    
    # Load models
    lr_model, xgb_model, metrics = load_models()
    
    if lr_model is None:
        st.error("❌ Models not found. Please train models first.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models..."):
                try:
                    train_and_save_models()
                    st.success("✅ Models trained successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
        return
    
    # Display applicant information
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Applicant Details")
        st.write(f"**Name:** {applicant_data['applicant_name']}")
        st.write(f"**Agent:** {applicant_data.get('agent_id', 'Unknown')}")
        st.write(f"**Location:** {applicant_data.get('location', 'Not specified')}")
        st.write(f"**Consent:** {'✅ Given' if applicant_data.get('consent_given', 0) else '❌ Not given'}")
    
    with col2:
        st.subheader("📊 Input Features")
        st.write(f"**Bill Delay:** {applicant_data.get('avg_bill_delay_days', 'N/A')} days")
        st.write(f"**Payment Ratio:** {applicant_data.get('on_time_payment_ratio', 'N/A')}")
        st.write(f"**Previous Loans:** {applicant_data.get('prev_loans_taken', 'N/A')}")
        st.write(f"**Previous Defaults:** {applicant_data.get('prev_loans_defaulted', 'N/A')}")
        st.write(f"**Community Endorsements:** {applicant_data.get('community_endorsements', 'N/A')}")
    
    if show_existing_scores:
        # Show existing scores
        st.markdown("---")
        st.subheader("📈 Risk Assessment Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            lr_score = applicant_data.get('score_logistic', 0)
            st.metric("Logistic Regression PD", f"{lr_score:.2%}")
        with col2:
            xgb_score = applicant_data.get('score_xgb', 0)  
            st.metric("XGBoost PD", f"{xgb_score:.2%}")
        with col3:
            avg_score = applicant_data.get('average_score', (lr_score + xgb_score) / 2)
            risk_category = applicant_data.get('risk_category', 'Unknown')
            st.metric("Risk Category", risk_category)
        
        # Trust Bar visualization
        if applicant_data.get('total_trust_score') is not None:
            st.subheader("🏆 Trust Bar")
            
            behavioral = applicant_data.get('behavioral_trust', 0)
            social = applicant_data.get('social_trust', 0)
            digital = applicant_data.get('digital_trace', 0)
            total = applicant_data.get('total_trust_score', 0)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Behavioral Trust", f"{behavioral:.1f}/100")
            with col2:
                st.metric("Social Trust", f"{social:.1f}/100")
            with col3:
                st.metric("Digital Trace", f"{digital:.1f}/100")
            with col4:
                graduation = "🎓 Graduated" if total >= 70 else "📚 In Progress"
                st.metric("Total Score", f"{total:.1f}/100", delta=graduation)
            
            # Trust bar visualization
            trust_data = pd.DataFrame({
                'Component': ['Behavioral Trust', 'Social Trust', 'Digital Trace'],
                'Score': [behavioral, social, digital],
                'Target': [70, 70, 70]  # Graduation threshold
            })
            
            fig = px.bar(trust_data, x='Component', y='Score', 
                        title="Trust Score Components (Graduation Threshold: 70%)")
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                         annotation_text="Graduation Threshold")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Process new applicant
        if st.button("🎯 Assess Risk Now", type="primary"):
            with st.spinner("Running risk assessment..."):
                # Prepare data for prediction
                pred_data = {
                    'avg_bill_delay_days': applicant_data.get('avg_bill_delay_days'),
                    'on_time_payment_ratio': applicant_data.get('on_time_payment_ratio'),
                    'prev_loans_taken': applicant_data.get('prev_loans_taken'),
                    'prev_loans_defaulted': applicant_data.get('prev_loans_defaulted'),
                    'community_endorsements': applicant_data.get('community_endorsements'),
                    'sim_card_tenure_months': applicant_data.get('sim_card_tenure_months'),
                    'recharge_frequency_per_month': applicant_data.get('recharge_frequency_per_month'),
                    'stable_location_ratio': applicant_data.get('stable_location_ratio')
                }
                
                # Get predictions
                prediction_results = predict_default_probability(pred_data)
                trust_scores = calculate_trust_scores(pred_data)
                
                # Determine risk category and status
                avg_score = prediction_results.get('average_score', 0)
                if avg_score < 0.3:
                    risk_category = 'Low'
                    final_status = 'Approved'
                elif avg_score < 0.7:
                    risk_category = 'Medium'
                    final_status = 'Review Required'
                else:
                    risk_category = 'High'
                    final_status = 'Rejected'
                
                # Update database
                scores_data = {
                    'score_logistic': prediction_results.get('score_logistic'),
                    'score_xgb': prediction_results.get('score_xgb'),
                    'average_score': avg_score,
                    'behavioral_trust': trust_scores['behavioral_trust'],
                    'social_trust': trust_scores['social_trust'],
                    'digital_trace': trust_scores['digital_trace'],
                    'total_trust_score': trust_scores['total_trust_score']
                }
                
                update_applicant_scores(int(applicant_data['applicant_id']), scores_data)
                
                # Log scoring consent
                log_consent(int(applicant_data['applicant_id']), 'scoring', True, 
                           'Risk scoring completed with model predictions')
                
                st.success(f"✅ Risk assessment completed! Status: {final_status}")
                st.rerun()

def explainability_page():
    """Page 3 - Explainability (Phase 2 spec)"""
    st.header("🔍 Explainability")
    st.markdown("*Global and local SHAP explanations for transparency*")
    
    # Check if explanations exist
    explanations_exist = (os.path.exists('explanations/global_explanations.json') and 
                         os.path.exists('explanations/local_explanations.json'))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌍 Generate Global SHAP"):
            with st.spinner("Generating global explanations..."):
                try:
                    global_exp, local_exp = generate_shap_explanations()
                    st.success("✅ Global SHAP explanations generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating explanations: {e}")
    
    with col2:
        if explanations_exist:
            st.success("✅ SHAP explanations available")
        else:
            st.warning("⚠️ No explanations found")
    
    if explanations_exist:
        # Load explanations
        import json
        with open('explanations/global_explanations.json', 'r') as f:
            global_explanations = json.load(f)
        
        # Global explanations
        st.subheader("🌍 Global Feature Importance")
        
        model_choice = st.selectbox("Select Model", ["logistic_regression", "xgboost"])
        
        if model_choice in global_explanations:
            model_data = global_explanations[model_choice]
            
            # Create feature importance chart
            importance_df = pd.DataFrame({
                'Feature': model_data['features'],
                'Importance': model_data['importance']
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title=f"Global Feature Importance - {model_choice.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Local explanations for specific applicant
        st.subheader("👤 Local Explanations")
        
        # Get processed applicants
        applicants_df = get_applicants()
        processed_applicants = applicants_df[applicants_df['data_processed'] == 1]
        
        if len(processed_applicants) > 0:
            selected_applicant = st.selectbox("Select Applicant", 
                                            processed_applicants['applicant_name'].values)
            
            if st.button("🔍 Generate Local SHAP"):
                applicant_row = processed_applicants[processed_applicants['applicant_name'] == selected_applicant].iloc[0]
                
                # Prepare applicant data
                applicant_features = {
                    'avg_bill_delay_days': applicant_row.get('avg_bill_delay_days'),
                    'on_time_payment_ratio': applicant_row.get('on_time_payment_ratio'),
                    'prev_loans_taken': applicant_row.get('prev_loans_taken'),
                    'prev_loans_defaulted': applicant_row.get('prev_loans_defaulted'),
                    'community_endorsements': applicant_row.get('community_endorsements'),
                    'sim_card_tenure_months': applicant_row.get('sim_card_tenure_months'),
                    'recharge_frequency_per_month': applicant_row.get('recharge_frequency_per_month'),
                    'stable_location_ratio': applicant_row.get('stable_location_ratio')
                }
                
                try:
                    explanation = explain_applicant(applicant_features, model_type=model_choice)
                    
                    # Display prediction
                    st.metric("Prediction Probability", f"{explanation['prediction_probability']:.2%}")
                    
                    # Display top factors
                    st.subheader("🎯 Top Contributing Factors")
                    for i, (factor, impact) in enumerate(explanation['top_factors'][:5]):
                        direction = "📈 Increases" if impact > 0 else "📉 Decreases"
                        st.write(f"{i+1}. **{factor}**: {direction} risk by {abs(impact):.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error generating local explanation: {e}")
        else:
            st.info("ℹ️ No processed applicants found for local explanations.")

def kfs_generator_page():
    """Page 4 - KFS Generator (Phase 2 spec)"""
    st.header("📄 KFS Generator")
    st.markdown("*Generate Key Fact Statement with PD score, loan terms, and top feature impacts*")
    
    # Get approved applicants
    applicants_df = get_applicants()
    approved_applicants = applicants_df[
        (applicants_df['final_status'] == 'Approved') | 
        (applicants_df['risk_category'] == 'Low')
    ]
    
    if len(approved_applicants) == 0:
        st.warning("⚠️ No approved applicants found for KFS generation.")
        st.info("💡 Tip: Assess applicants in the Risk Scoring page first.")
        return
    
    # Applicant selection
    selected_applicant = st.selectbox("Select Approved Applicant", 
                                    approved_applicants['applicant_name'].values)
    
    applicant_data = approved_applicants[approved_applicants['applicant_name'] == selected_applicant].iloc[0]
    
    # Display applicant summary
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👤 Applicant Summary")
        st.write(f"**Name:** {applicant_data['applicant_name']}")
        st.write(f"**Risk Category:** {applicant_data.get('risk_category', 'N/A')}")
        st.write(f"**Status:** {applicant_data.get('final_status', 'N/A')}")
        
    with col2:
        st.subheader("📊 Risk Scores")
        lr_score = applicant_data.get('score_logistic', 0)
        xgb_score = applicant_data.get('score_xgb', 0)
        avg_score = (lr_score + xgb_score) / 2
        st.write(f"**Logistic Regression PD:** {lr_score:.2%}")
        st.write(f"**XGBoost PD:** {xgb_score:.2%}")
        st.write(f"**Average PD:** {avg_score:.2%}")
    
    # Trust score display
    if applicant_data.get('total_trust_score') is not None:
        st.subheader("🏆 Trust Score")
        total_trust = applicant_data.get('total_trust_score', 0)
        graduation_status = "🎓 Graduated (≥70%)" if total_trust >= 70 else "📚 In Progress (<70%)"
        st.write(f"**Total Trust Score:** {total_trust:.1f}/100 - {graduation_status}")
    
    # KFS generation
    st.markdown("---")
    st.subheader("📄 Generate KFS Document")
    
    col1, col2 = st.columns(2)
    with col1:
        include_trust_scores = st.checkbox("Include Trust Score breakdown", value=True)
        include_compliance = st.checkbox("Include full compliance section", value=True)
    
    with col2:
        kfs_language = st.selectbox("Document Language", ["English", "Hindi (हिंदी)", "Bilingual"])
        document_type = st.selectbox("Document Type", ["Standard KFS", "Detailed Assessment"])
    
    if st.button("📄 Generate KFS Document", type="primary"):
        with st.spinner("Generating KFS document..."):
            try:
                # Prepare data for KFS generation
                prediction_results = {
                    'score_logistic': lr_score,
                    'score_xgb': xgb_score,
                    'average_score': avg_score
                }
                
                # Prepare applicant data
                kfs_applicant_data = {
                    'name': applicant_data['applicant_name'],
                    'avg_bill_delay_days': applicant_data.get('avg_bill_delay_days'),
                    'on_time_payment_ratio': applicant_data.get('on_time_payment_ratio'),
                    'prev_loans_taken': applicant_data.get('prev_loans_taken'),
                    'prev_loans_defaulted': applicant_data.get('prev_loans_defaulted'),
                    'community_endorsements': applicant_data.get('community_endorsements'),
                    'sim_card_tenure_months': applicant_data.get('sim_card_tenure_months'),
                    'recharge_frequency_per_month': applicant_data.get('recharge_frequency_per_month'),
                    'stable_location_ratio': applicant_data.get('stable_location_ratio')
                }
                
                # Trust scores (if requested)
                trust_scores = None
                if include_trust_scores:
                    trust_scores = {
                        'behavioral_trust': applicant_data.get('behavioral_trust', 0),
                        'social_trust': applicant_data.get('social_trust', 0),
                        'digital_trace': applicant_data.get('digital_trace', 0),
                        'total_trust_score': applicant_data.get('total_trust_score', 0)
                    }
                
                # Generate KFS
                kfs_path = generate_kfs_for_applicant(kfs_applicant_data, prediction_results, trust_scores)
                
                if kfs_path:
                    st.success(f"✅ KFS document generated successfully!")
                    st.info(f"📄 Document saved as: {kfs_path}")
                    
                    # Log KFS generation
                    log_consent(int(applicant_data['applicant_id']), 'kfs_generation', True, 
                               f'KFS document generated: {kfs_path}')
                    
                    # Display next steps
                    st.markdown("### 📋 Next Steps:")
                    st.markdown("1. 📧 Share KFS with applicant")
                    st.markdown("2. ⏰ Allow 24-hour cooling-off period")
                    st.markdown("3. 📝 Collect final acceptance/rejection")
                    st.markdown("4. 💰 Process loan if accepted")
                    
                else:
                    st.error("❌ Failed to generate KFS document")
                    
            except Exception as e:
                st.error(f"❌ Error generating KFS: {e}")

def offline_mode_page():
    """Page 5 - Offline Mode (Phase 2 spec)"""
    st.header("📡 Offline Mode")
    st.markdown("*Toggle offline/online mode and sync data*")
    
    # Current mode status
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.offline_mode:
            st.success("📴 Currently in Offline Mode")
            st.info("✅ All operations are stored locally")
        else:
            st.success("🌐 Currently Online")
            st.info("✅ Data can be synced to central server")
    
    with col2:
        # Mode toggle
        new_mode = st.toggle("Enable Offline Mode", value=st.session_state.offline_mode)
        if new_mode != st.session_state.offline_mode:
            st.session_state.offline_mode = new_mode
            if new_mode:
                st.success("📴 Switched to Offline Mode")
            else:
                st.success("🌐 Switched to Online Mode")
            st.rerun()
    
    # Sync status and controls
    st.markdown("---")
    st.subheader("🔄 Data Synchronization")
    
    sync_status = get_sync_status()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", sync_status['total_records'])
    with col2:
        st.metric("Unsynced Records", sync_status['unsynced_records'])
    with col3:
        st.metric("Processed Records", sync_status['processed_records'])
    with col4:
        sync_pct = sync_status['sync_percentage']
        st.metric("Sync Progress", f"{sync_pct:.1f}%")
    
    # Sync controls
    if not st.session_state.offline_mode:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Sync Now", type="primary"):
                with st.spinner("Syncing data..."):
                    synced_count = sync_data_to_csv()
                    if synced_count > 0:
                        st.success(f"✅ Successfully synced {synced_count} records")
                        st.rerun()
                    else:
                        st.info("📡 No records to sync")
        
        with col2:
            if st.button("📊 View Sync Log"):
                # Show sync history (placeholder)
                st.info("📋 Sync log functionality available in full version")
    
    else:
        st.warning("⚠️ Sync is disabled in offline mode")
        st.info("💡 Switch to online mode to enable data synchronization")
    
    # Local data overview
    st.markdown("---")
    st.subheader("💾 Local Database Overview")
    
    applicants_df = get_applicants()
    
    if len(applicants_df) > 0:
        # Status breakdown
        status_counts = applicants_df['final_status'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Status Distribution")
            for status, count in status_counts.items():
                st.write(f"**{status}:** {count}")
        
        with col2:
            st.subheader("🕒 Recent Activity")
            recent = applicants_df.sort_values('created_timestamp', ascending=False).head(5)
            for _, row in recent.iterrows():
                st.write(f"• {row['applicant_name']} - {row.get('final_status', 'Pending')}")
    
    else:
        st.info("📝 No local data found. Start onboarding applicants!")
    
    # Data export/backup
    st.markdown("---")
    st.subheader("📁 Data Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Create Backup"):
            try:
                from local_db import export_database_backup
                backup_path = export_database_backup()
                st.success(f"✅ Backup created: {backup_path}")
            except Exception as e:
                st.error(f"❌ Backup failed: {e}")
    
    with col2:
        if st.button("🗑️ Clear Local Data"):
            if st.checkbox("I understand this will delete all local data"):
                try:
                    from local_db import reset_database
                    reset_database()
                    st.success("✅ Local database cleared")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Clear failed: {e}")
