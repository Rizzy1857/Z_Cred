import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import authentication and database modules
from auth import init_auth, require_auth, login_user, logout_user, register_user, get_user_stats
from local_db import init_db, get_applicants, get_single_applicant, insert_applicant, update_applicant_scores
from model_pipeline import train_and_save_models, predict_default_probability
import shap_explain
import kfs_generator

def load_css():
    """Load custom CSS for professional design."""
    st.markdown("""
    <style>
    /* Global Styles */
    .main > div {
        padding: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1f2937 0%, #374151 100%);
    }
    
    /* Custom sidebar */
    .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .nav-item {
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid transparent;
        backdrop-filter: blur(10px);
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.15);
        border-left: 4px solid #60a5fa;
        transform: translateX(8px);
        box-shadow: 0 4px 20px rgba(96, 165, 250, 0.3);
    }
    
    .nav-item.active {
        background: rgba(255, 255, 255, 0.2);
        border-left: 4px solid #3b82f6;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .applicant-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .applicant-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
    }
    
    .trust-bar {
        background: #f1f5f9;
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        margin: 12px 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .trust-fill {
        height: 100%;
        background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);
        transition: width 0.8s ease;
        position: relative;
    }
    
    .trust-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 40%, rgba(255,255,255,0.3) 50%, transparent 60%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .obscurity-bar {
        background: #f1f5f9;
        border-radius: 12px;
        height: 24px;
        overflow: hidden;
        margin: 12px 0;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .obscurity-fill {
        height: 100%;
        background: linear-gradient(90deg, #374151, #6b7280, #9ca3af);
        transition: width 0.8s ease;
    }
    
    /* Forms */
    .auth-container {
        max-width: 420px;
        margin: 3rem auto;
        padding: 2.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .form-header {
        text-align: center;
        margin-bottom: 2.5rem;
        color: #1f2937;
    }
    
    .form-header h2 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Risk category badges */
    .risk-low {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    .slide-in-left {
        animation: slideInLeft 0.6s ease-out;
    }
    
    /* Page headers */
    .page-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main > div {
            padding: 0.5rem;
        }
        
        .auth-container {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1.2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the navigation sidebar."""
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-content">
            <h2 style="color: white; text-align: center; margin-bottom: 0.5rem; font-weight: 700;">
                Z-Cred Platform
            </h2>
            <div style="color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 2rem; font-size: 0.9rem;">
                Welcome, {st.session_state.user['name']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        stats = get_user_stats(st.session_state.user['user_id'])
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">
                {stats['total_applicants']}
            </div>
            <div style="opacity: 0.9; font-size: 0.9rem;">Total Applicants</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">
                {stats['avg_trust_score']}%
            </div>
            <div style="opacity: 0.9; font-size: 0.9rem;">Average Trust Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem;">
                {stats['graduated_percentage']}%
            </div>
            <div style="opacity: 0.9; font-size: 0.9rem;">Trust Graduates</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu - Professional, no emojis
        nav_options = [
            ("Dashboard", "home"),
            ("New Applicant", "add_applicant"),
            ("Risk Analysis", "applicant_details"),
            ("AI Insights", "explainability"),
            ("Credit Report", "kfs"),
            ("Data Sync", "offline"),
            ("Settings", "settings")
        ]
        
        for label, key in nav_options:
            is_active = st.session_state.get('current_page', 'home') == key
            
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.current_page = key
                st.rerun()
        
        st.markdown("---")
        
        # Logout button
        if st.button("Sign Out", use_container_width=True, type="secondary"):
            logout_user()
            st.rerun()

def auth_page():
    """Authentication page for login/register."""
    st.markdown('<div class="auth-container fade-in">', unsafe_allow_html=True)
    
    # Tab for login/register
    tab1, tab2 = st.tabs(["Sign In", "Create Account"])
    
    with tab1:
        st.markdown('<div class="form-header"><h2>Welcome Back</h2><p>Sign in to your Z-Cred account</p></div>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if email and password:
                    if login_user(email, password):
                        st.success("Login successful!")
                        st.session_state.current_page = 'home'
                        st.rerun()
                    else:
                        st.error("Invalid email or password")
                else:
                    st.warning("Please fill in all fields")
    
    with tab2:
        st.markdown('<div class="form-header"><h2>Join Z-Cred</h2><p>Create your account to get started</p></div>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email Address", placeholder="Enter your email")
            phone = st.text_input("Phone Number (optional)", placeholder="Enter your phone number")
            password = st.text_input("Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
            
            terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if name and email and password and confirm_password:
                    if not terms:
                        st.error("Please accept the Terms of Service")
                    elif password != confirm_password:
                        st.error("Passwords do not match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, result = register_user(name, email, phone, password)
                        if success:
                            st.success("Account created successfully! You can now sign in.")
                        else:
                            st.error(f"Registration failed: {result}")
                else:
                    st.warning("Please fill in all required fields")
    
    st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    """User dashboard/home page."""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="page-header">Risk Assessment Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f"**Good day, {st.session_state.user['name']}!** Here's your portfolio overview.")
    st.markdown("---")
    
    # Get user's applicants
    user_applicants = get_applicants(user_id=st.session_state.user['user_id'])
    
    if len(user_applicants) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                    border-radius: 15px; margin: 2rem 0;">
            <h3>Ready to Begin Risk Assessment?</h3>
            <p style="color: #64748b; margin: 1rem 0;">Start building your applicant portfolio by adding your first assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Add First Applicant", type="primary", use_container_width=True):
            st.session_state.current_page = 'add_applicant'
            st.rerun()
    else:
        # Portfolio overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Assessments", len(user_applicants))
        with col2:
            avg_trust = user_applicants['total_trust_score'].mean()
            st.metric("Avg Trust Score", f"{avg_trust:.1f}%" if not pd.isna(avg_trust) else "N/A")
        with col3:
            high_risk = len(user_applicants[user_applicants['risk_category'] == 'High'])
            st.metric("High Risk", high_risk)
        with col4:
            approved = len(user_applicants[user_applicants['final_status'] == 'Approved'])
            st.metric("Approved", approved)
        
        st.markdown("---")
        
        # Display applicants in professional cards
        st.subheader("Your Applicants Portfolio")
        
        for idx, row in user_applicants.iterrows():
            st.markdown(f"""
            <div class="applicant-card slide-in-left">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0; color: #1f2937;">{row['applicant_name']}</h4>
                        <small style="color: #6b7280;">ID: {row['applicant_id']} • Added: {pd.to_datetime(row['created_timestamp']).strftime('%Y-%m-%d')}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                if pd.notna(row['total_trust_score']):
                    trust_score = row['total_trust_score']
                    st.markdown(f"""
                    <div class="trust-bar">
                        <div class="trust-fill" style="width: {trust_score}%"></div>
                    </div>
                    <small>Trust Score: {trust_score:.1f}%</small>
                    """, unsafe_allow_html=True)
                else:
                    st.write("Processing...")
            
            with col2:
                if pd.notna(row['risk_category']):
                    risk_class = f"risk-{row['risk_category'].lower()}"
                    st.markdown(f'<span class="{risk_class}">{row["risk_category"]} Risk</span>', 
                              unsafe_allow_html=True)
            
            with col3:
                if st.button("View", key=f"view_{row['applicant_id']}"):
                    st.session_state.selected_applicant = row['applicant_id']
                    st.session_state.current_page = 'applicant_details'
                    st.rerun()
            
            with col4:
                if pd.notna(row['total_trust_score']):
                    if st.button("Report", key=f"kfs_{row['applicant_id']}"):
                        st.session_state.selected_applicant = row['applicant_id']
                        st.session_state.current_page = 'kfs'
                        st.rerun()
        
        # Add new applicant button
        st.markdown("---")
        if st.button("Add New Applicant", type="primary", use_container_width=True):
            st.session_state.current_page = 'add_applicant'
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def add_applicant_page():
    """Add new applicant page."""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    st.markdown('<h1 class="page-header">New Applicant Assessment</h1>', unsafe_allow_html=True)
    st.markdown("Complete the assessment form to generate risk scores and trust analysis.")
    st.markdown("---")
    
    with st.form("add_applicant_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            name = st.text_input("Full Name*", placeholder="Enter applicant's full name")
            
            st.subheader("Financial Behavior")
            avg_bill_delay = st.number_input("Average Bill Delay (days)", min_value=0.0, value=0.0, step=0.1,
                                           help="Average number of days bills are paid late")
            on_time_ratio = st.slider("On-time Payment Ratio", 0.0, 1.0, 0.8, 0.01,
                                    help="Percentage of bills paid on time")
            prev_loans = st.number_input("Previous Loans Taken", min_value=0, value=0,
                                       help="Total number of previous loans")
            prev_defaults = st.number_input("Previous Loan Defaults", min_value=0, value=0,
                                         help="Number of previous loan defaults")
        
        with col2:
            st.subheader("Social & Digital Profile")
            community_endorsements = st.number_input("Community Endorsements", min_value=0, value=0,
                                                    help="Number of community endorsements or references")
            sim_tenure = st.number_input("SIM Card Tenure (months)", min_value=0, value=12,
                                       help="How long has the applicant had their current SIM card")
            recharge_freq = st.number_input("Monthly Recharge Frequency", min_value=0, value=4,
                                          help="Number of mobile recharges per month")
            location_stability = st.slider("Location Stability Ratio", 0.0, 1.0, 0.7, 0.01,
                                         help="Consistency of location patterns")
            
            st.subheader("Additional Notes")
            notes = st.text_area("Assessment Notes", placeholder="Any additional notes about the applicant")
        
        # Consent section
        st.markdown("---")
        st.subheader("Data Privacy & Consent")
        consent = st.checkbox("I confirm that the applicant has provided informed consent for data processing, risk assessment, and credit evaluation in accordance with data protection regulations.*")
        
        # Submit button
        submit = st.form_submit_button("Generate Risk Assessment", type="primary", use_container_width=True)
        
        if submit:
            if not name:
                st.error("Applicant name is required")
            elif not consent:
                st.error("Applicant consent must be confirmed to proceed")
            else:
                # Prepare applicant data
                applicant_data = {
                    'name': name,
                    'avg_bill_delay_days': avg_bill_delay,
                    'on_time_payment_ratio': on_time_ratio,
                    'prev_loans_taken': prev_loans,
                    'prev_loans_defaulted': prev_defaults,
                    'community_endorsements': community_endorsements,
                    'sim_card_tenure_months': sim_tenure,
                    'recharge_frequency_per_month': recharge_freq,
                    'stable_location_ratio': location_stability,
                    'consent': 1,
                    'notes': notes
                }
                
                # Insert applicant
                try:
                    applicant_id = insert_applicant(
                        applicant_data, 
                        created_by=st.session_state.user['user_id']
                    )
                    
                    # Calculate risk scores
                    with st.spinner("Calculating risk scores and trust analysis..."):
                        try:
                            # Prepare data for prediction
                            input_data = pd.DataFrame([{
                                'avg_bill_delay_days': avg_bill_delay,
                                'on_time_payment_ratio': on_time_ratio,
                                'prev_loans_taken': prev_loans,
                                'prev_loans_defaulted': prev_defaults,
                                'community_endorsements': community_endorsements,
                                'sim_card_tenure_months': sim_tenure,
                                'recharge_frequency_per_month': recharge_freq,
                                'stable_location_ratio': location_stability
                            }])
                            
                            # Calculate scores
                            scores = predict_default_probability(input_data.iloc[0])
                            
                            # Calculate trust components
                            behavioral_trust = (on_time_ratio * 40) + ((1 - min(avg_bill_delay/30, 1)) * 20)
                            social_trust = min(community_endorsements * 10, 50) + (location_stability * 30)
                            digital_trace = min(sim_tenure / 24 * 30, 30) + min(recharge_freq / 10 * 20, 20)
                            total_trust = behavioral_trust + social_trust + digital_trace
                            
                            scores_data = {
                                'score_logistic': scores.get('score_logistic', 0),
                                'score_xgb': scores.get('score_xgb', 0),
                                'behavioral_trust': behavioral_trust,
                                'social_trust': social_trust,
                                'digital_trace': digital_trace,
                                'total_trust_score': total_trust
                            }
                            
                            # Update applicant with scores
                            update_applicant_scores(applicant_id, scores_data)
                            
                            st.success(f"Risk assessment completed for {name}!")
                            st.balloons()
                            st.session_state.selected_applicant = applicant_id
                            st.session_state.current_page = 'applicant_details'
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error calculating scores: {str(e)}")
                            
                except Exception as e:
                    st.error(f"Error adding applicant: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def applicant_details_page():
    """Applicant details page."""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    selected_id = st.session_state.get('selected_applicant')
    if not selected_id:
        st.warning("No applicant selected. Please select an applicant from the dashboard.")
        if st.button("Go to Dashboard"):
            st.session_state.current_page = 'home'
            st.rerun()
        return
    
    # Get applicant data
    applicant_data = get_single_applicant(selected_id)
    if len(applicant_data) == 0:
        st.error("Applicant not found.")
        return
    
    applicant = applicant_data.iloc[0]
    
    st.markdown(f'<h1 class="page-header">Risk Analysis: {applicant["applicant_name"]}</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Basic info cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Applicant ID", applicant['applicant_id'])
    with col2:
        if pd.notna(applicant['risk_category']):
            risk_class = f"risk-{applicant['risk_category'].lower()}"
            st.markdown(f'**Risk Level:** <span class="{risk_class}">{applicant["risk_category"]}</span>', 
                      unsafe_allow_html=True)
    with col3:
        st.metric("Status", applicant.get('final_status', 'Pending'))
    with col4:
        if pd.notna(applicant['created_timestamp']):
            created_date = pd.to_datetime(applicant['created_timestamp']).strftime('%Y-%m-%d')
            st.metric("Assessed", created_date)
    
    # Trust and Obscurity Analysis
    if pd.notna(applicant['total_trust_score']):
        st.markdown("---")
        st.subheader("Trust & Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            trust_score = applicant['total_trust_score']
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h4 style="color: #10b981; margin-bottom: 1rem;">Trust Score: {trust_score:.1f}%</h4>
                <div class="trust-bar">
                    <div class="trust-fill" style="width: {trust_score}%"></div>
                </div>
                <small style="color: #6b7280;">Higher scores indicate greater trustworthiness</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Trust components breakdown
            st.markdown("**Trust Components:**")
            if pd.notna(applicant['behavioral_trust']):
                st.metric("Behavioral Trust", f"{applicant['behavioral_trust']:.1f}%")
            if pd.notna(applicant['social_trust']):
                st.metric("Social Trust", f"{applicant['social_trust']:.1f}%")
            if pd.notna(applicant['digital_trace']):
                st.metric("Digital Trust", f"{applicant['digital_trace']:.1f}%")
        
        with col2:
            obscurity_score = 100 - trust_score
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h4 style="color: #6b7280; margin-bottom: 1rem;">Obscurity Level: {obscurity_score:.1f}%</h4>
                <div class="obscurity-bar">
                    <div class="obscurity-fill" style="width: {obscurity_score}%"></div>
                </div>
                <small style="color: #6b7280;">Lower scores indicate better data visibility</small>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk scores breakdown
            st.markdown("**Risk Scores:**")
            if pd.notna(applicant['score_logistic']):
                st.metric("Logistic Model", f"{applicant['score_logistic']:.3f}")
            if pd.notna(applicant['score_xgb']):
                st.metric("XGBoost Model", f"{applicant['score_xgb']:.3f}")
            if pd.notna(applicant['average_score']):
                st.metric("Final Score", f"{applicant['average_score']:.3f}")
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("AI Insights", type="secondary", use_container_width=True):
            st.session_state.current_page = 'explainability'
            st.rerun()
    
    with col2:
        if pd.notna(applicant['total_trust_score']):
            if st.button("Generate Report", type="secondary", use_container_width=True):
                st.session_state.current_page = 'kfs'
                st.rerun()
    
    with col3:
        if st.button("Edit Data", type="secondary", use_container_width=True):
            st.info("Edit functionality coming soon...")
    
    with col4:
        if st.button("Back to Dashboard", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()
    
    # Detailed data view
    with st.expander("View Complete Assessment Data"):
        st.dataframe(applicant_data, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    st.set_page_config(
        page_title="Z-Cred Risk Assessment Platform",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load CSS
    load_css()
    
    # Initialize authentication and database
    init_auth()
    init_db()
    
    # Check authentication
    if not require_auth():
        auth_page()
        return
    
    # Render sidebar
    render_sidebar()
    
    # Get current page
    current_page = st.session_state.get('current_page', 'home')
    
    # Route to appropriate page
    if current_page == 'home':
        home_page()
    elif current_page == 'add_applicant':
        add_applicant_page()
    elif current_page == 'applicant_details':
        applicant_details_page()
    elif current_page == 'explainability':
        st.markdown('<h1 class="page-header">AI Insights & Explainability</h1>', unsafe_allow_html=True)
        st.info("This feature will provide SHAP explanations and AI insights for risk decisions.")
    elif current_page == 'kfs':
        st.markdown('<h1 class="page-header">Credit Report Generator</h1>', unsafe_allow_html=True)
        st.info("This feature will generate comprehensive credit reports and Key Fact Statements.")
    elif current_page == 'offline':
        st.markdown('<h1 class="page-header">Data Synchronization</h1>', unsafe_allow_html=True)
        st.info("This feature manages offline/online data synchronization and backup.")
    elif current_page == 'settings':
        st.markdown('<h1 class="page-header">Account Settings</h1>', unsafe_allow_html=True)
        st.info("User preferences, account settings, and system configuration.")

if __name__ == "__main__":
    main()
