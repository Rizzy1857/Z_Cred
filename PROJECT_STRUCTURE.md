# Z-Cred Risk Assessment Platform

## Clean Project Structure

```
Z_Cred/
├── app.py                    # Main Streamlit application (professional, no emojis)
├── auth.py                   # Authentication system (login/register/sessions)
├── local_db.py              # Database operations (SQLite with user-specific data)
├── model_pipeline.py        # Machine learning models (Logistic Regression + XGBoost)
├── generate_data.py         # Synthetic data generation utilities
├── kfs_generator.py         # Credit report/KFS document generation
├── shap_explain.py          # AI explainability using SHAP
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License file
├── .gitignore             # Git ignore rules
├── auth.db                # Authentication database (SQLite)
├── local_applicants.db    # Applicant data database (SQLite)
├── data/                  # Data files
│   └── applicants.csv     # CSV data for training
├── models/                # Trained ML models
│   ├── lr_model.pkl       # Logistic Regression model
│   ├── xgb_model.pkl      # XGBoost model
│   ├── model_metrics.json # Model performance metrics
│   └── feature_importance.csv # Feature importance data
├── generated_kfs/         # Generated reports (PDF output)
└── backups/              # Database backups
```

## Core Features

### 1. Professional Authentication System
- Secure user registration and login
- Session management with expiration
- Password hashing with salt
- User-specific data isolation

### 2. Risk Assessment Platform
- **Dashboard**: Portfolio overview with key metrics
- **New Applicant**: Comprehensive assessment form
- **Risk Analysis**: Detailed risk and trust analysis
- **AI Insights**: SHAP explainability features
- **Credit Report**: PDF report generation
- **Data Sync**: Offline/online synchronization
- **Settings**: Account and system configuration

### 3. Trust & Risk Scoring
- **Behavioral Trust**: Payment history and bill delays
- **Social Trust**: Community endorsements and location stability
- **Digital Trust**: SIM tenure and usage patterns
- **Risk Categories**: Low/Medium/High classification
- **Obscurity Level**: Data visibility assessment

### 4. Machine Learning Models
- Logistic Regression for probability estimation
- XGBoost for ensemble predictions
- SHAP for model explainability
- Feature importance analysis

### 5. Professional Design
- Clean, modern interface without emojis
- Responsive design for all devices
- Professional color scheme and typography
- Smooth animations and transitions
- Card-based layout with shadows

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `streamlit run app.py`
3. Register an account or login
4. Start adding applicants for risk assessment

## Database Design

### Users Table (auth.db)
- User registration and authentication
- Session management
- Account settings

### Applicants Table (local_applicants.db)
- User-specific applicant data
- Risk scores and trust components
- Consent and compliance tracking
- Offline-first design with sync capabilities

## Security Features

- Password hashing with PBKDF2
- Session-based authentication
- User data isolation
- Consent management
- Audit trails for compliance

---

**Note**: This is a professional credit risk assessment platform designed for financial institutions and credit officers. All features are built with compliance, security, and user experience in mind.
