import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import xgboost as xgb
import joblib
import os
import json

def train_and_save_models(data_path='data/applicants.csv', test_size=0.3, random_state=42):
    """
    Trains Logistic Regression and XGBoost models according to Phase 2 specification.
    
    Phase 1: Logistic Regression
    - Impute missing values (median)
    - Scale numeric features  
    - Train/test split (70/30)
    
    Phase 2: XGBoost
    - Same preprocessing (scaling optional for tree models)
    - Handles missing values internally
    - Better performance expected
    
    Returns both models and performance metrics.
    """
    print("🚀 Starting Phase 2 Model Training Pipeline...")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}. Run generate_data.py first.")
    
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} records from {data_path}")
    
    # Prepare features and labels
    feature_columns = [
        'avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
        'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months', 
        'recharge_frequency_per_month', 'stable_location_ratio'
    ]
    
    X = df[feature_columns]
    y = df['default']
    
    print(f"📋 Features: {feature_columns}")
    print(f"🎯 Target distribution: {y.value_counts(normalize=True).to_dict()}")
    print(f"❓ Missing values per feature:")
    for col in feature_columns:
        missing_pct = (X[col].isnull().sum() / len(X)) * 100
        print(f"   {col}: {missing_pct:.1f}%")
    
    # Train/test split (70/30 as per spec)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✂️  Split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Create models directory
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"📁 Created {model_dir} directory")
    
    # === PHASE 1: LOGISTIC REGRESSION ===
    print("\n🏗️  Phase 1: Training Logistic Regression...")
    
    # Pipeline with imputation and scaling
    pipeline_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler()),                   # Scale features
        ('model', LogisticRegression(
            solver='liblinear', 
            random_state=random_state,
            max_iter=1000
        ))
    ])
    
    # Train the model
    pipeline_lr.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    y_pred_lr = pipeline_lr.predict(X_test)
    y_pred_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
    
    metrics_lr = {
        'auc': roc_auc_score(y_test, y_pred_proba_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr)
    }
    
    print(f"📈 Logistic Regression Results:")
    print(f"   AUC: {metrics_lr['auc']:.4f}")
    print(f"   F1:  {metrics_lr['f1']:.4f}")
    print(f"   Precision: {metrics_lr['precision']:.4f}")
    print(f"   Recall: {metrics_lr['recall']:.4f}")
    
    # === PHASE 2: XGBOOST ===
    print("\n🏗️  Phase 2: Training XGBoost...")
    
    # XGBoost pipeline (handles missing values internally)
    pipeline_xgb = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Still good practice
        ('model', xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        ))
    ])
    
    # Train the model
    pipeline_xgb.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = pipeline_xgb.predict(X_test)
    y_pred_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
    
    metrics_xgb = {
        'auc': roc_auc_score(y_test, y_pred_proba_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb)
    }
    
    print(f"📈 XGBoost Results:")
    print(f"   AUC: {metrics_xgb['auc']:.4f}")
    print(f"   F1:  {metrics_xgb['f1']:.4f}")
    print(f"   Precision: {metrics_xgb['precision']:.4f}")
    print(f"   Recall: {metrics_xgb['recall']:.4f}")
    
    # Model comparison
    print(f"\n🏆 Model Comparison:")
    print(f"   AUC Improvement: {metrics_xgb['auc'] - metrics_lr['auc']:+.4f}")
    print(f"   F1 Improvement:  {metrics_xgb['f1'] - metrics_lr['f1']:+.4f}")
    
    # Save models
    lr_model_path = f'{model_dir}/lr_model.pkl'
    xgb_model_path = f'{model_dir}/xgb_model.pkl'
    
    joblib.dump(pipeline_lr, lr_model_path)
    joblib.dump(pipeline_xgb, xgb_model_path)
    
    print(f"💾 Models saved:")
    print(f"   Logistic Regression: {lr_model_path}")
    print(f"   XGBoost: {xgb_model_path}")
    
    # Save metrics
    all_metrics = {
        'logistic_regression': metrics_lr,
        'xgboost': metrics_xgb,
        'feature_columns': feature_columns,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'random_state': random_state
    }
    
    metrics_path = f'{model_dir}/model_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"📊 Metrics saved: {metrics_path}")
    
    # Feature importance for XGBoost
    feature_importance = pipeline_xgb.named_steps['model'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 XGBoost Feature Importance:")
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save feature importance
    importance_path = f'{model_dir}/feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"📋 Feature importance saved: {importance_path}")
    
    print("\n✅ Phase 2 Model Training Complete!")
    
    return pipeline_lr, pipeline_xgb, all_metrics

def load_models():
    """Load trained models from disk."""
    try:
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        
        # Load metrics if available
        metrics = {}
        if os.path.exists('models/model_metrics.json'):
            with open('models/model_metrics.json', 'r') as f:
                metrics = json.load(f)
        
        return lr_model, xgb_model, metrics
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        return None, None, {}

def predict_default_probability(applicant_data, lr_model=None, xgb_model=None):
    """
    Predict default probability for a single applicant using both models.
    
    Args:
        applicant_data: Dict with applicant features
        lr_model: Trained logistic regression model
        xgb_model: Trained XGBoost model
    
    Returns:
        Dict with predictions from both models
    """
    if lr_model is None or xgb_model is None:
        lr_model, xgb_model, _ = load_models()
        if lr_model is None:
            raise ValueError("Models not found. Train models first.")
    
    # Prepare features in correct order
    feature_order = [
        'avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
        'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months', 
        'recharge_frequency_per_month', 'stable_location_ratio'
    ]
    
    # Create feature vector
    features = pd.DataFrame([{
        col: applicant_data.get(col, np.nan) for col in feature_order
    }])
    
    # Get predictions
    lr_prob = lr_model.predict_proba(features)[0][1]
    xgb_prob = xgb_model.predict_proba(features)[0][1]
    
    return {
        'score_logistic': lr_prob,
        'score_xgb': xgb_prob,
        'average_score': (lr_prob + xgb_prob) / 2,
        'features_used': feature_order
    }

if __name__ == '__main__':
    # Check if data exists
    if not os.path.exists('data/applicants.csv'):
        print("❌ Training data not found. Please run generate_data.py first.")
        exit(1)
    
    # Train models
    lr_model, xgb_model, metrics = train_and_save_models()
    
    # Test prediction on sample data
    print("\n🧪 Testing prediction on sample applicant...")
    sample_applicant = {
        'avg_bill_delay_days': 5,
        'on_time_payment_ratio': 0.8,
        'prev_loans_taken': 2,
        'prev_loans_defaulted': 0,
        'community_endorsements': 3,
        'sim_card_tenure_months': 24,
        'recharge_frequency_per_month': 10,
        'stable_location_ratio': 0.85
    }
    
    prediction = predict_default_probability(sample_applicant, lr_model, xgb_model)
    print(f"Sample prediction: {prediction}")
    
    print("\n🎉 Model pipeline setup complete!")