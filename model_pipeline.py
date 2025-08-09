import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import joblib
import os

def train_and_save_models(data_path='data/applicants.csv'):
    """
    Trains a Logistic Regression and an XGBoost model,
    saves the models and the scaler to disk.
    """
    print("Starting model training pipeline...")
    # Load data
    df = pd.read_csv(data_path)
    features = df.drop(['applicant_id', 'default'], axis=1)
    labels = df['default']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # --- Phase 1: Logistic Regression ---
    print("Training Logistic Regression model...")
    pipeline_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear', random_state=42))
    ])
    pipeline_lr.fit(X_train, y_train)
    y_pred_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
    print(f"Logistic Regression AUC: {auc_lr:.4f}")

    # --- Phase 2: XGBoost ---
    print("Training XGBoost model...")
    # XGBoost handles missing values internally, so we don't need the imputer in the pipeline
    pipeline_xgb = Pipeline([
        ('scaler', StandardScaler()),  # Scaling is still good practice
        ('model', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    pipeline_xgb.fit(X_train, y_train)
    y_pred_proba_xgb = pipeline_xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    print(f"XGBoost AUC: {auc_xgb:.4f}")

    # Save models and scaler
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    joblib.dump(pipeline_lr, f'{model_dir}/lr_model.pkl')
    joblib.dump(pipeline_xgb, f'{model_dir}/xgb_model.pkl')
    print("Models and scaler saved successfully.")

    return pipeline_lr, pipeline_xgb

if __name__ == '__main__':
    # Ensure data exists before running
    if not os.path.exists('data/applicants.csv'):
        print("Data file not found. Please run generate_data.py first.")
    else:
        train_and_save_models()