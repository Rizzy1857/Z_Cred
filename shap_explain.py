import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import json
import os
from model_pipeline import load_models

def generate_shap_explanations(data_path='data/applicants.csv', save_plots=True, n_samples=100):
    """
    Generates SHAP explanations for model interpretability according to Phase 2 spec.
    
    Global: Feature importance (bar plot)
    Local: Applicant-specific breakdown (waterfall plot)
    Output stored in JSON for GUI integration
    """
    print("🔍 Generating SHAP Explanations...")
    
    # Load models
    lr_model, xgb_model, metrics = load_models()
    if lr_model is None:
        raise ValueError("Models not found. Train models first using model_pipeline.py")
    
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Prepare features
    feature_columns = [
        'avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
        'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months', 
        'recharge_frequency_per_month', 'stable_location_ratio'
    ]
    
    X = df[feature_columns]
    y = df['default']
    
    # Sample data for SHAP (computational efficiency)
    if len(X) > n_samples:
        sample_indices = np.random.choice(X.index, n_samples, replace=False)
        X_sample = X.loc[sample_indices]
        y_sample = y.loc[sample_indices]
        print(f"📊 Using {n_samples} samples for SHAP analysis")
    else:
        X_sample = X
        y_sample = y
        print(f"📊 Using all {len(X)} samples for SHAP analysis")
    
    # Create explanations directory
    explain_dir = 'explanations'
    if not os.path.exists(explain_dir):
        os.makedirs(explain_dir)
    
    # === GLOBAL EXPLANATIONS ===
    print("\n🌍 Generating Global SHAP Explanations...")
    
    global_explanations = {}
    
    for model_name, model in [('logistic_regression', lr_model), ('xgboost', xgb_model)]:
        print(f"   Analyzing {model_name}...")
        
        # Create explainer
        if model_name == 'xgboost':
            # For tree models, use TreeExplainer
            explainer = shap.TreeExplainer(model.named_steps['model'])
            # Get preprocessed data
            X_preprocessed = model.named_steps['imputer'].transform(X_sample)
            shap_values = explainer.shap_values(X_preprocessed)
        else:
            # For linear models, use LinearExplainer or Explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
        
        # Calculate mean absolute SHAP values for feature importance
        if len(shap_values.shape) > 2:  # Multi-class
            shap_values = shap_values[:, :, 1]  # Use positive class
        
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance data
        importance_data = {
            'features': feature_columns,
            'importance': mean_shap.tolist(),
            'importance_rank': np.argsort(mean_shap)[::-1].tolist()
        }
        
        global_explanations[model_name] = importance_data
        
        # Save global SHAP plot
        if save_plots:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_columns, 
                            plot_type='bar', show=False)
            plt.title(f'Global SHAP Feature Importance - {model_name.title()}')
            plt.tight_layout()
            plot_path = f'{explain_dir}/global_shap_{model_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   📊 Global plot saved: {plot_path}")
    
    # Save global explanations
    global_path = f'{explain_dir}/global_explanations.json'
    with open(global_path, 'w') as f:
        json.dump(global_explanations, f, indent=2)
    print(f"🌍 Global explanations saved: {global_path}")
    
    # === LOCAL EXPLANATIONS (SAMPLE APPLICANTS) ===
    print("\n👤 Generating Local SHAP Explanations...")
    
    # Select diverse sample applicants for local explanations
    local_samples = []
    
    # High risk applicant
    high_risk_idx = y_sample.idxmax() if y_sample.max() == 1 else y_sample.idxmin()
    local_samples.append(('high_risk', high_risk_idx))
    
    # Low risk applicant  
    low_risk_idx = y_sample.idxmin() if y_sample.min() == 0 else y_sample.idxmax()
    local_samples.append(('low_risk', low_risk_idx))
    
    # Medium risk applicant (if available)
    if len(X_sample) > 2:
        # Find middle probability applicant
        lr_probs = lr_model.predict_proba(X_sample)[:, 1]
        median_idx = X_sample.index[np.argsort(lr_probs)[len(lr_probs)//2]]
        local_samples.append(('medium_risk', median_idx))
    
    local_explanations = {}
    
    for risk_type, applicant_idx in local_samples:
        print(f"   Analyzing {risk_type} applicant (ID: {applicant_idx})...")
        
        applicant_data = X_sample.loc[applicant_idx:applicant_idx]
        actual_default = y_sample.loc[applicant_idx]
        
        local_explanations[risk_type] = {
            'applicant_id': int(applicant_idx),
            'actual_default': int(actual_default),
            'features': X_sample.loc[applicant_idx].to_dict(),
            'models': {}
        }
        
        for model_name, model in [('logistic_regression', lr_model), ('xgboost', xgb_model)]:
            # Get prediction
            pred_prob = model.predict_proba(applicant_data)[0][1]
            
            # Get SHAP values for this applicant
            if model_name == 'xgboost':
                explainer = shap.TreeExplainer(model.named_steps['model'])
                X_preprocessed = model.named_steps['imputer'].transform(applicant_data)
                shap_vals = explainer.shap_values(X_preprocessed)[0]
            else:
                explainer = shap.Explainer(model, X_sample)
                shap_vals = explainer(applicant_data)
                if hasattr(shap_vals, 'values'):
                    shap_vals = shap_vals.values[0]
                else:
                    shap_vals = shap_vals[0]
            
            # Store local explanation
            local_explanations[risk_type]['models'][model_name] = {
                'prediction_probability': float(pred_prob),
                'shap_values': shap_vals.tolist() if hasattr(shap_vals, 'tolist') else shap_vals,
                'shap_base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0
            }
            
            # Save waterfall plot
            if save_plots:
                plt.figure(figsize=(10, 6))
                
                # Create waterfall-style plot manually
                feature_impact = list(zip(feature_columns, shap_vals))
                feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
                
                features, impacts = zip(*feature_impact)
                colors = ['red' if x > 0 else 'blue' for x in impacts]
                
                plt.barh(range(len(features)), impacts, color=colors, alpha=0.7)
                plt.yticks(range(len(features)), features)
                plt.xlabel('SHAP Value (Impact on Prediction)')
                plt.title(f'Local SHAP Explanation - {risk_type.title()} Applicant\n'
                         f'{model_name.title()} Model (Prediction: {pred_prob:.2%})')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.tight_layout()
                
                plot_path = f'{explain_dir}/local_shap_{model_name}_{risk_type}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   📊 Local plot saved: {plot_path}")
    
    # Save local explanations
    local_path = f'{explain_dir}/local_explanations.json'
    with open(local_path, 'w') as f:
        json.dump(local_explanations, f, indent=2)
    print(f"👤 Local explanations saved: {local_path}")
    
    print("\n✅ SHAP explanations generated successfully!")
    
    return global_explanations, local_explanations

def explain_applicant(applicant_data, model_type='xgboost'):
    """
    Generate SHAP explanation for a specific applicant.
    
    Args:
        applicant_data: Dict with applicant features
        model_type: 'logistic_regression' or 'xgboost'
    
    Returns:
        Dict with SHAP values and explanation
    """
    # Load models
    lr_model, xgb_model, _ = load_models()
    if lr_model is None:
        raise ValueError("Models not found. Train models first.")
    
    model = xgb_model if model_type == 'xgboost' else lr_model
    
    # Prepare features
    feature_columns = [
        'avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
        'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months', 
        'recharge_frequency_per_month', 'stable_location_ratio'
    ]
    
    features_df = pd.DataFrame([{
        col: applicant_data.get(col, np.nan) for col in feature_columns
    }])
    
    # Get prediction
    pred_prob = model.predict_proba(features_df)[0][1]
    
    # Create explainer and get SHAP values
    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model.named_steps['model'])
        X_preprocessed = model.named_steps['imputer'].transform(features_df)
        shap_vals = explainer.shap_values(X_preprocessed)[0]
        base_value = explainer.expected_value
    else:
        # For linear model, need background data
        bg_data = pd.read_csv('data/applicants.csv')[feature_columns].sample(100)
        explainer = shap.Explainer(model, bg_data)
        shap_vals = explainer(features_df).values[0]
        base_value = explainer.expected_value
    
    # Create explanation
    explanation = {
        'prediction_probability': float(pred_prob),
        'model_type': model_type,
        'base_value': float(base_value),
        'features': {
            feature_columns[i]: {
                'value': applicant_data.get(feature_columns[i], np.nan),
                'shap_value': float(shap_vals[i]),
                'impact': 'positive' if shap_vals[i] > 0 else 'negative'
            }
            for i in range(len(feature_columns))
        },
        'top_factors': sorted(
            [(feature_columns[i], float(shap_vals[i])) for i in range(len(feature_columns))],
            key=lambda x: abs(x[1]), reverse=True
        )[:3]
    }
    
    return explanation

if __name__ == '__main__':
    # Check if models exist
    if not os.path.exists('models/lr_model.pkl'):
        print("❌ Models not found. Please run model_pipeline.py first.")
        exit(1)
    
    # Generate explanations
    try:
        global_exp, local_exp = generate_shap_explanations()
        print("\n🎉 SHAP analysis complete!")
        
        # Test single applicant explanation
        print("\n🧪 Testing single applicant explanation...")
        sample_applicant = {
            'avg_bill_delay_days': 15,
            'on_time_payment_ratio': 0.6,
            'prev_loans_taken': 3,
            'prev_loans_defaulted': 1,
            'community_endorsements': 2,
            'sim_card_tenure_months': 12,
            'recharge_frequency_per_month': 8,
            'stable_location_ratio': 0.7
        }
        
        explanation = explain_applicant(sample_applicant)
        print(f"Prediction: {explanation['prediction_probability']:.2%}")
        print("Top 3 factors:")
        for factor, impact in explanation['top_factors']:
            print(f"  {factor}: {impact:+.3f}")
        
    except Exception as e:
        print(f"❌ Error generating SHAP explanations: {e}")
