import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_applicants=1000, default_rate=0.25):
    """
    Generates a synthetic dataset for the Project Z-Score prototype according to specification.
    Creates realistic correlations between features and default risk.
    
    Positive correlations → Lower default:
    - High on_time_payment_ratio
    - High community_endorsements  
    - High stable_location_ratio
    
    Negative correlations → Higher default:
    - High avg_bill_delay_days
    - High prev_loans_defaulted
    """
    np.random.seed(42)
    
    # Calculate number of defaulters and non-defaulters
    num_defaulters = int(num_applicants * default_rate)
    num_non_defaulters = num_applicants - num_defaulters
    
    print(f"Generating {num_applicants} records with {default_rate:.1%} default rate...")
    
    # Generate data for non-defaulters (default=0) with positive behavior patterns
    data_non_default = {
        'applicant_id': np.arange(1, num_non_defaulters + 1),
        'avg_bill_delay_days': np.random.randint(0, 8, num_non_defaulters),  # 0-30 days spec
        'on_time_payment_ratio': np.random.uniform(0.7, 1.0, num_non_defaulters),  # High for non-defaulters
        'prev_loans_taken': np.random.randint(0, 6, num_non_defaulters),  # 0-10 spec
        'prev_loans_defaulted': np.random.randint(0, 1, num_non_defaulters),  # Low for non-defaulters
        'community_endorsements': np.random.randint(2, 6, num_non_defaulters),  # 0-5 spec, higher for non-defaulters
        'sim_card_tenure_months': np.random.randint(12, 61, num_non_defaulters),  # 0-60 spec, stable for non-defaulters
        'recharge_frequency_per_month': np.random.randint(8, 21, num_non_defaulters),  # 0-20 spec, active users
        'stable_location_ratio': np.random.uniform(0.7, 1.0, num_non_defaulters),  # High stability
        'default': np.zeros(num_non_defaulters, dtype=int)
    }
    df_non_default = pd.DataFrame(data_non_default)
    
    # Generate data for defaulters (default=1) with negative behavior patterns
    data_default = {
        'applicant_id': np.arange(num_non_defaulters + 1, num_applicants + 1),
        'avg_bill_delay_days': np.random.randint(10, 31, num_defaulters),  # Higher delays
        'on_time_payment_ratio': np.random.uniform(0.0, 0.6, num_defaulters),  # Poor payment history
        'prev_loans_taken': np.random.randint(2, 11, num_defaulters),  # More loan history
        'prev_loans_defaulted': np.random.randint(1, 6, num_defaulters),  # Past defaults
        'community_endorsements': np.random.randint(0, 3, num_defaulters),  # Lower community trust
        'sim_card_tenure_months': np.random.randint(0, 30, num_defaulters),  # Less stable
        'recharge_frequency_per_month': np.random.randint(0, 12, num_defaulters),  # Less active
        'stable_location_ratio': np.random.uniform(0.0, 0.7, num_defaulters),  # Lower stability
        'default': np.ones(num_defaulters, dtype=int)
    }
    df_default = pd.DataFrame(data_default)
    
    # Combine and shuffle the data
    df = pd.concat([df_non_default, df_default], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['applicant_id'] = np.arange(1, num_applicants + 1)
    
    # Add 5-10% missing values for realism as per spec
    missing_rate = 0.08
    for col in ['avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
                'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months',
                'recharge_frequency_per_month', 'stable_location_ratio']:
        n_missing = int(len(df) * missing_rate)
        missing_indices = np.random.choice(df.index, n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # Add some noise to make data more realistic
    noise_factor = 0.1
    for col in ['on_time_payment_ratio', 'stable_location_ratio']:
        if col in df.columns:
            df[col] = df[col] + np.random.normal(0, noise_factor, len(df))
            df[col] = np.clip(df[col], 0, 1)  # Keep within bounds
    
    return df

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Generate synthetic dataset
    applicants_df = generate_synthetic_data(num_applicants=1000, default_rate=0.25)
    applicants_df.to_csv('data/applicants.csv', index=False)
    
    print(f"✅ Synthetic dataset created: {len(applicants_df)} records")
    print(f"📁 Saved to: data/applicants.csv")
    print(f"📊 Default rate: {applicants_df['default'].mean():.2%}")
    print(f"❓ Missing data rate: {applicants_df.isnull().sum().sum() / (len(applicants_df) * len(applicants_df.columns)):.2%}")
    
    print("\n📋 Sample Data:")
    print(applicants_df.head())
    
    print("\n📈 Feature Statistics:")
    print(applicants_df.describe())
    
    print("\n🎯 Default Distribution:")
    print(applicants_df['default'].value_counts(normalize=True))