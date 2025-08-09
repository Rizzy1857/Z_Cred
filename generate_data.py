import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_applicants=1000, default_rate=0.25):
    """
    Generates a synthetic dataset for the Project Z-Score prototype.
    Correlations are baked in to simulate real-world data patterns.
    """
    np.random.seed(42)

    # Calculate number of defaulters and non-defaulters
    num_defaulters = int(num_applicants * default_rate)
    num_non_defaulters = num_applicants - num_defaulters

    # Generate data for non-defaulters (default=0) with positive financial behavior
    data_non_default = {
        'applicant_id': np.arange(1, num_non_defaulters + 1),
        'avg_bill_delay_days': np.random.randint(0, 8, num_non_defaulters),
        'on_time_payment_ratio': np.random.uniform(0.7, 1.0, num_non_defaulters),
        'prev_loans_taken': np.random.randint(0, 3, num_non_defaulters),
        'prev_loans_defaulted': np.random.randint(0, 1, num_non_defaulters),
        'community_endorsements': np.random.randint(3, 6, num_non_defaulters),
        'sim_card_tenure_months': np.random.randint(12, 61, num_non_defaulters),
        'recharge_frequency_per_month': np.random.randint(5, 21, num_non_defaulters),
        'stable_location_ratio': np.random.uniform(0.8, 1.0, num_non_defaulters),
        'default': np.zeros(num_non_defaulters, dtype=int)
    }
    df_non_default = pd.DataFrame(data_non_default)

    # Generate data for defaulters (default=1) with negative financial behavior
    data_default = {
        'applicant_id': np.arange(num_non_defaulters + 1, num_applicants + 1),
        'avg_bill_delay_days': np.random.randint(5, 31, num_defaulters),
        'on_time_payment_ratio': np.random.uniform(0.0, 0.7, num_defaulters),
        'prev_loans_taken': np.random.randint(1, 6, num_defaulters),
        'prev_loans_defaulted': np.random.randint(1, 6, num_defaulters),
        'community_endorsements': np.random.randint(0, 4, num_defaulters),
        'sim_card_tenure_months': np.random.randint(0, 24, num_defaulters),
        'recharge_frequency_per_month': np.random.randint(0, 10, num_defaulters),
        'stable_location_ratio': np.random.uniform(0.0, 0.8, num_defaulters),
        'default': np.ones(num_defaulters, dtype=int)
    }
    df_default = pd.DataFrame(data_default)

    # Combine and shuffle the data
    df = pd.concat([df_non_default, df_default], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['applicant_id'] = np.arange(1, num_applicants + 1)

    # Add noise and missing values to mimic real-world data
    for col in ['avg_bill_delay_days', 'on_time_payment_ratio', 'prev_loans_taken',
                'prev_loans_defaulted', 'community_endorsements', 'sim_card_tenure_months',
                'recharge_frequency_per_month', 'stable_location_ratio']:
        missing_indices = df.sample(frac=0.08, random_state=42).index
        df.loc[missing_indices, col] = np.nan

    # Introduce some anomalies for realism
    df.loc[df['on_time_payment_ratio'].isnull(), 'on_time_payment_ratio'] = df.loc[df['on_time_payment_ratio'].isnull(), 'on_time_payment_ratio'].apply(
        lambda x: np.random.uniform(0.8, 1.0) if np.random.rand() > 0.5 else np.nan
    )
    
    return df

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    
    applicants_df = generate_synthetic_data()
    applicants_df.to_csv('data/applicants.csv', index=False)
    print(f"Synthetic dataset of {len(applicants_df)} applicants created and saved to 'data/applicants.csv'")
    print("\nSample Data:")
    print(applicants_df.head())
    print("\nDefault Distribution:")
    print(applicants_df['default'].value_counts(normalize=True))