import sqlite3
import pandas as pd

DB_FILE = 'local_applicants.db'

def init_db():
    """Initializes the SQLite database with the applicants table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applicants (
            applicant_id INTEGER PRIMARY KEY,
            applicant_name TEXT,
            avg_bill_delay_days REAL,
            on_time_payment_ratio REAL,
            prev_loans_taken REAL,
            prev_loans_defaulted REAL,
            community_endorsements REAL,
            sim_card_tenure_months REAL,
            recharge_frequency_per_month REAL,
            stable_location_ratio REAL,
            default_status INTEGER,
            score_logistic REAL,
            score_xgb REAL,
            trust_score REAL,
            behavioral_trust REAL,
            social_trust REAL,
            digital_trace REAL,
            status TEXT,
            consent_given INTEGER
        )
    """)
    conn.commit()
    conn.close()

def insert_applicant(applicant_data):
    """Inserts a new applicant record into the local database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO applicants (
            applicant_name, avg_bill_delay_days, on_time_payment_ratio,
            prev_loans_taken, prev_loans_defaulted, community_endorsements,
            sim_card_tenure_months, recharge_frequency_per_month,
            stable_location_ratio, status, consent_given
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        applicant_data['name'], applicant_data['avg_bill_delay_days'],
        applicant_data['on_time_payment_ratio'], applicant_data['prev_loans_taken'],
        applicant_data['prev_loans_defaulted'], applicant_data['community_endorsements'],
        applicant_data['sim_card_tenure_months'], applicant_data['recharge_frequency_per_month'],
        applicant_data['stable_location_ratio'], 'Obscure', applicant_data['consent']
    ))
    conn.commit()
    conn.close()

def get_applicants():
    """Retrieves all applicants from the local database."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM applicants", conn)
    conn.close()
    return df

def update_applicant_scores(applicant_id, scores):
    """Updates the scores for a given applicant."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE applicants SET
            score_logistic = ?,
            score_xgb = ?,
            trust_score = ?,
            behavioral_trust = ?,
            social_trust = ?,
            digital_trace = ?,
            status = ?
        WHERE applicant_id = ?
    """, (
        scores['score_logistic'], scores['score_xgb'], scores['total_trust_score'],
        scores['behavioral_trust'], scores['social_trust'], scores['digital_trace'],
        scores['status'], applicant_id
    ))
    conn.commit()
    conn.close()

def get_single_applicant(applicant_id):
    """Retrieves a single applicant's data."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT * FROM applicants WHERE applicant_id = {applicant_id}", conn)
    conn.close()
    return df

def sync_data_to_csv(df, filename='synced_data.csv'):
    """Simulates syncing local data to a central CSV."""
    df.to_csv(f'data/{filename}', index=False)
    print(f"Data synced successfully to data/{filename}")