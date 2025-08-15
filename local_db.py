import sqlite3
import pandas as pd
import json
import os
from datetime import datetime

# Database configuration
DB_FILE = 'local_applicants.db'

def init_db():
    """
    Initializes the SQLite database with the applicants table according to Phase 2 spec.
    Implements offline-first design for field agent operations.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # --- Start of migration logic ---
    # Check if 'applicants' table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='applicants'")
    table_exists = cursor.fetchone()

    if table_exists:
        # Table exists, check for columns and add them if they don't exist.
        cursor.execute("PRAGMA table_info(applicants)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'created_offline' not in columns:
            print("MIGRATING: Adding 'created_offline' column to 'applicants' table.")
            cursor.execute("ALTER TABLE applicants ADD COLUMN created_offline INTEGER DEFAULT 1")
            
        if 'synced' not in columns:
            print("MIGRATING: Adding 'synced' column to 'applicants' table.")
            cursor.execute("ALTER TABLE applicants ADD COLUMN synced INTEGER DEFAULT 0")
            
        if 'updated_timestamp' not in columns:
            print("MIGRATING: Adding 'updated_timestamp' column to 'applicants' table.")
            # Adding with a default value for existing rows
            cursor.execute("ALTER TABLE applicants ADD COLUMN updated_timestamp TEXT")
            cursor.execute("UPDATE applicants SET updated_timestamp = created_timestamp WHERE updated_timestamp IS NULL")

    # --- End of migration logic ---
    
    # Create applicants table with all required fields
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS applicants (
            applicant_id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_name TEXT NOT NULL,
            
            -- Core risk features (as per dataset schema)
            avg_bill_delay_days REAL,
            on_time_payment_ratio REAL,
            prev_loans_taken REAL,
            prev_loans_defaulted REAL,
            community_endorsements REAL,
            sim_card_tenure_months REAL,
            recharge_frequency_per_month REAL,
            stable_location_ratio REAL,
            
            -- Model predictions
            score_logistic REAL,
            score_xgb REAL,
            average_score REAL,
            
            -- Trust bar components
            behavioral_trust REAL,
            social_trust REAL,
            digital_trace REAL,
            total_trust_score REAL,
            
            -- Risk assessment
            risk_category TEXT, -- Low/Medium/High
            final_status TEXT,  -- Approved/Review/Rejected
            
            -- Compliance & consent
            consent_given INTEGER DEFAULT 0,
            consent_timestamp TEXT,
            data_processed INTEGER DEFAULT 0,
            
            -- Offline-first fields
            created_offline INTEGER DEFAULT 1,
            synced INTEGER DEFAULT 0,
            created_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            
            -- Additional metadata
            notes TEXT,
            agent_id TEXT,
            location TEXT
        )
    """)
    
    # Create sync log table for offline operations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sync_log (
            sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sync_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            records_synced INTEGER,
            sync_status TEXT, -- success/failed
            error_message TEXT
        )
    """)
    
    # Create consent log table for compliance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consent_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id INTEGER,
            consent_type TEXT, -- data_processing/scoring/kfs_generation
            consent_given INTEGER,
            consent_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            consent_details TEXT,
            FOREIGN KEY (applicant_id) REFERENCES applicants (applicant_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print("🗄️  Database initialized with offline-first design")

def insert_applicant(applicant_data, agent_id=None, location=None):
    """
    Inserts a new applicant record into the local database.
    Implements offline-first pattern for field operations.
    
    Args:
        applicant_data: Dict with applicant information
        agent_id: Field agent identifier
        location: Geographic location of assessment
    
    Returns:
        applicant_id: ID of inserted record
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Prepare data with defaults
    timestamp = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO applicants (
            applicant_name, avg_bill_delay_days, on_time_payment_ratio,
            prev_loans_taken, prev_loans_defaulted, community_endorsements,
            sim_card_tenure_months, recharge_frequency_per_month, stable_location_ratio,
            consent_given, consent_timestamp, created_offline, 
            agent_id, location, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        applicant_data.get('name', ''),
        applicant_data.get('avg_bill_delay_days'),
        applicant_data.get('on_time_payment_ratio'),
        applicant_data.get('prev_loans_taken'),
        applicant_data.get('prev_loans_defaulted'),
        applicant_data.get('community_endorsements'),
        applicant_data.get('sim_card_tenure_months'),
        applicant_data.get('recharge_frequency_per_month'),
        applicant_data.get('stable_location_ratio'),
        applicant_data.get('consent', 0),
        timestamp if applicant_data.get('consent', 0) else None,
        1,  # created_offline
        agent_id,
        location,
        applicant_data.get('notes', '')
    ))
    
    applicant_id = cursor.lastrowid
    
    # Log consent if given
    if applicant_data.get('consent', 0):
        log_consent(applicant_id, 'data_processing', True, 
                   'Initial consent for data processing and risk assessment')
    
    conn.commit()
    conn.close()
    
    print(f"👤 Applicant {applicant_data.get('name', 'Unknown')} added (ID: {applicant_id})")
    return applicant_id

def update_applicant_scores(applicant_id, scores_data):
    """
    Updates risk scores and trust components for an applicant.
    
    Args:
        applicant_id: Applicant ID
        scores_data: Dict with score information
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Determine risk category
    avg_score = scores_data.get('average_score', 
                               (scores_data.get('score_logistic', 0) + scores_data.get('score_xgb', 0)) / 2)
    
    if avg_score < 0.3:
        risk_category = 'Low'
        final_status = 'Approved'
    elif avg_score < 0.7:
        risk_category = 'Medium'
        final_status = 'Review Required'
    else:
        risk_category = 'High'
        final_status = 'Rejected'
    
    cursor.execute("""
        UPDATE applicants SET
            score_logistic = ?,
            score_xgb = ?,
            average_score = ?,
            behavioral_trust = ?,
            social_trust = ?,
            digital_trace = ?,
            total_trust_score = ?,
            risk_category = ?,
            final_status = ?,
            data_processed = 1,
            updated_timestamp = ?
        WHERE applicant_id = ?
    """, (
        scores_data.get('score_logistic'),
        scores_data.get('score_xgb'),
        avg_score,
        scores_data.get('behavioral_trust'),
        scores_data.get('social_trust'),
        scores_data.get('digital_trace'),
        scores_data.get('total_trust_score'),
        risk_category,
        final_status,
        datetime.now().isoformat(),
        applicant_id
    ))
    
    conn.commit()
    conn.close()
    
    print(f"📊 Scores updated for applicant {applicant_id}: {final_status} ({risk_category} risk)")

def get_applicants(unsynced_only=False, processed_only=False):
    """
    Retrieves applicants from the local database with filtering options.
    
    Args:
        unsynced_only: Return only records not yet synced
        processed_only: Return only records with risk scores
    
    Returns:
        DataFrame with applicant records
    """
    conn = sqlite3.connect(DB_FILE)
    
    query = "SELECT * FROM applicants"
    conditions = []
    
    if unsynced_only:
        conditions.append("synced = 0")
    if processed_only:
        conditions.append("data_processed = 1")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY created_timestamp DESC"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_single_applicant(applicant_id):
    """Retrieves a single applicant's data by ID."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT * FROM applicants WHERE applicant_id = ?", 
        conn, params=[applicant_id]
    )
    conn.close()
    return df

def log_consent(applicant_id, consent_type, consent_given, details=None):
    """
    Logs consent actions for compliance tracking.
    
    Args:
        applicant_id: Applicant ID
        consent_type: Type of consent (data_processing, scoring, kfs_generation)
        consent_given: Boolean consent status
        details: Additional consent details
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO consent_log (applicant_id, consent_type, consent_given, consent_details)
        VALUES (?, ?, ?, ?)
    """, (applicant_id, consent_type, int(consent_given), details))
    
    conn.commit()
    conn.close()

def sync_data_to_csv(filename='synced_data.csv', mark_synced=True):
    """
    Simulates syncing local data to central CSV file.
    Implements the offline-first sync pattern.
    
    Args:
        filename: Target CSV filename
        mark_synced: Whether to mark records as synced
    
    Returns:
        Number of records synced
    """
    # Get unsynced records
    df = get_applicants(unsynced_only=True)
    
    if len(df) == 0:
        print("📡 No unsynced records to sync")
        return 0
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Save to CSV
    output_path = f'data/{filename}'
    df.to_csv(output_path, index=False)
    
    # Mark records as synced if requested
    if mark_synced:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        applicant_ids = df['applicant_id'].tolist()
        placeholders = ','.join(['?' for _ in applicant_ids])
        
        cursor.execute(f"""
            UPDATE applicants 
            SET synced = 1, updated_timestamp = ?
            WHERE applicant_id IN ({placeholders})
        """, [datetime.now().isoformat()] + applicant_ids)
        
        # Log sync operation
        cursor.execute("""
            INSERT INTO sync_log (records_synced, sync_status)
            VALUES (?, ?)
        """, (len(df), 'success'))
        
        conn.commit()
        conn.close()
    
    print(f"📡 Synced {len(df)} records to {output_path}")
    return len(df)

def get_sync_status():
    """Returns current sync status and statistics."""
    conn = sqlite3.connect(DB_FILE)
    
    # Get record counts
    total_records = pd.read_sql_query("SELECT COUNT(*) as count FROM applicants", conn).iloc[0]['count']
    unsynced_records = pd.read_sql_query("SELECT COUNT(*) as count FROM applicants WHERE synced = 0", conn).iloc[0]['count']
    processed_records = pd.read_sql_query("SELECT COUNT(*) as count FROM applicants WHERE data_processed = 1", conn).iloc[0]['count']
    
    # Get last sync info
    last_sync = pd.read_sql_query("""
        SELECT sync_timestamp, records_synced, sync_status 
        FROM sync_log 
        ORDER BY sync_timestamp DESC 
        LIMIT 1
    """, conn)
    
    conn.close()
    
    status = {
        'total_records': total_records,
        'unsynced_records': unsynced_records,
        'processed_records': processed_records,
        'sync_percentage': ((total_records - unsynced_records) / total_records * 100) if total_records > 0 else 0,
        'last_sync': last_sync.to_dict('records')[0] if len(last_sync) > 0 else None
    }
    
    return status

def reset_database():
    """Resets the database by dropping all tables and recreating them."""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("🗑️  Database file removed")
    
    init_db()
    print("🔄 Database reset complete")

def export_database_backup(backup_filename=None):
    """
    Creates a full backup of the database in JSON format.
    
    Args:
        backup_filename: Optional custom filename
    
    Returns:
        Path to backup file
    """
    if backup_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"database_backup_{timestamp}.json"
    
    conn = sqlite3.connect(DB_FILE)
    
    # Export all tables
    backup_data = {}
    
    # Applicants table
    backup_data['applicants'] = pd.read_sql_query("SELECT * FROM applicants", conn).to_dict('records')
    
    # Sync log
    backup_data['sync_log'] = pd.read_sql_query("SELECT * FROM sync_log", conn).to_dict('records')
    
    # Consent log
    backup_data['consent_log'] = pd.read_sql_query("SELECT * FROM consent_log", conn).to_dict('records')
    
    # Metadata
    backup_data['metadata'] = {
        'backup_timestamp': datetime.now().isoformat(),
        'total_applicants': len(backup_data['applicants']),
        'database_file': DB_FILE
    }
    
    conn.close()
    
    # Save backup
    backup_path = f"data/{backup_filename}"
    with open(backup_path, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    print(f"💾 Database backup saved: {backup_path}")
    return backup_path

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Test functionality
    print("\n🧪 Testing database functionality...")
    
    # Test applicant insertion
    test_applicant = {
        'name': 'Test Applicant',
        'avg_bill_delay_days': 5,
        'on_time_payment_ratio': 0.8,
        'prev_loans_taken': 1,
        'prev_loans_defaulted': 0,
        'community_endorsements': 3,
        'sim_card_tenure_months': 24,
        'recharge_frequency_per_month': 12,
        'stable_location_ratio': 0.85,
        'consent': 1,
        'notes': 'Test record for database validation'
    }
    
    applicant_id = insert_applicant(test_applicant, agent_id='AGENT001', location='Test Location')
    
    # Test score updates
    test_scores = {
        'score_logistic': 0.25,
        'score_xgb': 0.22,
        'behavioral_trust': 75,
        'social_trust': 60,
        'digital_trace': 80,
        'total_trust_score': 71.67
    }
    
    update_applicant_scores(applicant_id, test_scores)
    
    # Test data retrieval
    all_applicants = get_applicants()
    print(f"📊 Total applicants: {len(all_applicants)}")
    
    # Test sync status
    status = get_sync_status()
    print(f"📡 Sync status: {status['unsynced_records']} unsynced of {status['total_records']} total")
    
    # Test backup
    backup_path = export_database_backup()
    
    print("\n✅ Database functionality test complete!")