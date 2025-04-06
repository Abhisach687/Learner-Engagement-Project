import sqlite3
import os
import glob

def get_best_hyperparameters(db_file):
    """Extract best hyperparameters from a single database file."""
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Check if the required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not all(table in tables for table in ['trials', 'trial_params', 'trial_values']):
            print(f"  Warning: Required tables missing in {db_file}")
            conn.close()
            return None
            
        # Find the best trial (with lowest loss)
        cursor.execute("""
            SELECT t.trial_id, v.value 
            FROM trials t
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.state = 'COMPLETE'
            ORDER BY v.value ASC
            LIMIT 1;
        """)
        best_trial = cursor.fetchone()
        
        if not best_trial:
            print(f"  No completed trials found in {db_file}")
            conn.close()
            return None
            
        best_trial_id, best_loss = best_trial
        
        # Get hyperparameters for the best trial
        cursor.execute("""
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?;
        """, (best_trial_id,))
        
        hyperparams = {row[0]: row[1] for row in cursor.fetchall()}
        hyperparams['loss'] = best_loss
        hyperparams['trial_id'] = best_trial_id
        
        conn.close()
        return hyperparams
        
    except sqlite3.Error as e:
        print(f"  Error with database {os.path.basename(db_file)}: {e}")
        if 'conn' in locals():
            conn.close()
        return None

def analyze_all_databases(directory=None):
    """Find all .db files and extract best hyperparameters from each."""
    if directory is None:
        directory = os.path.dirname(os.path.dirname(
            r"C:/Users/abhis/Downloads/Documents/Learner Engagement Project/notebooks/"))
    
    # Find all .db files in the directory and its subdirectories
    db_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.db'):
                db_files.append(os.path.join(root, file))
    
    if not db_files:
        print(f"No .db files found in {directory}")
        return
    
    print(f"Found {len(db_files)} database files.")
    
    # Process each database file
    results = {}
    for db_file in db_files:
        filename = os.path.basename(db_file)
        print(f"\nAnalyzing {filename}...")
        
        best_params = get_best_hyperparameters(db_file)
        if best_params:
            results[filename] = best_params
    
    # Print summary of best hyperparameters
    print("\n===== BEST HYPERPARAMETERS FOR EACH MODEL =====")
    for filename, params in results.items():
        print(f"\nFile: {filename}")
        print(f"Best Trial ID: {params.pop('trial_id')}")
        print(f"Loss: {params.pop('loss')}")
        print("Hyperparameters:")
        for name, value in params.items():
            print(f"  - {name}: {value}")
    
    return results

if __name__ == "__main__":
    analyze_all_databases()