import sqlite3
import os
from pathlib import Path

# Path to the database
BASE_DIR = Path("C:/Users/abhis/Downloads/Documents/Learner Engagement Project")
db_path = BASE_DIR / "notebooks" / "tuning_eff_v2l_refined_crossattn_cbam.db"

# Verify database file exists
if not os.path.exists(db_path):
    print(f"Database file not found at {db_path}")
    exit(1)

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# First let's check the schema to understand the database
try:
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\n===== Database Schema Analysis for {db_path.name} =====")
    print("Tables found:", ', '.join([t[0] for t in tables]))
    
    for table in tables:
        table_name = table[0]
        print(f"\n-- Schema for {table_name} table --")
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
    
    # For trials table, let's check what columns are used for trial tracking
    if any(t[0] == 'trials' for t in tables):
        print("\n-- Analyzing trials table --")
        cursor.execute("SELECT * FROM trials LIMIT 1")
        cols = [description[0] for description in cursor.description]
        print(f"Columns: {', '.join(cols)}")
        
        # Count total trials
        cursor.execute("SELECT COUNT(*) FROM trials")
        total_count = cursor.fetchone()[0]
        print(f"Total trials: {total_count}")
        
        # Try to identify state/status column
        state_column = None
        for possible_col in ['state', 'status', 'trial_state']:
            if possible_col in cols:
                state_column = possible_col
                break
        
        if state_column:
            cursor.execute(f"SELECT {state_column}, COUNT(*) FROM trials GROUP BY {state_column}")
            state_counts = cursor.fetchall()
            print(f"\nBreakdown by {state_column}:")
            for state, count in state_counts:
                print(f"- {state}: {count}")
        
        # Try to identify value/objective column
        value_column = None
        for possible_col in ['value', 'objective_value', 'loss', 'score']:
            if possible_col in cols:
                value_column = possible_col
                break
        
        if value_column and state_column:
            cursor.execute(f"""
            SELECT trial_id, {value_column}, datetime_start, datetime_complete 
            FROM trials 
            WHERE {state_column} = 'COMPLETE' AND {value_column} IS NOT NULL
            ORDER BY {value_column} ASC
            LIMIT 5
            """)
            best_trials = cursor.fetchall()
            
            print(f"\nTop 5 best trials (sorted by {value_column}):")
            for trial in best_trials:
                print(f"Trial #{trial[0]}: {value_column} = {trial[1]}, Started: {trial[2]}, Completed: {trial[3]}")
                
            # Get params for best trials if available
            params_table = None
            for possible_table in ['trial_params', 'params', 'parameters']:
                try:
                    cursor.execute(f"SELECT * FROM {possible_table} LIMIT 1")
                    params_table = possible_table
                    break
                except sqlite3.OperationalError:
                    continue
            
            if params_table:
                print("\nParameters for best trials:")
                for trial in best_trials:
                    cursor.execute(f"SELECT * FROM {params_table} WHERE trial_id = ?", (trial[0],))
                    params = cursor.fetchall()
                    print(f"Trial #{trial[0]} params:")
                    for param in params:
                        print(f"  {param}")
    
except sqlite3.OperationalError as e:
    print(f"Error analyzing database: {e}")
    print("Let's try an alternative approach...")
    
    # Let's at least see what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables found:", ', '.join([t[0] for t in tables]))
    
    # For each table, show a sample row
    for table in tables:
        try:
            table_name = table[0]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cursor.fetchone()
            cols = [description[0] for description in cursor.description]
            print(f"\nSample from {table_name} table:")
            print(f"Columns: {cols}")
            print(f"Sample row: {sample}")
            
            # Count rows
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Total rows: {count}")
        except sqlite3.OperationalError as e:
            print(f"Error accessing {table_name}: {e}")

# Close connection
conn.close()