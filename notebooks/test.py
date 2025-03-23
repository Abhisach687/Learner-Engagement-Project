import sqlite3
import os

db_path = r"C:/Users/abhis/Downloads/Documents/Learner Engagement Project/notebooks/tuning_effv2l_bilstm_flow_ema.db"

def debug_database():
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_path}:")
        print(tables)
        
        # Check study table
        if ('studies',) in tables:
            cursor.execute("SELECT * FROM studies;")
            print("\nStudies:", cursor.fetchall())
            
        if ('trials',) in tables:
            cursor.execute("SELECT trial_id, study_id, state FROM trials;")
            print("\nTrials:", cursor.fetchall())
            
        # Replace your current debug code with:
        cursor.execute("""
            SELECT t.trial_id, p.param_name, p.param_value, v.value 
            FROM trials t
            JOIN trial_params p ON t.trial_id = p.trial_id
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = 1;
        """)
        results = cursor.fetchall()
        for row in results:
            print(f"Trial {row[0]}: {row[1]}={row[2]}, loss={row[3]}")
            
    except sqlite3.OperationalError as e:
        print(f"Database locked or corrupted: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    debug_database()