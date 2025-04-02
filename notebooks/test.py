import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

db_path = r"C:/Users/abhis/Downloads/Documents/Learner Engagement Project/notebooks/mobilenettcn_balance_tuning.db"

def analyze_optuna_study():
    if not os.path.exists(db_path):
        print(f"Database file {db_path} does not exist.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Basic database info
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_path}:")
        print(tables)
        
        # Get study info
        cursor.execute("SELECT * FROM studies;")
        studies = cursor.fetchall()
        print(f"\nStudies: {studies}")
        
        # Get study direction (minimize or maximize)
        cursor.execute("SELECT direction FROM study_directions WHERE study_id = 1;")
        direction = cursor.fetchall()
        direction = direction[0][0] if direction else "MINIMIZE"
        print(f"Study Direction: {direction}")
        
        # Get trial statuses
        cursor.execute("SELECT trial_id, study_id, state FROM trials;")
        trials = cursor.fetchall()
        print(f"\nTotal Trials: {len(trials)}")
        
        # Count trials by state
        states = [trial[2] for trial in trials]
        state_counts = Counter(states)
        print("\nTrial Status Counts:")
        for state, count in state_counts.items():
            print(f"{state}: {count}")
        
        # Get parameter names and their distributions from trial_params
        cursor.execute("""
            SELECT DISTINCT param_name FROM trial_params;
        """)
        param_names = [row[0] for row in cursor.fetchall()]
        print(f"\nParameters: {param_names}")
        
        # Create a DataFrame of all trials with parameters and values
        cursor.execute("""
            SELECT t.trial_id, t.state, v.value as objective_value
            FROM trials t
            LEFT JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE t.study_id = 1
            ORDER BY t.trial_id;
        """)
        trial_data = cursor.fetchall()
        df_trials = pd.DataFrame(trial_data, columns=['trial_id', 'state', 'objective_value'])
        
        # Get all parameters for each trial
        cursor.execute("""
            SELECT trial_id, param_name, param_value 
            FROM trial_params
            ORDER BY trial_id;
        """)
        param_data = cursor.fetchall()
        
        # Transform parameters into a wide format
        param_df = pd.DataFrame(param_data, columns=['trial_id', 'param_name', 'param_value'])
        param_wide = param_df.pivot(index='trial_id', columns='param_name', values='param_value').reset_index()
        
        # Merge trials with parameters
        full_df = pd.merge(df_trials, param_wide, on='trial_id')
        
        # Map categorical parameters if needed
        # For batch_size: 0 -> 32, 1 -> 64 (example mapping)
        if 'batch_size' in full_df.columns:
            print("\nParameter Mappings:")
            print("  batch_size: 0.0 -> 32, 1.0 -> 64")
            print("  hidden_dim: 0.0 -> 64, 1.0 -> 128, 2.0 -> 256")
        
        # Display parameter ranges
        print("\nParameter Distributions:")
        for param in param_names:
            values = full_df[param].dropna().values
            if len(values) > 0:
                print(f"  {param}: min={np.min(values):.6f}, max={np.max(values):.6f}, unique values={len(np.unique(values))}")
        
        # Show completed trials with valid objective values
        completed_trials = full_df[(full_df['state'] == 'COMPLETE') & (~full_df['objective_value'].isna())].copy()
        if len(completed_trials) > 0:
            # For minimization problem, sort in ascending order
            # For maximization problem, sort in descending order
            ascending = direction == "MINIMIZE"
            completed_trials.sort_values('objective_value', ascending=ascending, inplace=True)
            
            best_trial = completed_trials.iloc[0]
            print(f"\nBest Trial: #{best_trial['trial_id']} with objective value={best_trial['objective_value']}")
            print("Parameters:")
            for param in param_names:
                print(f"  {param}: {best_trial[param]}")
            
            # Statistics of completed trials
            print("\nCompleted Trials Statistics:")
            print(completed_trials['objective_value'].describe())
            
            # Visualize parameter relationships with objective value
            print("\nParameter Correlations with Objective Value:")
            correlations = {}
            for param in param_names:
                # Check if parameter has variation
                if len(completed_trials[param].unique()) > 1:
                    correlation = completed_trials['objective_value'].corr(completed_trials[param])
                    correlations[param] = correlation
                    print(f"  {param}: {correlation:.4f}")
                else:
                    print(f"  {param}: No variation (all values are {completed_trials[param].iloc[0]})")
            
            # Create visualization of parameter importance based on correlation
            plt.figure(figsize=(10, 5))
            plt.bar(correlations.keys(), correlations.values())
            plt.xlabel('Parameter')
            plt.ylabel('Correlation with Objective Value')
            plt.title('Parameter Importance (Correlation with Objective Value)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot to file
            plot_path = os.path.join(os.path.dirname(db_path), 'parameter_importance.png')
            plt.savefig(plot_path)
            print(f"\nParameter importance plot saved to: {plot_path}")
            
            # Parallel coordinates plot with only numeric columns
            try:
                from pandas.plotting import parallel_coordinates
                plt.figure(figsize=(12, 6))
                
                # Create a copy with only numeric columns for parallel plot
                plot_df = completed_trials[['trial_id'] + param_names].copy()
                
                # Add a performance category column for coloring
                perf_bins = pd.qcut(completed_trials['objective_value'], 
                                    min(3, len(completed_trials)),
                                    labels=['best', 'medium', 'worst'] if direction == "MINIMIZE" else ['worst', 'medium', 'best'])
                plot_df['performance'] = perf_bins
                
                parallel_coordinates(plot_df, 'performance', colormap='viridis')
                plt.title('Parallel Coordinates Plot of Parameters')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plot_path = os.path.join(os.path.dirname(db_path), 'parallel_coordinates.png')
                plt.savefig(plot_path)
                print(f"\nParallel coordinates plot saved to: {plot_path}")
            except Exception as e:
                print(f"\nCould not create parallel coordinates plot: {e}")
        
        # Check for failed trials (marked COMPLETE but no objective value)
        failed_trials = full_df[(full_df['state'] == 'COMPLETE') & (full_df['objective_value'].isna())]
        if not failed_trials.empty:
            print(f"\nFailed Trials (COMPLETE but no objective value): {len(failed_trials)}")
            print("These may indicate trials that crashed or had errors:")
            for _, row in failed_trials.iterrows():
                print(f"  Trial #{row['trial_id']} - Parameters:")
                for param in param_names:
                    print(f"    {param}: {row[param]}")
                print("")

        # Show running trials
        running_trials = full_df[full_df['state'] == 'RUNNING']
        if not running_trials.empty:
            print(f"\nRunning Trials: {len(running_trials)}")
            print("Trial IDs:", running_trials['trial_id'].tolist())
            
        # Suggest next steps
        print("\nRecommendations:")
        if len(completed_trials) < 20:
            print("- Continue running more trials to explore the parameter space better")
        
        if 'num_frames' in param_names and len(full_df['num_frames'].unique()) == 1:
            print("- 'num_frames' parameter has no variation, consider expanding its search space")
        
        if len(failed_trials) > 0:
            print("- Investigate failed trials to understand why they didn't complete successfully")
            
    except Exception as e:
        print(f"Error analyzing database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    analyze_optuna_study()