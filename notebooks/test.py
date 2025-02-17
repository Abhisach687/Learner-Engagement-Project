import optuna

# Load the study
study = optuna.load_study(study_name="mobilev2_tcn_study", storage="sqlite:///tuning.db")

# Print all current trials
print("Current trials:", study.trials)

# Get the latest trial
latest_trial = study.trials[-1]

# Print the parameters in a tidy way
print("Parameters:")
for param_name, param_value in latest_trial.params.items():
    print(f"  {param_name}: {param_value}")