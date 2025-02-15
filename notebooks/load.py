import ast
import datetime
import optuna
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import CategoricalDistribution, IntDistribution, FloatDistribution

# Step 1: Function to load backup file trials
def load_backup_file(file_path):
    trials = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Trial "):
                continue
            parts = line.split(" | ")
            if len(parts) != 3:
                continue
            try:
                trial_num = int(parts[0].split(" ")[1])
                val_loss_part = parts[1].split("Val Loss: ")[1]
                if val_loss_part == "None":
                    continue  # Skip trials without a valid loss
                val_loss = float(val_loss_part)
                params_str = parts[2].split("Params: ")[1]
                params = ast.literal_eval(params_str)
                trials.append((trial_num, val_loss, params))
            except Exception as e:
                print(f"Error parsing line: {line} => {e}")
    return trials

# Step 2: Define a key mapping for conversion
key_mapping = {
    'batch_sz': 'batch_size',
    'freeze_block': 'freeze_until_block',
    'hidden_ch': 'convlstm_hidden'
}

def convert_trial_params(old_params):
    new_params = {}
    for key, value in old_params.items():
        new_key = key_mapping.get(key, key)  # Remap key if needed
        # Convert to proper type:
        if new_key in ["seq_len", "batch_size", "freeze_until_block", "convlstm_hidden"]:
            try:
                new_params[new_key] = int(value)
            except Exception as e:
                print(f"[SKIP] Conversion error for {new_key}: {value} -> {e}")
                continue
        elif new_key == "lr":
            try:
                new_params[new_key] = float(value)
            except Exception as e:
                print(f"[SKIP] Conversion error for {new_key}: {value} -> {e}")
                continue
        else:
            new_params[new_key] = value
    return new_params

# Step 3: Define distributions matching your objective function
distributions = {
    "seq_len": CategoricalDistribution([15]),
    "batch_size": CategoricalDistribution([8, 16]),
    "freeze_until_block": IntDistribution(0, 4),
    "convlstm_hidden": CategoricalDistribution([64, 128, 256]),
    "lr": FloatDistribution(1e-5, 5e-4, log=True)
}

# Step 4: Convert backup trials and add them to the study
def add_converted_trials_to_db(study, backup_trials):
    for trial_number, val_loss, params in backup_trials:
        converted_params = convert_trial_params(params)
        # Create a FrozenTrial with the required fields:
        frozen_trial = FrozenTrial(
            number=trial_number,
            trial_id=str(trial_number),
            state=TrialState.COMPLETE,
            value=val_loss,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params=converted_params,
            distributions=distributions,
            user_attrs={},
            system_attrs={},
            intermediate_values={}
        )
        study.add_trial(frozen_trial)
    print(f"[INFO] Added {len(backup_trials)} trials to the study.")

if __name__ == "__main__":
    # Create or load a fresh study
    study_name = "my_effb0_convlstm_study"
    storage_url = "sqlite:///optuna_study.db"
    study = optuna.create_study(study_name=study_name, storage=storage_url, load_if_exists=True)
    
    # Load backup trials from file
    backup_trials = load_backup_file("optuna_trials_backup.txt")
    print("[BACKUP TRIALS] Loaded trials from file:")
    for trial_number, val_loss, params in backup_trials:
        print(f"Trial {trial_number} | Val Loss: {val_loss} | Params: {params}")
    
    # Add the converted trials into the study database
    add_converted_trials_to_db(study, backup_trials)
    
    # List the trials in the study after import
    print("\n[OPTUNA] Completed Trials after import:")
    for trial in study.trials:
        print(f"Trial {trial.number} | Val Loss: {trial.value} | Params: {trial.params}")
