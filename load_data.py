import os
import sys
import ctypes
import subprocess
import deeplake

# --- 1. Check & Request Admin Privileges (Windows) ---
def is_admin():
    """Check if the script is running with administrative privileges."""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def run_as_admin():
    """Re-run the program with administrative privileges, handling virtual environments."""
    script = os.path.abspath(sys.argv[0])
    params = " ".join([script] + sys.argv[1:])

    # Check if running inside a virtual environment
    if sys.prefix != sys.base_prefix:  # Running inside venv
        python_executable = os.path.join(sys.prefix, "Scripts", "python.exe")
    else:
        python_executable = sys.executable

    # Relaunch the script as admin
    subprocess.run(["powershell", "Start-Process", python_executable, "-ArgumentList", f'"{params}"', "-Verb", "RunAs"], shell=True)

if __name__ == "__main__":
    if not is_admin():
        print("Re-launching with administrative privileges...")
        run_as_admin()
        sys.exit()

    print("✅ Running with administrative privileges.")

    # --- 2. Define Local Dataset Path ---
    DATA_PATH = os.path.abspath("data/DAiSEE_Train")
    os.makedirs(DATA_PATH, exist_ok=True)

    # --- 3. Set Deep Lake Cache Directory Properly ---
    CACHE_PATH = os.path.abspath("data/deeplake_cache")
    os.environ["DEEPLAKE_CACHE_DIR"] = CACHE_PATH  # ✅ Correct way
    print(f"🔹 Deep Lake Cache Directory: {CACHE_PATH}")

    # --- 4. Download Dataset Locally to Allow Full Read/Write Permissions ---
    try:
        print("🔄 Downloading dataset...")
        ds = deeplake.deepcopy("hub://activeloop/daisee-train", dest=DATA_PATH)
        print(f"✅ Dataset successfully downloaded to: {DATA_PATH}")

    except AttributeError:
        print("⚠️ `deeplake.deepcopy()` doesn't exist in your version. Trying `deeplake.load()` with local storage...")

        try:
            ds = deeplake.load("hub://activeloop/daisee-train")
            ds.export(DATA_PATH)  # Export dataset locally
            print(f"✅ Dataset successfully exported to: {DATA_PATH}")
        except Exception as e:
            print(f"❌ Final error loading dataset: {e}")
            sys.exit(1)

    # --- 5. Verify & Fix Write Permissions (OS-Specific) ---
    try:
        if os.name == "nt":  # Windows
            os.system(f'icacls "{DATA_PATH}" /grant Everyone:F /T')
        else:  # Linux/macOS
            os.system(f"chmod -R 777 {DATA_PATH}")

        print(f"🔓 Write permissions fixed for: {DATA_PATH}")

    except Exception as e:
        print(f"❌ Error fixing permissions: {e}")

    print("🚀 ✅ Setup complete. Dataset is ready for use.")
