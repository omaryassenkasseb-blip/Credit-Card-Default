import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n==========================================")
    print(f"🚀 Running: {script_path}")
    print(f"==========================================")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"❌ Error occurred while executing {script_path}")
        sys.exit(1)

def main():
    # 1. Data Preprocessing & Feature Engineering
    run_script("src/preprocessing.py")

    # 2. Baseline Model Training
    run_script("src/train.py")
    
    # 3. Compare Multiple Models
    run_script("src/compare.py")
    
    # 4. Hyperparameter Tuning with Optuna
    run_script("src/tune.py")
    
    # 5. Final Evaluation & Visualizations
    run_script("src/evaluate.py")
    
    print("\n==========================================")
    print("✅ Full Machine Learning Pipeline Completed!")
    print("==========================================")
    
    # 6. Launch Streamlit Dashboard
    print("\n🌐 Launching Streamlit Dashboard...")
    subprocess.run(["streamlit", "run", "app/streamlit_app.py"])

if __name__ == "__main__":
    main()