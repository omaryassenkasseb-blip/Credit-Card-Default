import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_and_preprocess_data

def train_base_models():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    # Define baseline models with clean keys (no spaces)
    models = {
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    # Ensure output directory exists
    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        print(f"Training base model: {name}...")
        model.fit(X_train, y_train)
        
        # Safe filename formatting
        safe_name = name.replace(" ", "_")
        file_path = f"models/{safe_name}_base.pkl"
        
        joblib.dump(model, file_path)
        print(f"Saved: {file_path}")

if __name__ == "__main__":
    train_base_models()