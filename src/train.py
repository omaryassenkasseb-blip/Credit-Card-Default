import joblib
import os
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_and_preprocess_data

def train_base_models():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    models = {
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
        "Random_Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }
    
    os.makedirs("models", exist_ok=True)
    
    for name, model in models.items():
        print(f"Training base model: {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"models/{name}_base.pkl")
        
    print("All base models trained and saved successfully in models/")

if __name__ == "__main__":
    train_base_models()