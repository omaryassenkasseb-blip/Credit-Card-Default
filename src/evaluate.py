import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import pandas as pd
from sklearn.metrics import confusion_matrix
from preprocessing import load_and_preprocess_data

def evaluate_best_model():
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    
    best_model_path = "models/best_model.pkl"
    if not os.path.exists(best_model_path):
        print("Best model not found. Please run src/tune.py first.")
        return
        
    model = joblib.load(best_model_path)
    y_pred = model.predict(X_test)
    
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("outputs/confusion_matrix.png", bbox_inches='tight')
    plt.close()
    
    # 2. Feature Importance (For Tree-based models)
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_df.head(10), x='Importance', y='Feature', palette='magma')
        plt.title("Top 10 Feature Importance - Best Model")
        plt.savefig("outputs/feature_importance.png", bbox_inches='tight')
        plt.close()
        
    print("Evaluation completed! Figures saved to outputs/ confusion_matrix.png and feature_importance.png")

if __name__ == "__main__":
    evaluate_best_model()