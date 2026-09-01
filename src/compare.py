import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocessing import load_and_preprocess_data

def compare_models():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    model_names = ["LightGBM", "XGBoost", "Random_Forest", "KNN"]
    results = []
    
    os.makedirs("outputs", exist_ok=True)
    
    for name in model_names:
        path = f"models/{name}_base.pkl"
        if not os.path.exists(path):
            continue
            
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
    df_res = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    df_res.to_csv("outputs/model_comparison.csv", index=False)
    
    # Plot Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x="Model", y="ROC-AUC", palette="viridis")
    plt.title("Model Comparison - Cross Validation / Test ROC-AUC Score")
    plt.ylim(0.5, 1.0)
    plt.ylabel("ROC-AUC Score")
    plt.savefig("outputs/model_comparison_chart.png", bbox_inches='tight')
    plt.close()
    
    print("Comparison complete! Metrics saved to outputs/model_comparison.csv and plot saved to outputs/model_comparison_chart.png")

if __name__ == "__main__":
    compare_models()