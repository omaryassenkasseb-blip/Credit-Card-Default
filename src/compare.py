import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_and_preprocess_data

def compare_models():
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()

    models = {
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
        "Random_Forest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

    df_res = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)

    os.makedirs("outputs", exist_ok=True)
    df_res.to_csv("outputs/model_comparison.csv", index=False)

    # Plot Comparison Chart (Fixed Seaborn Warning)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_res, x="Model", y="ROC-AUC", hue="Model", palette="viridis", legend=False)
    plt.title("Model ROC-AUC Comparison")
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig("outputs/model_comparison_chart.png")
    plt.close()

    print("Comparison complete! Metrics saved to outputs/model_comparison.csv and plot saved to outputs/model_comparison_chart.png")

if __name__ == "__main__":
    compare_models()