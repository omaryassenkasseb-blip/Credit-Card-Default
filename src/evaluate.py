import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import load_and_preprocess_data

def evaluate_best_model():
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()

    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Best model file not found! Please run src/tune.py first.")

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    os.makedirs("outputs", exist_ok=True)

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # 2. Feature Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=feat_df.head(10), x="Importance", y="Feature", hue="Feature", palette="magma", legend=False)
        plt.title("Top 10 Feature Importances")
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png")
        plt.close()

    print("Evaluation completed! Figures saved to outputs/confusion_matrix.png and feature_importance.png")

if __name__ == "__main__":
    evaluate_best_model()