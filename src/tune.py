import optuna
import joblib
import os
import numpy as np
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from preprocessing import load_and_preprocess_data

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial, X_train, y_train):
    classifier_name = trial.suggest_categorical("classifier", ["LightGBM", "XGBoost", "Random_Forest", "KNN"])

    if classifier_name == "LightGBM":
        n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("lgb_max_depth", 3, 12)
        learning_rate = trial.suggest_float("lgb_lr", 0.01, 0.2, log=True)
        num_leaves = trial.suggest_int("lgb_num_leaves", 20, 150)
        subsample = trial.suggest_float("lgb_subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0)

        model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )

    elif classifier_name == "XGBoost":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        learning_rate = trial.suggest_float("xgb_lr", 0.01, 0.2, log=True)
        subsample = trial.suggest_float("xgb_subsample", 0.6, 1.0)

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            scale_pos_weight=3.5,
            random_state=42,
            eval_metric="logloss"
        )

    elif classifier_name == "Random_Forest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 250)
        max_depth = trial.suggest_int("rf_max_depth", 3, 15)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42
        )

    else: # KNN
        n_neighbors = trial.suggest_int("knn_n_neighbors", 3, 20)
        weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    # 5-Fold Cross-Validation evaluating on ROC-AUC
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1).mean()
    return score

def tune_models():
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()

    study = optuna.create_study(direction="maximize")
    print("🚀 Running Optuna Optimization with Cross-Validation...")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

    print(f"\n✨ Best Trial ROC-AUC Score: {study.best_value:.4f}")
    print("📌 Best Parameters:", study.best_params)

    # Train and save the best model overall
    best_classifier = study.best_params["classifier"]
    
    if best_classifier == "LightGBM":
        best_model = LGBMClassifier(
            n_estimators=study.best_params["lgb_n_estimators"],
            max_depth=study.best_params["lgb_max_depth"],
            learning_rate=study.best_params["lgb_lr"],
            num_leaves=study.best_params["lgb_num_leaves"],
            subsample=study.best_params["lgb_subsample"],
            colsample_bytree=study.best_params["lgb_colsample_bytree"],
            is_unbalance=True,
            random_state=42,
            verbose=-1
        )
    elif best_classifier == "XGBoost":
        best_model = XGBClassifier(
            n_estimators=study.best_params["xgb_n_estimators"],
            max_depth=study.best_params["xgb_max_depth"],
            learning_rate=study.best_params["xgb_lr"],
            subsample=study.best_params["xgb_subsample"],
            scale_pos_weight=3.5,
            random_state=42,
            eval_metric="logloss"
        )
    elif best_classifier == "Random_Forest":
        best_model = RandomForestClassifier(
            n_estimators=study.best_params["rf_n_estimators"],
            max_depth=study.best_params["rf_max_depth"],
            class_weight="balanced",
            random_state=42
        )
    else:
        best_model = KNeighborsClassifier(
            n_neighbors=study.best_params["knn_n_neighbors"],
            weights=study.best_params["knn_weights"]
        )

    best_model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"🎯 Saved best model ({best_classifier}) to models/best_model.pkl")

if __name__ == "__main__":
    tune_models()