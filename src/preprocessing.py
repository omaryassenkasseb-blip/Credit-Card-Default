import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import os

def load_and_preprocess_data(file_path="data/raw/UCI_Credit_Card.csv", test_size=0.2, random_state=42):
    """
    Loads credit card dataset, cleans categorical anomalies,
    saves cleaned version to processed folder, splits into train/test, and applies scaling.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please place UCI_Credit_Card.csv in data/raw/ directory.")

    df = pd.read_csv(file_path)

    # Drop ID column if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Standardize target column name
    target_col = 'default.payment.next.month' if 'default.payment.next.month' in df.columns else 'default_payment_next_month'

    # Clean known anomalies in UCI Credit Card Dataset
    if 'EDUCATION' in df.columns:
        df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    if 'MARRIAGE' in df.columns:
        df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    # Save processed cleaned dataframe
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_data.csv", index=False)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standard scaling for numerical features
    num_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ]
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Convert back to DataFrames to retain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=num_features, index=X_test.index)

    # Save preprocessor for inference in Streamlit
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, num_features

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
    print("Data successfully loaded, processed, & saved to data/processed/cleaned_data.csv!")
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")