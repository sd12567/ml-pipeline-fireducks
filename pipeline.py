import fireducks as fd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump
import mlflow

def load_data(filepath):
    # FireDucks-style data loading (mocked as we can't actually use firedrake here)
    print("Loading data using FireDucks (mock)...")
    df = pd.read_csv(filepath)  # Replace with fd.read_csv() in real scenario
    return df

def preprocess_data(df):
    # Example preprocessing
    print("Preprocessing data...")
    df.dropna(inplace=True)
    df = df[df['target'].notnull()]
    return df

def feature_engineering(df):
    print("Applying feature engineering...")
    # Dummy example: encoding categorical variables
    df = pd.get_dummies(df)
    return df

def train_model(X_train, y_train):
    print("Training model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def save_model(model, filename="models/model.joblib"):
    dump(model, filename)
    print(f"Model saved to {filename}")

def track_with_mlflow(model):
    print("Tracking with MLflow...")
    mlflow.set_experiment("FireDucks_ML_Pipeline")
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("model_type", "RandomForestClassifier")
        print("MLflow tracking complete.")

if __name__ == "__main__":
    df = load_data("data/sample.csv")
    df = preprocess_data(df)
    df = feature_engineering(df)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    track_with_mlflow(model)
