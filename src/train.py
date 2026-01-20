from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from src.config import DATA_FILE, MODEL_DIR, MODEL_FILE, RANDOM_STATE
from src.preprocess import load_data, preprocess_data



def main():
    # 1) Load dataset
    df = load_data(str(DATA_FILE))

    # 2) Preprocess
    X, y = preprocess_data(df)

    # 3) Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Train model
    model = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000))
])

    model.fit(X_train, y_train)

    # 5) Quick evaluation (accuracy)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # 6) Save model + columns (columns are needed later for prediction)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
