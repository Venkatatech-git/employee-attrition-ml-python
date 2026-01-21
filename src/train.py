from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.config import DATA_FILE, MODEL_DIR, MODEL_FILE, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from src.preprocess import load_data, preprocess_data


def main():
    # Ensure folders exist
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1) Load data
    df = load_data(str(DATA_FILE))

    # 2) Preprocess
    X, y = preprocess_data(df)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 4) Train
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 5) Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    # 6) Save model + metadata
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")

    # 7) Save metrics
    metrics_file = OUTPUT_DIR / "metrics.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
    print(f"Saved metrics to: {metrics_file}")


if __name__ == "__main__":
    main()
