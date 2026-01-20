import pandas as pd
from src.config import TARGET_COLUMN


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df: pd.DataFrame):
    """
    Clean and prepare data for model training.
    Returns X (features) and y (target).
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    # Drop missing values
    df = df.dropna().copy()

    # Separate target
    y = df[TARGET_COLUMN]

    # Convert Yes/No to 1/0 if needed
    if y.dtype == object:
        y = y.map({"Yes": 1, "No": 0})

    X = df.drop(columns=[TARGET_COLUMN])

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True)

    return X, y
