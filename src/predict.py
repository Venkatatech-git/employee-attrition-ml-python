import joblib
import pandas as pd
from pathlib import Path

from src.config import MODEL_PATH, DATA_PATH


def main():

    print("Loading pipeline...")
    model = joblib.load(MODEL_PATH)

    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["Attrition"])

    # use real row
    sample = X.iloc[[0]].copy()

    sample["Age"] = 35
    sample["MonthlyIncome"] = 5000
    sample["YearsAtCompany"] = 5
    sample["BusinessTravel"] = "Travel_Rarely"
    sample["Department"] = "Research & Development"

    print("Making prediction...")
    prediction = model.predict(sample)

    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
