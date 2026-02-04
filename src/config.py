from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", BASE_DIR / "models/model.pkl")
)

DATA_PATH = Path(
    os.getenv("DATA_PATH", BASE_DIR / "data/raw.csv")
)
