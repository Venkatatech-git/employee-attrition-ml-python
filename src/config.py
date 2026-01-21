from pathlib import Path

# Project root = folder that contains "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "raw.csv"   # <-- rename later if needed
TARGET_COLUMN = "Attrition"        # <-- update based on your dataset column name

# Models
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.pkl"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUTPUT_DIR = PROJECT_ROOT / "outputs"
