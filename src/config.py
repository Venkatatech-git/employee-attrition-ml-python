from pathlib import Path

# Project root (folder containing src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "raw.csv"
TARGET_COLUMN = "Attrition"

# Models
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "model.pkl"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Training
RANDOM_STATE = 42
TEST_SIZE = 0.2
