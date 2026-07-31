
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "bank.csv"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

TARGET = "deposit"
POSITIVE_LABEL = "yes"

NUMERIC_FEATURES = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month", "poutcome",
]

LEAKY_FEATURE = "duration"

TEST_SIZE = 0.20
CV_FOLDS = 5
