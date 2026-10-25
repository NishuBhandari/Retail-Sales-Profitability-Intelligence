"""
CONFIG & SHARED UTILITIES
Retail Sales & Profitability Intelligence
Shared paths, configuration constants, directory setup, and preprocessor helpers.
"""

import sys
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# Silence harmless warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Seed for reproducibility
RANDOM_STATE = 42

# ------------------------- PATH CONFIGURATION -------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"
CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_sales_data.csv"
SUMMARY_PATH = OUTPUT_DIR / "evaluation_summary.json"
DASHBOARD_PATH = OUTPUT_DIR / "dashboard.html"

# Ensure output directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def resolve_data_path():
    """
    Find raw dataset path across primary and fallback locations
    to guarantee flawless execution in VS Code, terminals, and subdirectories.
    """
    candidates = [
        BASE_DIR / "SalesDatafor preprocessing(maneet kaur).csv",
        Path.cwd() / "SalesDatafor preprocessing(maneet kaur).csv",
        BASE_DIR.parent / "SalesDatafor preprocessing(maneet kaur).csv",
        Path(r"C:\Users\Nishu\OneDrive\Documents\pb2\SalesDatafor preprocessing(maneet kaur).csv"),
        Path(r"C:\Users\Nishu\Downloads\SalesDatafor preprocessing(maneet kaur).csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return BASE_DIR / "SalesDatafor preprocessing(maneet kaur).csv"


DATA_PATH = resolve_data_path()


# ------------------------- SHARED HELPERS -------------------------
def save_plot(filename, dpi=180):
    """Save the current Matplotlib figure into figures directory with high DPI and close it."""
    plt.tight_layout()
    target_path = FIGURES_DIR / filename
    plt.savefig(target_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED FIGURE] {target_path.name}")


def make_one_hot_encoder():
    """Create OneHotEncoder compatible across scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features, categorical_features, polynomial=False):
    """
    Build a ColumnTransformer that imputes, scales, and encodes tabular features.
    If polynomial=True, applies degree-2 polynomial expansion to numeric features.
    """
    if polynomial:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ])
    else:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )
