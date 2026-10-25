"""
PORTION 3: REGRESSION MODELING (PREDICT CONTINUOUS SALES)
Retail Sales & Profitability Intelligence

Tasks:
- Prepare feature matrix and target ('Sales')
- Split into train (80%) and test (20%) sets
- Train and benchmark:
    1. Linear Regression (Baseline)
    2. Decision Tree Regressor
    3. Random Forest Regressor
    4. Polynomial Regression (Degree 2)
- Compute evaluation metrics (MAE, MSE, RMSE, R²)
- Plot actual vs. predicted values and residual errors
- Serialize trained regression models to outputs/models/
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    CLEANED_DATA_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    make_preprocessor,
    save_plot,
)


def compute_metrics(y_true, y_pred):
    """Calculate MAE, MSE, RMSE, and R2 score."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "mse": round(mse, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def run_regression(df=None):
    """
    Train and evaluate multiple regression pipelines predicting 'Sales'.
    
    Args:
        df (pd.DataFrame, optional): Preprocessed DataFrame. If None, loads from disk.
        
    Returns:
        tuple: (results_dict, best_model_name, fitted_models, feature_names)
    """
    print("\n" + "=" * 65)
    print("PORTION 3: REGRESSION MODELING (PREDICT SALES)")
    print("=" * 65)

    if df is None:
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}. Run Portion 1 first.")
        df = pd.read_csv(CLEANED_DATA_PATH)

    numeric_features = ["Order Quantity", "Unit Price", "Shipping Cost", "Order Month", "Order Year"]
    categorical_features = ["Order Priority", "Ship Mode", "Customer Segment", "Product Category"]

    X = df[numeric_features + categorical_features]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("regressor", LinearRegression()),
        ]),
        "Decision Tree Regressor": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("regressor", DecisionTreeRegressor(max_depth=5, min_samples_split=6, random_state=RANDOM_STATE)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=False)),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)),
        ]),
        "Polynomial Regression (Deg 2)": Pipeline([
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, polynomial=True)),
            ("regressor", LinearRegression()),
        ]),
    }

    results = {}
    fitted_models = {}
    predictions = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_m = compute_metrics(y_train, y_train_pred)
        test_m = compute_metrics(y_test, y_test_pred)

        results[name] = {"train": train_m, "test": test_m}
        fitted_models[name] = pipeline
        predictions[name] = (y_test, y_test_pred)

        print(f"\n--- {name} ---")
        print(f"  Train: R2 = {train_m['r2']:.4f} | RMSE = ${train_m['rmse']:,.2f} | MAE = ${train_m['mae']:,.2f}")
        print(f"  Test:  R2 = {test_m['r2']:.4f} | RMSE = ${test_m['rmse']:,.2f} | MAE = ${test_m['mae']:,.2f}")

    # Determine best regression model based on Test R2
    best_model_name = max(results.keys(), key=lambda k: results[k]["test"]["r2"])
    best_pipeline = fitted_models[best_model_name]
    print(f"\n>>> Best Regression Model: {best_model_name} (Test R2: {results[best_model_name]['test']['r2']:.4f})")

    # Serialize models to outputs/models/
    joblib.dump(best_pipeline, MODELS_DIR / "best_sales_regressor.joblib")
    joblib.dump(fitted_models["Linear Regression"], MODELS_DIR / "linear_regression.joblib")
    joblib.dump(fitted_models["Decision Tree Regressor"], MODELS_DIR / "decision_tree_regressor.joblib")
    joblib.dump(fitted_models["Random Forest Regressor"], MODELS_DIR / "random_forest_regressor.joblib")
    joblib.dump(fitted_models["Polynomial Regression (Deg 2)"], MODELS_DIR / "polynomial_regression.joblib")
    print(f"[SAVED] Regression models serialized to {MODELS_DIR}")

    # Plot actual vs predicted & residual errors
    best_actual, best_pred = predictions[best_model_name]
    residuals = best_actual - best_pred

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].scatter(best_actual, best_pred, alpha=0.65, color="#0284c7", edgecolor="white", s=50)
    max_val = max(best_actual.max(), best_pred.max())
    axes[0].plot([0, max_val], [0, max_val], color="#ef4444", linestyle="--", linewidth=2.2, label="Perfect Fit (y=x)")
    axes[0].set_title(f"Actual vs Predicted Sales ({best_model_name})", fontweight="bold")
    axes[0].set_xlabel("Actual Sales ($)")
    axes[0].set_ylabel("Predicted Sales ($)")
    axes[0].legend()

    sns.histplot(residuals, kde=True, ax=axes[1], color="#f59e0b", bins=25)
    axes[1].axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Residual Error Distribution (Actual - Predicted)", fontweight="bold")
    axes[1].set_xlabel("Residual ($)")
    save_plot("08_regression_actual_vs_predicted.png")

    return results, best_model_name, fitted_models, (numeric_features, categorical_features)


if __name__ == "__main__":
    run_regression()
