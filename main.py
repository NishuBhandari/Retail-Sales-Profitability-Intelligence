"""
MAIN PIPELINE ORCHESTRATOR
Retail Sales & Profitability Intelligence

Connects all 5 modular portions:
  - Portion 1: Data Preprocessing & Cleaning (portion1_preprocessing.py)
  - Portion 2: Exploratory Data Analysis (portion2_eda.py)
  - Portion 3: Regression Modeling (portion3_regression.py)
  - Portion 4: Classification Modeling (portion4_classification.py)
  - Portion 5: Dashboard & Evaluation Summary (portion5_dashboard.py)
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from portion1_preprocessing import run_preprocessing
from portion2_eda import run_eda
from portion3_regression import run_regression
from portion4_classification import run_classification
from portion5_dashboard import run_dashboard_summary
from config import OUTPUT_DIR, FIGURES_DIR, MODELS_DIR, DASHBOARD_PATH, SUMMARY_PATH, CLEANED_DATA_PATH


def main():
    print("=" * 65)
    print("RETAIL SALES & PROFITABILITY INTELLIGENCE (5-PORTION MODULAR PIPELINE)")
    print("=" * 65)

    # Portion 1: Preprocessing & Cleaning
    df = run_preprocessing()

    # Portion 2: Exploratory Data Analysis & Plots
    run_eda(df)

    # Portion 3: Regression Pipeline (Predict Sales)
    reg_results, best_reg, reg_models, reg_feat_names = run_regression(df)

    # Portion 4: Classification Pipeline (Predict Profitability)
    clf_results, best_clf, clf_models, clf_feat_names = run_classification(
        df, reg_models=reg_models, reg_feat_names=reg_feat_names
    )

    # Portion 5: Evaluation Summary & Interactive HTML Dashboard
    run_dashboard_summary(
        df=df,
        reg_results=reg_results,
        clf_results=clf_results,
        best_reg=best_reg,
        best_clf=best_clf,
    )

    print("\n" + "=" * 65)
    print("ALL 5 PORTIONS EXECUTED AND CONNECTED SUCCESSFULLY!")
    print(f"Artifacts and outputs saved to: {OUTPUT_DIR}")
    print(f"  - Cleaned Data: {CLEANED_DATA_PATH}")
    print(f"  - Figures:      {FIGURES_DIR}")
    print(f"  - Models:       {MODELS_DIR}")
    print(f"  - Summary JSON: {SUMMARY_PATH}")
    print(f"  - BI Dashboard: {DASHBOARD_PATH}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
