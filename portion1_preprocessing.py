"""
PORTION 1: DATA LOADING, CLEANING & PREPROCESSING
Retail Sales & Profitability Intelligence

Tasks:
- Load raw sales dataset with automatic path fallback
- Clean categorical typos ('Loww' -> 'Low')
- Logically impute missing values using domain formulas
- Engineer temporal features and binary 'is_profitable' classification target
- Export cleaned dataset to outputs/cleaned_sales_data.csv
"""

import sys
from pathlib import Path

# Ensure module can be imported from any VS Code working directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from config import DATA_PATH, CLEANED_DATA_PATH


def run_preprocessing(data_path=DATA_PATH):
    """
    Load raw CSV data, perform cleaning, handle missing values,
    engineer temporal and profitability features, and save cleaned CSV.
    
    Returns:
        pd.DataFrame: Cleaned and feature-engineered dataset.
    """
    print("\n" + "=" * 65)
    print("PORTION 1: DATA LOADING & PREPROCESSING")
    print("=" * 65)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Reading dataset from: {data_path.name}")
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Inspect missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nMissing values detected in raw data:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"  - {col}: {count} missing")

    # 1. Clean categorical inconsistencies
    typo_count = (df["Order Priority"] == "Loww").sum()
    if typo_count > 0:
        print(f"Correcting {typo_count} typo instances ('Loww' -> 'Low') in 'Order Priority'...")
        df["Order Priority"] = df["Order Priority"].replace({"Loww": "Low"})

    # Impute categorical nulls using column mode
    df["Order Priority"] = df["Order Priority"].fillna(df["Order Priority"].mode()[0])
    df["Ship Mode"] = df["Ship Mode"].fillna(df["Ship Mode"].mode()[0])
    df["Customer Name"] = df["Customer Name"].fillna("Unknown Customer")

    # 2. Impute numeric values using logical business relationships
    # Unit Price: Sales / Order Quantity where possible, else median
    unit_price_calc = df["Sales"] / df["Order Quantity"].replace(0, np.nan)
    df["Unit Price"] = df["Unit Price"].fillna(unit_price_calc).fillna(df["Unit Price"].median())

    # Sales: Order Quantity * Unit Price where missing, else median
    sales_calc = df["Order Quantity"] * df["Unit Price"]
    df["Sales"] = df["Sales"].fillna(sales_calc).fillna(df["Sales"].median())

    # Profit: fillna with median profit
    df["Profit"] = df["Profit"].fillna(df["Profit"].median())

    # 3. Create Classification Target: Profitability
    # 1 = Profitable (Profit > 0), 0 = Loss / Break-even (Profit <= 0)
    df["is_profitable"] = (df["Profit"] > 0).astype(int)

    # 4. Temporal feature engineering from Order Date
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed", errors="coerce")
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter
    df["Order DayOfWeek"] = df["Order Date"].dt.dayofweek
    df["Year_Month"] = df["Order Date"].dt.to_period("M").astype(str)

    # 5. Advanced Domain Ratios (for feature analysis)
    # Unit Shipping Cost: Shipping Cost per unit quantity
    df["Unit_Shipping_Cost"] = (df["Shipping Cost"] / df["Order Quantity"].replace(0, 1)).round(2)
    # Shipping Ratio: Shipping Cost as % of Sales
    df["Shipping_Cost_Ratio"] = (df["Shipping Cost"] / df["Sales"].replace(0, np.nan)).fillna(0).round(4)

    # Note constant company column
    if "company" in df.columns and df["company"].nunique() == 1:
        print(f"Column 'company' is constant ('{df['company'].iloc[0]}'). Preserved for reference.")

    # Save cleaned data to disk
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    print(f"Target breakdown ('is_profitable'):")
    print(f"  - Profitable (1):   {(df['is_profitable'] == 1).sum()} ({(df['is_profitable'] == 1).mean():.1%})")
    print(f"  - Unprofitable (0): {(df['is_profitable'] == 0).sum()} ({(df['is_profitable'] == 0).mean():.1%})")

    return df


if __name__ == "__main__":
    run_preprocessing()
