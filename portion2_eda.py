"""
PORTION 2: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
Retail Sales & Profitability Intelligence

Tasks:
- Compute and plot correlation matrix
- Generate boxplots for outlier analysis
- Analyze distributions of Sales and Profitability
- Visualize performance by Product Category and Customer Segment
- Chart monthly revenue & profit time-series trend (matching executive dashboard)
- Plot top 10 customers by sales volume
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLEANED_DATA_PATH, save_plot


def run_eda(df=None):
    """
    Generate high-resolution EDA figures and save them to outputs/figures/.
    
    Args:
        df (pd.DataFrame, optional): Preprocessed DataFrame. If None, loads from disk.
    
    Returns:
        pd.DataFrame: Active DataFrame used for EDA.
    """
    print("\n" + "=" * 65)
    print("PORTION 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)

    if df is None:
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}. Run Portion 1 first.")
        df = pd.read_csv(CLEANED_DATA_PATH)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.titlesize": 14,
    })

    # 1. Correlation Heatmap
    plt.figure(figsize=(8.5, 6.5))
    numeric_corr_cols = ["Order Quantity", "Sales", "Profit", "Unit Price", "Shipping Cost"]
    corr_matrix = df[numeric_corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1.2,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
    )
    plt.title("Correlation Matrix of Numeric Order Dimensions", fontweight="bold", pad=14)
    save_plot("01_correlation_heatmap.png")

    # 2. Outlier Detection Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    sns.boxplot(y=df["Sales"], ax=axes[0], color="#2b6cb0", width=0.4, fliersize=4)
    axes[0].set_title("Sales ($) Distribution", fontweight="bold")
    axes[0].set_ylabel("Sales ($)")

    sns.boxplot(y=df["Profit"], ax=axes[1], color="#2f855a", width=0.4, fliersize=4)
    axes[1].set_title("Profit ($) Distribution", fontweight="bold")
    axes[1].set_ylabel("Profit ($)")

    sns.boxplot(y=df["Shipping Cost"], ax=axes[2], color="#c53030", width=0.4, fliersize=4)
    axes[2].set_title("Shipping Cost ($) Distribution", fontweight="bold")
    axes[2].set_ylabel("Shipping Cost ($)")
    save_plot("02_numeric_boxplots.png")

    # 3. Sales & Profit Distributions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.histplot(df["Sales"], kde=True, ax=axes[0], color="#0284c7", bins=30)
    axes[0].set_title("Sales Distribution (Positively Skewed)", fontweight="bold")
    axes[0].set_xlabel("Sales ($)")
    axes[0].set_ylabel("Order Count")

    sns.histplot(
        data=df,
        x="Profit",
        hue="is_profitable",
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
        palette={1: "#10b981", 0: "#ef4444"},
    )
    axes[1].axvline(0, color="#1e293b", linestyle="--", linewidth=1.5, alpha=0.8)
    axes[1].set_title("Profit Density: Profitable vs Loss Orders", fontweight="bold")
    axes[1].set_xlabel("Profit ($)")
    axes[1].legend(title="Status", labels=["Profitable (>0)", "Loss (<=0)"])
    save_plot("03_sales_profit_distributions.png")

    # 4. Product Category Breakdown
    cat_summary = (
        df.groupby("Product Category")[["Sales", "Profit", "Order Quantity"]]
        .sum()
        .reset_index()
    )
    cat_summary["Profit Margin %"] = (cat_summary["Profit"] / cat_summary["Sales"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    x_pos = np.arange(len(cat_summary["Product Category"]))
    width = 0.35

    axes[0].bar(x_pos - width / 2, cat_summary["Sales"], width, label="Total Sales ($)", color="#1e40af", alpha=0.9)
    axes[0].bar(x_pos + width / 2, cat_summary["Profit"], width, label="Total Profit ($)", color="#059669", alpha=0.9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(cat_summary["Product Category"], fontweight="semibold")
    axes[0].set_title("Total Sales & Profit by Category", fontweight="bold")
    axes[0].set_ylabel("Total Amount ($)")
    axes[0].legend()

    bars = axes[1].bar(
        cat_summary["Product Category"],
        cat_summary["Profit Margin %"],
        color=["#0284c7", "#10b981", "#f59e0b"],
        alpha=0.9,
    )
    axes[1].set_title("Net Profit Margin (%) by Product Category", fontweight="bold")
    axes[1].set_ylabel("Profit Margin (%)")
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.2f}%", ha="center", fontweight="bold")
    save_plot("04_sales_profit_by_category.png")

    # 5. Ship Mode & Customer Segment Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.countplot(
        data=df,
        x="Ship Mode",
        hue="Order Priority",
        palette="Blues_r",
        ax=axes[0],
    )
    axes[0].set_title("Order Volume by Ship Mode & Priority", fontweight="bold")
    axes[0].set_ylabel("Order Count")
    axes[0].tick_params(axis="x", rotation=15)

    seg_summary = df.groupby("Customer Segment")["Sales"].sum().sort_values(ascending=False).reset_index()
    sns.barplot(data=seg_summary, x="Customer Segment", y="Sales", palette="mako", ax=axes[1])
    axes[1].set_title("Gross Revenue by Customer Segment", fontweight="bold")
    axes[1].set_ylabel("Total Sales ($)")
    axes[1].tick_params(axis="x", rotation=15)
    save_plot("05_ship_mode_customer_segment.png")

    # 6. Monthly Revenue & Profit Trend (Matching User Dashboard Image)
    monthly_trend = (
        df.groupby("Year_Month")[["Sales", "Profit"]]
        .sum()
        .sort_index()
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(16, 6))
    x_idx = np.arange(len(monthly_trend))
    bar_color = "#00B4D8"
    line_color = "#2EC4B6"

    ax1.bar(x_idx, monthly_trend["Sales"], color=bar_color, width=0.62, label="Total Revenue", alpha=0.88)
    ax1.set_ylabel("Total Revenue ($)", color="#0284c7", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0284c7")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(monthly_trend["Year_Month"], rotation=45, ha="right", fontsize=9.5)
    ax1.set_xlabel("Year_Month", fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(x_idx, monthly_trend["Profit"], color=line_color, marker="o", markersize=5, linewidth=2.5, label="Total Profit")
    ax2.axhline(0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Total Profit ($)", color="#0f766e", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="#0f766e")
    ax2.grid(False)

    plt.title("Revenue & Profit Over Time (Executive Monthly Trend)", fontweight="bold", fontsize=14, pad=12)
    save_plot("06_monthly_sales_profit_trend.png")

    # 7. Top 10 Customers by Revenue (Emulating Dashboard Top 10)
    top_cust = df.groupby("Customer Name")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False).head(10).reset_index()
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = sns.barplot(data=top_cust, y="Customer Name", x="Sales", palette="crest", ax=ax)
    ax.set_title("Top 10 Customers by Cumulative Revenue", fontweight="bold")
    ax.set_xlabel("Total Sales ($)")
    ax.set_ylabel("Customer Name")
    for i, v in enumerate(top_cust["Sales"]):
        ax.text(v + 150, i, f"${v/1000:.1f}K", va="center", fontsize=9.5, fontweight="bold", color="#1e293b")
    save_plot("07_top_customers.png")

    print("[SUCCESS] All 7 EDA visualization figures generated.")
    return df


if __name__ == "__main__":
    run_eda()
