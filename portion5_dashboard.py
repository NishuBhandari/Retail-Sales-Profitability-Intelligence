"""
PORTION 5: EVALUATION SUMMARY & INTERACTIVE EXECUTIVE DASHBOARD
Retail Sales & Profitability Intelligence

Tasks:
- Compile evaluation metrics and dataset statistics into outputs/evaluation_summary.json
- Generate the executive dashboard (outputs/dashboard.html & index.html for GitHub Pages)
  featuring dark theme, glassmorphic KPI cards, Chart.js trends,
  category breakdown table, ML model benchmark tables, and high-res figure gallery.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pandas as pd
from config import CLEANED_DATA_PATH, SUMMARY_PATH, DASHBOARD_PATH, BASE_DIR


def save_evaluation_summary(df, reg_results, clf_results, best_reg="Polynomial Regression (Deg 2)", best_clf="Random Forest Classifier"):
    """Save comprehensive evaluation metrics, data summary, and KPIs to JSON."""
    summary_data = {
        "dataset_metrics": {
            "total_records": int(len(df)),
            "date_range": [str(df["Order Date"].min()), str(df["Order Date"].max())],
            "total_sales": round(float(df["Sales"].sum()), 2),
            "total_profit": round(float(df["Profit"].sum()), 2),
            "profit_margin_percentage": round(float((df["Profit"].sum() / df["Sales"].sum()) * 100), 2),
            "total_quantity": int(df["Order Quantity"].sum()),
            "profitable_orders_count": int(df["is_profitable"].sum()),
            "unprofitable_orders_count": int((df["is_profitable"] == 0).sum()),
            "profitable_order_ratio": round(float(df["is_profitable"].mean()), 4),
        },
        "regression": {
            "target": "Sales",
            "best_model": best_reg,
            "models": reg_results,
        },
        "classification": {
            "target": "is_profitable",
            "best_model": best_clf,
            "models": clf_results,
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    print(f"\n[SAVED] Comprehensive summary JSON saved to: {SUMMARY_PATH}")
    return summary_data


def build_dashboard_html(df, reg_results, clf_results, img_prefix="figures/"):
    """
    Build the full HTML dashboard string with interactive Chart.js charts,
    executive KPI cards, category breakdown, model tables, and figure gallery.
    """
    tot_revenue = float(df["Sales"].sum())
    tot_profit = float(df["Profit"].sum())
    margin_pct = (tot_profit / tot_revenue) * 100 if tot_revenue > 0 else 0
    tot_qty = int(df["Order Quantity"].sum())
    prof_rate = float(df["is_profitable"].mean()) * 100

    # Category breakdown table data
    cat_df = df.groupby("Product Category").agg(
        Total_Revenue=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Order Quantity", "sum"),
    ).reset_index()
    cat_df["Profit_Margin"] = (cat_df["Total_Profit"] / cat_df["Total_Revenue"]) * 100

    # Monthly Trend Data for Chart.js
    monthly = df.groupby("Year_Month").agg(
        Revenue=("Sales", "sum"),
        Profit=("Profit", "sum"),
    ).sort_index().reset_index()

    months_labels = monthly["Year_Month"].tolist()
    revenue_vals = monthly["Revenue"].round(2).tolist()
    profit_vals = monthly["Profit"].round(2).tolist()

    cat_labels = cat_df["Product Category"].tolist()
    cat_revenue = cat_df["Total_Revenue"].round(2).tolist()

    # Format category table rows
    cat_rows_html = ""
    for _, row in cat_df.iterrows():
        p_color = "#10b981" if row["Total_Profit"] > 0 else "#ff6b6b"
        cat_rows_html += f"""<tr>
            <td><strong>{row['Product Category']}</strong></td>
            <td>${row['Total_Revenue']:,.2f}</td>
            <td style="color: {p_color};">${row['Total_Profit']:,.2f}</td>
            <td><strong>{row['Profit_Margin']:.2f}%</strong></td>
            <td>{row['Total_Quantity']:,}</td>
        </tr>"""

    # Format regression rows
    reg_rows_html = ""
    for m_name, metrics in reg_results.items():
        test_m = metrics.get("test", metrics)
        is_best = "Poly" in m_name
        star = " ★" if is_best else ""
        badge_style = "background: rgba(0, 210, 211, 0.2); color: #00d2d3;" if is_best else "background: rgba(255,255,255,0.06); color: #cbd5e1;"
        reg_rows_html += f"""<tr>
            <td><strong>{m_name}{star}</strong></td>
            <td><span style="display:inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; {badge_style}">{test_m.get('r2', 0):.4f}</span></td>
            <td>${test_m.get('rmse', 0):,.2f}</td>
            <td>${test_m.get('mae', 0):,.2f}</td>
        </tr>"""

    # Format classification rows
    clf_rows_html = ""
    for m_name, metrics in clf_results.items():
        test_m = metrics.get("test", metrics)
        is_best = "Forest" in m_name or "Tree" in m_name
        star = " ★" if "Forest" in m_name else ""
        badge_style = "background: rgba(16, 185, 129, 0.2); color: #10b981;" if "Forest" in m_name else "background: rgba(255,255,255,0.06); color: #cbd5e1;"
        clf_rows_html += f"""<tr>
            <td><strong>{m_name}{star}</strong></td>
            <td><span style="display:inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; {badge_style}">{test_m.get('accuracy', 0):.1%}</span></td>
            <td>{test_m.get('f1', 0):.4f}</td>
            <td>{test_m.get('precision', 0):.4f}</td>
            <td>{test_m.get('recall', 0):.4f}</td>
        </tr>"""

    margin_color = "green" if margin_pct > 0 else "coral"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Sales & Profitability Intelligence Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {{
            --bg-dark: #121824;
            --card-bg: #1a2332;
            --card-border: #253347;
            --accent-cyan: #00d2d3;
            --accent-green: #10b981;
            --accent-coral: #ff6b6b;
            --accent-yellow: #feca57;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 18px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            font-size: 26px;
            font-weight: 700;
            color: var(--text-main);
            letter-spacing: -0.5px;
        }}
        .header .badge {{
            background: #2563eb;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .kpi-value.cyan {{ color: var(--accent-cyan); }}
        .kpi-value.green {{ color: var(--accent-green); }}
        .kpi-value.coral {{ color: var(--accent-coral); }}
        .kpi-value.yellow {{ color: var(--accent-yellow); }}
        .kpi-label {{
            font-size: 13px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-title {{
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-main);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        tr:last-child td {{
            border-bottom: none;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.02);
        }}
        .ml-section {{
            margin-top: 32px;
            border-top: 1px solid var(--card-border);
            padding-top: 24px;
        }}
        .model-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .metric-badge {{
            display: inline-block;
            background: rgba(0, 210, 211, 0.15);
            color: var(--accent-cyan);
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
            margin-right: 6px;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 16px;
            margin-top: 20px;
        }}
        .gallery-item {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            overflow: hidden;
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .gallery-title {{
            padding: 10px 14px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-muted);
            background: #141b26;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Retail Sales & Profitability Intelligence</h1>
            <p style="color: var(--text-muted); font-size: 13px; margin-top: 4px;">Executive Performance Dashboard & Machine Learning Diagnostics</p>
        </div>
        <div class="badge">ML Production Pipeline</div>
    </div>

    <!-- Executive KPI Cards -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-value cyan">${tot_revenue:,.2f}</div>
            <div class="kpi-label">Total Revenue</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value green">${tot_profit:,.2f}</div>
            <div class="kpi-label">Total Profit</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value {margin_color}">{margin_pct:.2f}%</div>
            <div class="kpi-label">Profit Margin %</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value yellow">{tot_qty:,}</div>
            <div class="kpi-label">Total Quantity Sold</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value cyan">{prof_rate:.1f}%</div>
            <div class="kpi-label">Profitable Order Rate</div>
        </div>
    </div>

    <!-- Main Charts Row -->
    <div class="grid-2">
        <div class="card">
            <div class="card-title">
                <span>Revenue & Profit Over Time</span>
                <span style="font-size: 12px; color: var(--text-muted);">Monthly Breakdown</span>
            </div>
            <div style="height: 320px; position: relative;">
                <canvas id="monthlyTrendChart"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <span>Revenue by Category</span>
                <span style="font-size: 12px; color: var(--text-muted);">Share of Sales</span>
            </div>
            <div style="height: 320px; display: flex; align-items: center; justify-content: center; position: relative;">
                <canvas id="categoryDonutChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Category Breakdown Table -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-title">Category Intelligence Breakdown</div>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Total Revenue</th>
                    <th>Total Profit</th>
                    <th>Profit Margin %</th>
                    <th>Total Quantity</th>
                </tr>
            </thead>
            <tbody>
                {cat_rows_html}
                <tr>
                    <td>Total</td>
                    <td>${tot_revenue:,.2f}</td>
                    <td style="color: {'#10b981' if tot_profit > 0 else '#ff6b6b'};">${tot_profit:,.2f}</td>
                    <td>{margin_pct:.2f}%</td>
                    <td>{tot_qty:,}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <!-- Machine Learning Section -->
    <div class="ml-section">
        <h2 style="font-size: 20px; font-weight: 700; margin-bottom: 16px;">Machine Learning Evaluation & Diagnostics</h2>
        <div class="model-grid">
            <!-- Regression Card -->
            <div class="card">
                <div class="card-title">
                    <span>Regression: Predicting Sales</span>
                    <span class="metric-badge">Best: Poly Reg (Deg 2)</span>
                </div>
                <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 14px;">Evaluates Linear Regression, Decision Tree, Random Forest, and Polynomial Features (Deg 2).</p>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Test R²</th>
                            <th>Test RMSE</th>
                            <th>Test MAE</th>
                        </tr>
                    </thead>
                    <tbody>
                        {reg_rows_html}
                    </tbody>
                </table>
            </div>

            <!-- Classification Card -->
            <div class="card">
                <div class="card-title">
                    <span>Classification: Predicting Profitability</span>
                    <span class="metric-badge">Best: Random Forest</span>
                </div>
                <p style="color: var(--text-muted); font-size: 13px; margin-bottom: 14px;">Predicts whether an order will yield positive profit (is_profitable = 1).</p>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Test Accuracy</th>
                            <th>Test F1-Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {clf_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Visual Figures Gallery -->
        <h3 style="font-size: 16px; font-weight: 700; margin-top: 24px; margin-bottom: 12px; color: var(--text-muted);">Generated High-Resolution Visual Artifacts</h3>
        <div class="gallery-grid">
            <div class="gallery-item">
                <div class="gallery-title">06. Monthly Revenue & Profit Trend</div>
                <img src="{img_prefix}06_monthly_sales_profit_trend.png" alt="Monthly Trend">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">08. Regression: Actual vs Predicted</div>
                <img src="{img_prefix}08_regression_actual_vs_predicted.png" alt="Regression Actual vs Pred">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">09. Classification: Confusion Matrices</div>
                <img src="{img_prefix}09_classification_confusion_matrices.png" alt="Confusion Matrices">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">10. Feature Importance Drivers</div>
                <img src="{img_prefix}10_feature_importance.png" alt="Feature Importance">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">01. Correlation Heatmap</div>
                <img src="{img_prefix}01_correlation_heatmap.png" alt="Correlation Heatmap">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">04. Category Sales & Profit</div>
                <img src="{img_prefix}04_sales_profit_by_category.png" alt="Category Sales Profit">
            </div>
        </div>
    </div>

    <script>
        window.addEventListener('DOMContentLoaded', function() {{
            // Chart 1: Revenue & Profit Combo
            const canvasTrend = document.getElementById('monthlyTrendChart');
            if (canvasTrend) {{
                const ctxTrend = canvasTrend.getContext('2d');
                new Chart(ctxTrend, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(months_labels)},
                        datasets: [
                            {{
                                type: 'bar',
                                label: 'Total Revenue ($)',
                                data: {json.dumps(revenue_vals)},
                                backgroundColor: '#00d2d3',
                                borderRadius: 3,
                                yAxisID: 'y'
                            }},
                            {{
                                type: 'line',
                                label: 'Total Profit ($)',
                                data: {json.dumps(profit_vals)},
                                borderColor: '#10b981',
                                borderWidth: 2.5,
                                pointBackgroundColor: '#10b981',
                                pointRadius: 2.5,
                                tension: 0.2,
                                yAxisID: 'y1'
                            }}
                        ]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            x: {{
                                ticks: {{ color: '#94a3b8', font: {{ size: 9 }}, maxRotation: 45 }},
                                grid: {{ color: '#1e293b' }}
                            }},
                            y: {{
                                position: 'left',
                                ticks: {{ color: '#00d2d3', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'k' : v) }},
                                grid: {{ color: '#1e293b' }}
                            }},
                            y1: {{
                                position: 'right',
                                ticks: {{ color: '#10b981', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'k' : v) }},
                                grid: {{ drawOnChartArea: false }}
                            }}
                        }},
                        plugins: {{
                            legend: {{ labels: {{ color: '#f1f5f9' }} }}
                        }}
                    }}
                }});
            }}

            // Chart 2: Category Donut
            const canvasDonut = document.getElementById('categoryDonutChart');
            if (canvasDonut) {{
                const ctxDonut = canvasDonut.getContext('2d');
                new Chart(ctxDonut, {{
                    type: 'doughnut',
                    data: {{
                        labels: {json.dumps(cat_labels)},
                        datasets: [{{
                            data: {json.dumps(cat_revenue)},
                            backgroundColor: ['#2563eb', '#00d2d3', '#feca57'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            legend: {{
                                position: 'bottom',
                                labels: {{ color: '#f1f5f9', font: {{ size: 11 }} }}
                            }}
                        }}
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""
    return html


def run_dashboard_summary(df=None, reg_results=None, clf_results=None, best_reg="Polynomial Regression (Deg 2)", best_clf="Random Forest Classifier"):
    """Orchestrates portion 5 summary and dashboard creation."""
    print("\n" + "=" * 65)
    print("PORTION 5: SUMMARY REPORT & INTERACTIVE DASHBOARD")
    print("=" * 65)

    if df is None:
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}. Run Portion 1 first.")
        df = pd.read_csv(CLEANED_DATA_PATH)

    if reg_results is None or clf_results is None:
        if SUMMARY_PATH.exists():
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                prev_summary = json.load(f)
                reg_results = prev_summary.get("regression", {}).get("models", {})
                clf_results = prev_summary.get("classification", {}).get("models", {})
                best_reg = prev_summary.get("regression", {}).get("best_model", best_reg)
                best_clf = prev_summary.get("classification", {}).get("best_model", best_clf)
        else:
            reg_results = {}
            clf_results = {}

    save_evaluation_summary(df, reg_results, clf_results, best_reg, best_clf)

    # 1. Write outputs/dashboard.html (relative to outputs folder, images in figures/)
    html_local = build_dashboard_html(df, reg_results, clf_results, img_prefix="figures/")
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_local)
    print(f"[SAVED] Dashboard saved to: {DASHBOARD_PATH}")

    # 2. Write root index.html for GitHub Pages (images in outputs/figures/)
    html_root = build_dashboard_html(df, reg_results, clf_results, img_prefix="outputs/figures/")
    index_path = BASE_DIR / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_root)
    print(f"[SAVED] Root index.html for GitHub Pages saved to: {index_path}")

    print("[SUCCESS] Portion 5 completed successfully.")


if __name__ == "__main__":
    run_dashboard_summary()
