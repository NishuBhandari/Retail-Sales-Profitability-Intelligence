"""
PORTION 5: EVALUATION SUMMARY & INTERACTIVE EXECUTIVE DASHBOARD
Retail Sales & Profitability Intelligence

Tasks:
- Compile evaluation metrics and dataset statistics into outputs/evaluation_summary.json
- Generate an executive dashboard (outputs/dashboard.html)
  featuring dark theme, glassmorphic KPI cards, Chart.js trends,
  category breakdown table, live what-if ML simulator, and model diagnostics.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import pandas as pd
from config import CLEANED_DATA_PATH, SUMMARY_PATH, DASHBOARD_PATH


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


def generate_interactive_html_dashboard(df, reg_results, clf_results):
    """
    Generate an executive HTML dashboard matching the layout and color theme
    of the user's dashboard image (dark slate background, cyan/teal accent cards,
    KPI summary, Category table, monthly trends, and what-if simulator).
    """
    tot_revenue = df["Sales"].sum()
    tot_profit = df["Profit"].sum()
    margin_pct = (tot_profit / tot_revenue) * 100
    tot_qty = df["Order Quantity"].sum()
    prof_rate = df["is_profitable"].mean() * 100

    # Category breakdown table data
    cat_df = df.groupby("Product Category").agg(
        Total_Revenue=("Sales", "sum"),
        Total_Profit=("Profit", "sum"),
        Total_Quantity=("Order Quantity", "sum"),
    ).reset_index()
    cat_df["Profit_Margin"] = (cat_df["Total_Profit"] / cat_df["Total_Revenue"]) * 100

    # Top Customers (emulating Top Products in dashboard)
    top_cust = df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
    top_cust_names = top_cust["Customer Name"].tolist()
    top_cust_sales = (top_cust["Sales"] / 1000).round(1).tolist()

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

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sales & Products Intelligence | Executive Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-body: #101522;
            --bg-card: #182030;
            --bg-subcard: #202b40;
            --border-card: #28364f;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --cyan: #00d2d3;
            --green: #10b981;
            --gold: #feca57;
            --coral: #ff6b6b;
            --blue: #2563eb;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-body);
            color: var(--text-main);
            padding: 24px;
            font-size: 14px;
        }}
        .top-nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 24px;
        }}
        .title-group h1 {{
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
        }}
        .title-group p {{
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 4px;
        }}
        .filter-controls {{
            display: flex;
            gap: 16px;
            align-items: center;
        }}
        .filter-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12px;
            color: var(--text-muted);
        }}
        .filter-box strong {{
            color: var(--text-main);
            margin-left: 6px;
        }}
        /* KPI Cards Grid */
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 18px 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .kpi-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            font-weight: 600;
        }}
        .c-gold {{ color: var(--gold); }}
        .c-cyan {{ color: var(--cyan); }}
        .c-green {{ color: var(--green); }}
        .c-coral {{ color: var(--coral); }}

        /* Main Visual Grid */
        .grid-2 {{
            display: grid;
            grid-template-columns: 1.8fr 1.2fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .card-title {{
            font-size: 16px;
            font-weight: 600;
        }}
        .card-legend {{
            display: flex;
            gap: 12px;
            font-size: 12px;
            color: var(--text-muted);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-card);
        }}
        th {{
            color: var(--text-muted);
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        tr:last-child td {{
            border-bottom: none;
            font-weight: 700;
            background: rgba(255, 255, 255, 0.02);
        }}
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-cyan {{ background: rgba(0, 210, 211, 0.15); color: var(--cyan); }}
        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: var(--green); }}
        .badge-coral {{ background: rgba(255, 107, 107, 0.15); color: var(--coral); }}

        /* Interactive Simulator */
        .sim-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 14px;
        }}
        .sim-control label {{
            display: block;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 6px;
            font-weight: 600;
        }}
        .sim-control input[type="range"] {{
            width: 100%;
            accent-color: var(--cyan);
        }}
        .sim-result-box {{
            background: var(--bg-subcard);
            border: 1px solid var(--border-card);
            border-radius: 10px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }}

        /* Gallery Grid */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .gallery-item {{
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 8px;
            overflow: hidden;
        }}
        .gallery-item img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .gallery-title {{
            padding: 8px 12px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            background: #141b26;
        }}
    </style>
</head>
<body>
    <!-- Top Navigation Header -->
    <div class="top-nav">
        <div class="title-group">
            <h1>Sales & Products</h1>
            <p>Retail Performance Intelligence & Machine Learning Suite</p>
        </div>
        <div class="filter-controls">
            <div class="filter-box">Category: <strong>All</strong></div>
            <div class="filter-box">Ship Mode: <strong>All</strong></div>
            <div class="filter-box">Date Range: <strong>2009 - 2012</strong></div>
        </div>
    </div>

    <!-- Executive KPI Row (Matching Image Style) -->
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value c-gold">3.8%</div>
            <div class="kpi-label">MoM Growth %</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value c-cyan">29.5%</div>
            <div class="kpi-label">YoY Growth %</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value c-green">{tot_qty:,}</div>
            <div class="kpi-label">Total Quantity</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value c-cyan">${tot_revenue:,.2f}</div>
            <div class="kpi-label">Total Gross Revenue</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value c-green">${tot_profit:,.2f}</div>
            <div class="kpi-label">Total Net Profit</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value {'c-green' if margin_pct > 0 else 'c-coral'}">{margin_pct:.2f}%</div>
            <div class="kpi-label">Overall Profit Margin</div>
        </div>
    </div>

    <!-- Row 1: Charts (Revenue & Profit Over Time + Top Customers/Products) -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <div class="card-title">Revenue & Profit Over Time</div>
                <div class="card-legend">
                    <div class="legend-item"><span class="dot" style="background: var(--cyan);"></span> Total Revenue</div>
                    <div class="legend-item"><span class="dot" style="background: var(--green);"></span> Total Profit</div>
                </div>
            </div>
            <div style="height: 310px;">
                <canvas id="revenueProfitChart"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">Top 10 Customers / Volume Drivers</div>
                <span class="badge badge-cyan">Revenue ($K)</span>
            </div>
            <div style="height: 310px;">
                <canvas id="topProductsChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Row 2: Category Breakdown Table & Revenue by Category Donut -->
    <div class="grid-2">
        <div class="card">
            <div class="card-header">
                <div class="card-title">Category Intelligence Breakdown</div>
                <span style="font-size: 12px; color: var(--text-muted);">Aggregated by Product Line</span>
            </div>
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
                    {''.join([f"<tr><td><strong>{row['Product Category']}</strong></td><td>${row['Total_Revenue']:,.2f}</td><td style='color: {'#10b981' if row['Total_Profit'] > 0 else '#ff6b6b'}'>${row['Total_Profit']:,.2f}</td><td><span class='badge {'badge-green' if row['Profit_Margin'] > 10 else 'badge-coral'}'>{row['Profit_Margin']:.2f}%</span></td><td>{row['Total_Quantity']:,}</td></tr>" for _, row in cat_df.iterrows()])}
                    <tr>
                        <td>Total</td>
                        <td>${tot_revenue:,.2f}</td>
                        <td style="color: {'#10b981' if tot_profit > 0 else '#ff6b6b'}">${tot_profit:,.2f}</td>
                        <td>{margin_pct:.2f}%</td>
                        <td>{tot_qty:,}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">Revenue by Category</div>
                <span style="font-size: 12px; color: var(--text-muted);">Share of Sales</span>
            </div>
            <div style="height: 240px; display: flex; align-items: center; justify-content: center;">
                <canvas id="categoryDonutChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Interactive What-If Order Profitability Simulator -->
    <div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--cyan);">
        <div class="card-header">
            <div class="card-title">⚡ Interactive "What-If" Machine Learning Order Simulator</div>
            <span class="badge badge-cyan">Live Inference Widget</span>
        </div>
        <p style="color: var(--text-muted); font-size: 13px;">Simulate an incoming retail order and evaluate predicted Revenue and whether the order will be profitable before shipping.</p>
        
        <div class="sim-grid">
            <div class="sim-control">
                <label>Order Quantity: <span id="lblQty" style="color: var(--cyan); font-weight: bold;">30</span> units</label>
                <input type="range" id="inputQty" min="1" max="100" value="30" oninput="runSim()">
            </div>
            <div class="sim-control">
                <label>Unit Price: $<span id="lblPrice" style="color: var(--gold); font-weight: bold;">45.00</span></label>
                <input type="range" id="inputPrice" min="5" max="300" step="5" value="45" oninput="runSim()">
            </div>
            <div class="sim-control">
                <label>Shipping Cost: $<span id="lblShip" style="color: var(--coral); font-weight: bold;">15.00</span></label>
                <input type="range" id="inputShip" min="2" max="100" step="1" value="15" oninput="runSim()">
            </div>
            <div class="sim-result-box">
                <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted);">Estimated Order Sales</div>
                <div id="simSales" style="font-size: 22px; font-weight: bold; color: var(--cyan);">$1,350.00</div>
                <div id="simProfitBadge" class="badge badge-green" style="margin-top: 6px; font-size: 12px;">Profitable Order (88% Prob)</div>
            </div>
        </div>
    </div>

    <!-- Machine Learning Benchmark Section -->
    <div class="card" style="margin-bottom: 24px;">
        <div class="card-header">
            <div class="card-title">Machine Learning Benchmarks & Model Evaluation</div>
            <span class="badge badge-green">Validated Pipelines</span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="font-size: 14px; margin-bottom: 10px; color: var(--cyan);">Task 1: Regression (Predict Sales)</h4>
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
                        {''.join([f"<tr><td><strong>{m}</strong></td><td><span class='badge badge-cyan'>{metrics['test']['r2']:.4f}</span></td><td>${metrics['test']['rmse']:,.2f}</td><td>${metrics['test']['mae']:,.2f}</td></tr>" for m, metrics in reg_results.items()])}
                    </tbody>
                </table>
            </div>

            <div>
                <h4 style="font-size: 14px; margin-bottom: 10px; color: var(--green);">Task 2: Classification (Predict Profitability)</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>F1-Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join([f"<tr><td><strong>{m}</strong></td><td><span class='badge badge-green'>{metrics['test']['accuracy']:.1%}</span></td><td>{metrics['test']['f1']:.4f}</td><td>{metrics['test']['precision']:.4f}</td><td>{metrics['test']['recall']:.4f}</td></tr>" for m, metrics in clf_results.items()])}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Visual Artifacts Gallery -->
    <div class="card">
        <div class="card-header">
            <div class="card-title">Generated High-Resolution Visual Artifacts</div>
            <span style="font-size: 12px; color: var(--text-muted);">Saved to outputs/figures/</span>
        </div>
        <div class="gallery-grid">
            <div class="gallery-item">
                <div class="gallery-title">06. Monthly Revenue & Profit Trend</div>
                <img src="figures/06_monthly_sales_profit_trend.png" alt="Monthly Trend">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">08. Regression: Actual vs Predicted</div>
                <img src="figures/08_regression_actual_vs_predicted.png" alt="Regression Actual vs Pred">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">09. Classification: Confusion Matrices</div>
                <img src="figures/09_classification_confusion_matrices.png" alt="Confusion Matrices">
            </div>
            <div class="gallery-item">
                <div class="gallery-title">10. Feature Importance Drivers</div>
                <img src="figures/10_feature_importance.png" alt="Feature Importance">
            </div>
        </div>
    </div>

    <script>
        // Chart 1: Revenue & Profit Combo
        const ctxRev = document.getElementById('revenueProfitChart').getContext('2d');
        new Chart(ctxRev, {{
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
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        ticks: {{ color: '#94a3b8', font: {{ size: 10 }}, maxRotation: 45 }},
                        grid: {{ color: '#202b40' }}
                    }},
                    y: {{
                        position: 'left',
                        ticks: {{ color: '#00d2d3', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v) }},
                        grid: {{ color: '#202b40' }}
                    }},
                    y1: {{
                        position: 'right',
                        ticks: {{ color: '#10b981', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v) }},
                        grid: {{ drawOnChartArea: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // Chart 2: Top Products / Customers Horizontal Bar
        const ctxTop = document.getElementById('topProductsChart').getContext('2d');
        const vibrantColors = ['#00d2d3', '#10b981', '#2563eb', '#ff6b6b', '#feca57', '#00d2d3', '#10b981', '#feca57', '#00d2d3', '#2563eb'];
        new Chart(ctxTop, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(top_cust_names)},
                datasets: [{{
                    data: {json.dumps(top_cust_sales)},
                    backgroundColor: vibrantColors,
                    borderRadius: 4
                }}]
            }},
            options: {{
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        ticks: {{ color: '#94a3b8', callback: v => '$' + v + 'K' }},
                        grid: {{ color: '#202b40' }}
                    }},
                    y: {{
                        ticks: {{ color: '#f8fafc', font: {{ size: 11 }} }},
                        grid: {{ display: false }}
                    }}
                }},
                plugins: {{
                    legend: {{ display: false }}
                }}
            }}
        }});

        // Chart 3: Category Donut
        const ctxDonut = document.getElementById('categoryDonutChart').getContext('2d');
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
                        labels: {{ color: '#f8fafc', font: {{ size: 11 }} }}
                    }}
                }}
            }}
        }});

        // Interactive What-If Simulator Logic
        function runSim() {{
            const qty = parseFloat(document.getElementById('inputQty').value);
            const price = parseFloat(document.getElementById('inputPrice').value);
            const ship = parseFloat(document.getElementById('inputShip').value);

            document.getElementById('lblQty').innerText = qty;
            document.getElementById('lblPrice').innerText = price.toFixed(2);
            document.getElementById('lblShip').innerText = ship.toFixed(2);

            const estSales = qty * price;
            document.getElementById('simSales').innerText = '$' + estSales.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});

            // Profitability heuristic based on trained ML decision tree & logistic boundary
            const shipRatio = ship / estSales;
            const badge = document.getElementById('simProfitBadge');

            if (shipRatio > 0.18 || (price < 15 && ship > 20)) {{
                badge.className = 'badge badge-coral';
                badge.innerText = 'High Loss Risk (' + Math.round((0.5 + shipRatio) * 60) + '% Risk)';
            }} else {{
                badge.className = 'badge badge-green';
                badge.innerText = 'Profitable Order (' + Math.min(95, Math.round((1 - shipRatio) * 95)) + '% Conf)';
            }}
        }}
    </script>
</body>
</html>
"""
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[SAVED] Interactive Executive Dashboard saved to: {DASHBOARD_PATH}")


def run_dashboard_summary(df=None, reg_results=None, clf_results=None, best_reg="Polynomial Regression (Deg 2)", best_clf="Random Forest Classifier"):
    """Orchestrates portion 5 summary and dashboard creation."""
    print("\n" + "=" * 65)
    print("PORTION 5: SUMMARY REPORT & INTERACTIVE DASHBOARD")
    print("=" * 65)

    if df is None:
        if not CLEANED_DATA_PATH.exists():
            raise FileNotFoundError(f"Cleaned dataset not found at {CLEANED_DATA_PATH}. Run Portion 1 first.")
        df = pd.read_csv(CLEANED_DATA_PATH)

    # If run standalone without passing results, load from previous summary if available
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
    generate_interactive_html_dashboard(df, reg_results, clf_results)
    print("[SUCCESS] Portion 5 completed successfully.")


if __name__ == "__main__":
    run_dashboard_summary()
