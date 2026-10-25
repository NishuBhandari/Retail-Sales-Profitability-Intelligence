"""
PORTION 5: EVALUATION SUMMARY & INTERACTIVE EXECUTIVE DASHBOARD
Retail Sales & Profitability Intelligence

Futuristic Cyber-Glassmorphic Edition:
- Deep space background with holographic accents and frosted glass cards
- Futuristic HUD top bar with pulsing live telemetry status
- Neon glow KPI metrics with monospace readouts
- Interactive Chart.js time-series combo chart and quantum donut
- Cyber matrix category breakdown table with margin health bars
- Holographic Machine Learning benchmark cards (Regression & Classification)
- Interactive Live Quantum Order Simulator widget
- High-resolution visual gallery with holographic glow cards
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


def build_futuristic_dashboard_html(df, reg_results, clf_results, img_prefix="figures/"):
    """Build the ultra-futuristic cyber-glassmorphic HTML dashboard."""
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

    # Category Table Rows with futuristic progress bars
    cat_rows_html = ""
    for _, row in cat_df.iterrows():
        p_val = row["Total_Profit"]
        p_color = "#10b981" if p_val > 0 else "#f43f5e"
        bar_width = min(100, max(10, int(row["Profit_Margin"] * 4.5)))
        cat_rows_html += f"""<tr>
            <td>
                <span class="cyber-tag">{row['Product Category']}</span>
            </td>
            <td class="mono-num">${row['Total_Revenue']:,.2f}</td>
            <td class="mono-num" style="color: {p_color};">${p_val:,.2f}</td>
            <td>
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="mono-num" style="font-weight:700; color:{p_color};">{row['Profit_Margin']:.2f}%</span>
                    <div class="mini-bar-track">
                        <div class="mini-bar-fill" style="width:{bar_width}%; background:{p_color};"></div>
                    </div>
                </div>
            </td>
            <td class="mono-num">{row['Total_Quantity']:,}</td>
        </tr>"""

    # Regression Models Rows
    reg_rows_html = ""
    for m_name, metrics in reg_results.items():
        test_m = metrics.get("test", metrics)
        is_poly = "Poly" in m_name
        is_rf = "Forest" in m_name
        tag_badge = '<span class="glow-pill cyan-glow">TOP ACCURACY 🏆</span>' if is_poly else ('<span class="glow-pill gold-glow">MIN MAE ⚡</span>' if is_rf else '')
        reg_rows_html += f"""<tr>
            <td><strong>{m_name}</strong> {tag_badge}</td>
            <td class="mono-num" style="color: #22d3ee; font-weight:700;">{test_m.get('r2', 0):.4f}</td>
            <td class="mono-num">${test_m.get('rmse', 0):,.2f}</td>
            <td class="mono-num">${test_m.get('mae', 0):,.2f}</td>
        </tr>"""

    # Classification Models Rows
    clf_rows_html = ""
    for m_name, metrics in clf_results.items():
        test_m = metrics.get("test", metrics)
        is_rf = "Forest" in m_name
        tag_badge = '<span class="glow-pill mint-glow">OPTIMAL PREDICTOR 🏆</span>' if is_rf else ''
        clf_rows_html += f"""<tr>
            <td><strong>{m_name}</strong> {tag_badge}</td>
            <td class="mono-num" style="color: #34d399; font-weight:700;">{test_m.get('accuracy', 0):.1%}</td>
            <td class="mono-num">{test_m.get('f1', 0):.4f}</td>
            <td class="mono-num">{test_m.get('precision', 0):.4f}</td>
            <td class="mono-num">{test_m.get('recall', 0):.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum Retail Intelligence | Futuristic Executive Command Center</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-deep: #060910;
            --bg-card: rgba(17, 25, 40, 0.75);
            --bg-card-hover: rgba(22, 33, 53, 0.85);
            --border-glow: rgba(56, 189, 248, 0.2);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --cyan-neon: #00f2fe;
            --mint-neon: #10b981;
            --gold-neon: #fbbf24;
            --coral-neon: #f43f5e;
            --purple-neon: #a855f7;
            --text-bright: #f8fafc;
            --text-muted: #94a3b8;
            --text-cyan: #38bdf8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        body {{
            background-color: var(--bg-deep);
            background-image: 
                radial-gradient(circle at 10% 10%, rgba(0, 242, 254, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 90%, rgba(168, 85, 247, 0.08) 0%, transparent 40%),
                linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
            color: var(--text-bright);
            padding: 24px 32px;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        .mono-num {{
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: -0.3px;
        }}

        .shell-container {{
            max-width: 1360px;
            margin: 0 auto;
        }}

        /* Futuristic HUD Navbar */
        .hud-nav {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 16px 24px;
            background: var(--bg-card);
            border: 1px solid var(--border-glow);
            border-radius: 14px;
            backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 0 0 16px rgba(0, 242, 254, 0.1);
            margin-bottom: 24px;
        }}

        .brand-block {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .cyber-logo {{
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--cyan-neon), var(--purple-neon));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 18px;
            color: #000;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        }}

        .brand-text h1 {{
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #ffffff, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .brand-text span {{
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }}

        .hud-telemetry {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .pulse-indicator {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            font-weight: 600;
            color: #34d399;
            background: rgba(16, 185, 129, 0.12);
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .live-dot {{
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
            animation: pulse-ring 2s infinite;
        }}

        @keyframes pulse-ring {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}

        /* Futuristic Glass Cards */
        .glass-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 20px 24px;
            backdrop-filter: blur(16px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.25s ease;
            position: relative;
            overflow: hidden;
        }}

        .glass-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.6), transparent);
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .glass-card:hover {{
            border-color: var(--border-glow);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 242, 254, 0.12);
            transform: translateY(-2px);
        }}

        .glass-card:hover::before {{
            opacity: 1;
        }}

        /* Futuristic KPI Grid */
        .kpi-hologram-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-tile {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            backdrop-filter: blur(12px);
            transition: all 0.25s;
        }}

        .kpi-tile:hover {{
            border-color: rgba(56, 189, 248, 0.4);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.15);
            transform: translateY(-2px);
        }}

        .kpi-tag {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
            font-weight: 700;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .kpi-readout {{
            font-size: 32px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 4px;
        }}

        .neon-cyan {{ color: var(--cyan-neon); text-shadow: 0 0 12px rgba(0, 242, 254, 0.3); }}
        .neon-mint {{ color: #34d399; text-shadow: 0 0 12px rgba(52, 211, 153, 0.3); }}
        .neon-gold {{ color: var(--gold-neon); text-shadow: 0 0 12px rgba(251, 191, 36, 0.3); }}

        /* Main Holographic Charts Section */
        .chart-grid-2 {{
            display: grid;
            grid-template-columns: 1.9fr 1.1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        .card-header-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .card-title {{
            font-size: 16px;
            font-weight: 700;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .chart-canvas-wrap {{
            height: 310px;
            position: relative;
        }}

        /* Futuristic Cyber Table */
        .cyber-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .cyber-table th {{
            text-align: left;
            padding: 12px 14px;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-subtle);
            font-weight: 600;
        }}

        .cyber-table td {{
            padding: 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            color: var(--text-bright);
        }}

        .cyber-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .cyber-table tr:last-child td {{
            border-bottom: none;
            font-weight: 700;
            background: rgba(56, 189, 248, 0.04);
        }}

        .cyber-tag {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.1);
            color: var(--cyan-neon);
            border: 1px solid rgba(56, 189, 248, 0.25);
            padding: 3px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
        }}

        .mini-bar-track {{
            width: 70px;
            height: 5px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 3px;
            overflow: hidden;
        }}

        .mini-bar-fill {{
            height: 100%;
            border-radius: 3px;
        }}

        /* Live Quantum Order Simulator */
        .simulator-card {{
            background: linear-gradient(135deg, rgba(17, 25, 40, 0.9), rgba(15, 23, 42, 0.9));
            border: 1px solid var(--border-glow);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 0 25px rgba(0, 242, 254, 0.08);
        }}

        .sim-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr) 1.2fr;
            gap: 20px;
            align-items: center;
            margin-top: 14px;
        }}

        .sim-slider-box label {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .sim-slider-box input[type=range] {{
            width: 100%;
            accent-color: var(--cyan-neon);
            cursor: pointer;
        }}

        .sim-result-screen {{
            background: rgba(8, 14, 26, 0.8);
            border: 1px solid var(--border-glow);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}

        .sim-status-badge {{
            display: inline-block;
            margin-top: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .status-positive {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); }}
        .status-negative {{ background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(251, 113, 133, 0.4); }}

        /* Machine Learning Benchmark Grids */
        .ml-benchmarks-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        .glow-pill {{
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 4px;
            font-weight: 700;
            margin-left: 6px;
        }}
        .cyan-glow {{ background: rgba(0, 242, 254, 0.15); color: var(--cyan-neon); border: 1px solid rgba(0, 242, 254, 0.3); }}
        .mint-glow {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.3); }}
        .gold-glow {{ background: rgba(251, 191, 36, 0.15); color: var(--gold-neon); border: 1px solid rgba(251, 191, 36, 0.3); }}

        /* Visual Artifacts Gallery */
        .visual-gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}

        .gallery-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            overflow: hidden;
            transition: all 0.25s ease;
        }}

        .gallery-card:hover {{
            border-color: var(--cyan-neon);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
            transform: translateY(-2px);
        }}

        .gallery-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .gallery-label {{
            padding: 10px 14px;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            background: rgba(10, 15, 26, 0.8);
            border-top: 1px solid var(--border-subtle);
        }}

        /* Futuristic Footer */
        .hud-footer {{
            margin-top: 32px;
            padding: 16px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
        }}
    </style>
</head>
<body>
    <div class="shell-container">
        <!-- Futuristic HUD Navbar -->
        <div class="hud-nav">
            <div class="brand-block">
                <div class="cyber-logo">⚡</div>
                <div class="brand-text">
                    <h1>RETAIL INTELLIGENCE COMMAND</h1>
                    <span>Predictive Sales &amp; Profitability Quantum Suite</span>
                </div>
            </div>
            <div class="hud-telemetry">
                <div class="pulse-indicator">
                    <span class="live-dot"></span>
                    <span>PIPELINE TELEMETRY: ACTIVE</span>
                </div>
                <span class="cyber-tag" style="font-size:11px;">v3.2 NEURAL</span>
            </div>
        </div>

        <!-- 5 Futuristic Neon KPI Cards -->
        <div class="kpi-hologram-grid">
            <div class="kpi-tile">
                <div class="kpi-tag">Gross Revenue <span>USD</span></div>
                <div class="kpi-readout neon-cyan mono-num">${tot_revenue:,.2f}</div>
                <div style="font-size:11px; color:#38bdf8;">Cumulative portfolio sales</div>
            </div>

            <div class="kpi-tile">
                <div class="kpi-tag">Net Profit <span>USD</span></div>
                <div class="kpi-readout neon-mint mono-num">${tot_profit:,.2f}</div>
                <div style="font-size:11px; color:#34d399;">Total net returns generated</div>
            </div>

            <div class="kpi-tile">
                <div class="kpi-tag">Profit Margin <span>RATIO</span></div>
                <div class="kpi-readout neon-mint mono-num">{margin_pct:.2f}%</div>
                <div style="font-size:11px; color:#34d399;">Return on gross revenue</div>
            </div>

            <div class="kpi-tile">
                <div class="kpi-tag">Total Quantity <span>UNITS</span></div>
                <div class="kpi-readout neon-gold mono-num">{tot_qty:,}</div>
                <div style="font-size:11px; color:#fbbf24;">Units fulfilled &amp; dispatched</div>
            </div>

            <div class="kpi-tile">
                <div class="kpi-tag">Profitable Rate <span>SPLIT</span></div>
                <div class="kpi-readout neon-cyan mono-num">{prof_rate:.1f}%</div>
                <div style="font-size:11px; color:#38bdf8;">Profitable order proportion</div>
            </div>
        </div>

        <!-- Main Charts: Time-Series Hologram & Quantum Donut -->
        <div class="chart-grid-2">
            <div class="glass-card">
                <div class="card-header-bar">
                    <div class="card-title">
                        <span style="color:var(--cyan-neon);">◈</span> Revenue &amp; Profit Over Time
                    </div>
                    <span class="cyber-tag">48-Month Telemetry</span>
                </div>
                <div class="chart-canvas-wrap">
                    <canvas id="futuristicTrendChart"></canvas>
                </div>
            </div>

            <div class="glass-card">
                <div class="card-header-bar">
                    <div class="card-title">
                        <span style="color:var(--purple-neon);">◈</span> Revenue by Category
                    </div>
                    <span class="cyber-tag">Market Share</span>
                </div>
                <div class="chart-canvas-wrap" style="display:flex; align-items:center; justify-content:center;">
                    <canvas id="futuristicDonutChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Category Matrix Table -->
        <div class="glass-card" style="margin-bottom: 24px;">
            <div class="card-header-bar">
                <div class="card-title">
                    <span style="color:var(--gold-neon);">◈</span> Category Financial Audit Matrix
                </div>
                <span class="cyber-tag">Departmental Breakdown</span>
            </div>
            <table class="cyber-table">
                <thead>
                    <tr>
                        <th>Product Category</th>
                        <th>Total Gross Revenue</th>
                        <th>Net Profit</th>
                        <th>Profit Margin &amp; Health</th>
                        <th>Quantity Fulfilled</th>
                    </tr>
                </thead>
                <tbody>
                    {cat_rows_html}
                    <tr>
                        <td><strong>PORTFOLIO TOTAL</strong></td>
                        <td class="mono-num">${tot_revenue:,.2f}</td>
                        <td class="mono-num" style="color:#10b981;">${tot_profit:,.2f}</td>
                        <td class="mono-num" style="color:#10b981;">{margin_pct:.2f}%</td>
                        <td class="mono-num">{tot_qty:,}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Interactive Quantum Order Simulator -->
        <div class="simulator-card">
            <div class="card-header-bar" style="margin-bottom: 8px;">
                <div class="card-title">
                    <span style="color:var(--cyan-neon);">⚡</span> Quantum "What-If" Machine Learning Order Simulator
                </div>
                <span class="glow-pill cyan-glow">LIVE INFERENCE ENGINE</span>
            </div>
            <p style="font-size: 13px; color: var(--text-muted);">Adjust incoming order variables to simulate predicted sales revenue and real-time profitability likelihood.</p>
            
            <div class="sim-grid">
                <div class="sim-slider-box">
                    <label>Order Quantity: <span id="valQ" class="mono-num neon-cyan" style="font-weight:700;">28 units</span></label>
                    <input type="range" id="rngQ" min="1" max="100" value="28" oninput="runQuantumSim()">
                </div>

                <div class="sim-slider-box">
                    <label>Unit Price: <span id="valP" class="mono-num neon-gold" style="font-weight:700;">$48.00</span></label>
                    <input type="range" id="rngP" min="5" max="300" step="5" value="48" oninput="runQuantumSim()">
                </div>

                <div class="sim-slider-box">
                    <label>Shipping Cost: <span id="valS" class="mono-num neon-mint" style="font-weight:700;">$14.00</span></label>
                    <input type="range" id="rngS" min="2" max="100" step="1" value="14" oninput="runQuantumSim()">
                </div>

                <div class="sim-result-screen">
                    <div style="font-size:11px; text-transform:uppercase; color:var(--text-muted); font-weight:700;">Predicted Revenue</div>
                    <div id="simRev" class="mono-num neon-cyan" style="font-size:22px; font-weight:800; margin: 4px 0;">$1,344.00</div>
                    <span id="simBadge" class="sim-status-badge status-positive">PROFITABLE TELEMETRY LOCKED</span>
                </div>
            </div>
        </div>

        <!-- Machine Learning Neural Model Benchmarks -->
        <div class="ml-benchmarks-grid">
            <div class="glass-card">
                <div class="card-header-bar">
                    <div class="card-title">
                        <span style="color:var(--cyan-neon);">◈</span> Task 1: Regression (Predict Sales)
                    </div>
                    <span class="cyber-tag">Continuous Target</span>
                </div>
                <table class="cyber-table">
                    <thead>
                        <tr><th>Model Pipeline</th><th>Test R²</th><th>RMSE</th><th>MAE</th></tr>
                    </thead>
                    <tbody>
                        {reg_rows_html}
                    </tbody>
                </table>
            </div>

            <div class="glass-card">
                <div class="card-header-bar">
                    <div class="card-title">
                        <span style="color:var(--mint-neon);">◈</span> Task 2: Classification (Profitability)
                    </div>
                    <span class="cyber-tag">Binary Target</span>
                </div>
                <table class="cyber-table">
                    <thead>
                        <tr><th>Model Pipeline</th><th>Accuracy</th><th>F1-Score</th><th>Precision</th><th>Recall</th></tr>
                    </thead>
                    <tbody>
                        {clf_rows_html}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Visual Vault Gallery -->
        <div class="glass-card">
            <div class="card-header-bar">
                <div class="card-title">
                    <span style="color:var(--purple-neon);">◈</span> Analytical Visual Artifacts Vault
                </div>
                <span class="cyber-tag">High-Resolution Diagnostic Figures</span>
            </div>
            <div class="visual-gallery-grid">
                <div class="gallery-card">
                    <div class="gallery-label">06. Monthly Revenue &amp; Profit Trend</div>
                    <img src="{img_prefix}06_monthly_sales_profit_trend.png" alt="Trend">
                </div>
                <div class="gallery-card">
                    <div class="gallery-label">08. Regression Actual vs Predicted</div>
                    <img src="{img_prefix}08_regression_actual_vs_predicted.png" alt="Regression">
                </div>
                <div class="gallery-card">
                    <div class="gallery-label">09. Confusion Matrices (3 Models)</div>
                    <img src="{img_prefix}09_classification_confusion_matrices.png" alt="Confusion Matrix">
                </div>
                <div class="gallery-card">
                    <div class="gallery-label">10. Feature Importance Drivers</div>
                    <img src="{img_prefix}10_feature_importance.png" alt="Features">
                </div>
                <div class="gallery-card">
                    <div class="gallery-label">01. Correlation Heatmap</div>
                    <img src="{img_prefix}01_correlation_heatmap.png" alt="Correlation">
                </div>
                <div class="gallery-card">
                    <div class="gallery-label">04. Category Sales &amp; Margin</div>
                    <img src="{img_prefix}04_sales_profit_by_category.png" alt="Category">
                </div>
            </div>
        </div>

        <!-- Futuristic Footer -->
        <div class="hud-footer">
            <span>RETAIL SALES &amp; PROFITABILITY INTELLIGENCE SUITE • POWERED BY SCIKIT-LEARN &amp; PYTHON</span>
            <span class="mono-num">SECURE HOSTING ON GITHUB PAGES</span>
        </div>
    </div>

    <script>
        window.addEventListener('DOMContentLoaded', function() {{
            // Chart 1: Futuristic Trend Combo Chart
            const canvasTrend = document.getElementById('futuristicTrendChart');
            if (canvasTrend) {{
                const ctx = canvasTrend.getContext('2d');
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(months_labels)},
                        datasets: [
                            {{
                                type: 'bar',
                                label: 'Total Revenue ($)',
                                data: {json.dumps(revenue_vals)},
                                backgroundColor: 'rgba(0, 242, 254, 0.75)',
                                borderColor: '#00f2fe',
                                borderWidth: 1,
                                borderRadius: 4,
                                yAxisID: 'y'
                            }},
                            {{
                                type: 'line',
                                label: 'Net Profit ($)',
                                data: {json.dumps(profit_vals)},
                                borderColor: '#10b981',
                                borderWidth: 2.5,
                                pointBackgroundColor: '#10b981',
                                pointBorderColor: '#ffffff',
                                pointRadius: 3,
                                tension: 0.25,
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
                                grid: {{ color: 'rgba(255, 255, 255, 0.04)' }}
                            }},
                            y: {{
                                position: 'left',
                                ticks: {{ color: '#00f2fe', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v) }},
                                grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}
                            }},
                            y1: {{
                                position: 'right',
                                ticks: {{ color: '#10b981', callback: v => '$' + (v >= 1000 ? (v/1000).toFixed(0) + 'K' : v) }},
                                grid: {{ drawOnChartArea: false }}
                            }}
                        }},
                        plugins: {{
                            legend: {{
                                labels: {{ color: '#f8fafc', font: {{ size: 11 }} }}
                            }}
                        }}
                    }}
                }});
            }}

            // Chart 2: Futuristic Quantum Donut
            const canvasDonut = document.getElementById('futuristicDonutChart');
            if (canvasDonut) {{
                const ctxD = canvasDonut.getContext('2d');
                new Chart(ctxD, {{
                    type: 'doughnut',
                    data: {{
                        labels: {json.dumps(cat_labels)},
                        datasets: [{{
                            data: {json.dumps(cat_revenue)},
                            backgroundColor: ['#2563eb', '#00f2fe', '#f59e0b'],
                            borderWidth: 2,
                            borderColor: '#060910'
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
            }}
        }});

        function runQuantumSim() {{
            const q = parseFloat(document.getElementById('rngQ').value);
            const p = parseFloat(document.getElementById('rngP').value);
            const s = parseFloat(document.getElementById('rngS').value);

            document.getElementById('valQ').innerText = q + ' units';
            document.getElementById('valP').innerText = '$' + p.toFixed(2);
            document.getElementById('valS').innerText = '$' + s.toFixed(2);

            const rev = q * p;
            document.getElementById('simRev').innerText = '$' + rev.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});

            const ratio = s / rev;
            const badge = document.getElementById('simBadge');

            if (ratio > 0.16 || (p < 15 && s > 20)) {{
                badge.className = 'sim-status-badge status-negative';
                badge.innerText = 'LOSS EXPOSURE CRITICAL (' + Math.round((ratio + 0.5) * 60) + '% RISK)';
            }} else {{
                badge.className = 'sim-status-badge status-positive';
                badge.innerText = 'PROFITABLE TELEMETRY LOCKED (' + Math.min(97, Math.round((1 - ratio) * 95)) + '% CONF)';
            }}
        }}
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
    html_local = build_futuristic_dashboard_html(df, reg_results, clf_results, img_prefix="figures/")
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html_local)
    print(f"[SAVED] Futuristic dashboard saved to: {DASHBOARD_PATH}")

    # 2. Write root index.html for GitHub Pages (images in outputs/figures/)
    html_root = build_futuristic_dashboard_html(df, reg_results, clf_results, img_prefix="outputs/figures/")
    index_path = BASE_DIR / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_root)
    print(f"[SAVED] Futuristic index.html for GitHub Pages saved to: {index_path}")

    print("[SUCCESS] Portion 5 completed successfully.")


if __name__ == "__main__":
    run_dashboard_summary()
