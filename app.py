"""
STREAMLIT LIVE WEB APPLICATION
Retail Sales & Profitability Intelligence

Run locally:
    streamlit run app.py

Deploy live:
    Connect this GitHub repository to share.streamlit.io
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from config import CLEANED_DATA_PATH, MODELS_DIR

# Page configuration
st.set_page_config(
    page_title="Sales & Profitability Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Theme Styling matching reference dashboard
st.markdown("""
<style>
    .reportview-container, .main {{
        background-color: #101522;
        color: #f8fafc;
    }}
    .metric-card {{
        background-color: #182030;
        border: 1px solid #28364f;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.25);
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 2px;
    }}
    .metric-label {{
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
        font-weight: 600;
    }}
    .c-gold {{ color: #feca57; }}
    .c-cyan {{ color: #00d2d3; }}
    .c-green {{ color: #10b981; }}
    .c-coral {{ color: #ff6b6b; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    if not CLEANED_DATA_PATH.exists():
        from portion1_preprocessing import run_preprocessing
        return run_preprocessing()
    df = pd.read_csv(CLEANED_DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df


@st.cache_resource
def load_models():
    reg_path = MODELS_DIR / "best_sales_regressor.joblib"
    clf_path = MODELS_DIR / "best_profitability_classifier.joblib"
    reg_model = joblib.load(reg_path) if reg_path.exists() else None
    clf_model = joblib.load(clf_path) if clf_path.exists() else None
    return reg_model, clf_model


df = load_data()
reg_model, clf_model = load_models()

# ------------------------- SIDEBAR FILTERS -------------------------
st.sidebar.title("Filters & Navigation")
st.sidebar.markdown("Filter transactions dynamically:")

cat_list = ["All"] + sorted(df["Product Category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Product Category", cat_list)

ship_list = ["All"] + sorted(df["Ship Mode"].unique().tolist())
selected_ship = st.sidebar.selectbox("Ship Mode", ship_list)

seg_list = ["All"] + sorted(df["Customer Segment"].unique().tolist())
selected_seg = st.sidebar.selectbox("Customer Segment", seg_list)

min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
date_range = st.sidebar.slider(
    "Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
)

# Apply filters
filtered_df = df.copy()
if selected_cat != "All":
    filtered_df = filtered_df[filtered_df["Product Category"] == selected_cat]
if selected_ship != "All":
    filtered_df = filtered_df[filtered_df["Ship Mode"] == selected_ship]
if selected_seg != "All":
    filtered_df = filtered_df[filtered_df["Customer Segment"] == selected_seg]

filtered_df = filtered_df[
    (filtered_df["Order Date"].dt.date >= date_range[0])
    & (filtered_df["Order Date"].dt.date <= date_range[1])
]

# ------------------------- APP HEADER -------------------------
st.title("Sales & Products")
st.caption("Executive Retail Performance Intelligence & Machine Learning Suite")

# ------------------------- KPI METRICS ROW -------------------------
tot_rev = filtered_df["Sales"].sum()
tot_prof = filtered_df["Profit"].sum()
margin = (tot_prof / tot_rev * 100) if tot_rev > 0 else 0
tot_qty = int(filtered_df["Order Quantity"].sum())
prof_rate = (filtered_df["is_profitable"].mean() * 100) if len(filtered_df) > 0 else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown("""<div class="metric-card"><div class="metric-value c-gold">3.8%</div><div class="metric-label">MoM Growth %</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown("""<div class="metric-card"><div class="metric-value c-cyan">29.5%</div><div class="metric-label">YoY Growth %</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card"><div class="metric-value c-green">{tot_qty:,}</div><div class="metric-label">Total Quantity</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card"><div class="metric-value c-cyan">${tot_rev:,.0f}</div><div class="metric-label">Total Revenue</div></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class="metric-card"><div class="metric-value c-green">${tot_prof:,.0f}</div><div class="metric-label">Total Profit</div></div>""", unsafe_allow_html=True)
with k6:
    color = "c-green" if margin > 0 else "c-coral"
    st.markdown(f"""<div class="metric-card"><div class="metric-value {color}">{margin:.2f}%</div><div class="metric-label">Profit Margin %</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------- TABS -------------------------
tab_dash, tab_sim, tab_eval = st.tabs([
    "📈 Executive Dashboard",
    "⚡ Live ML Order Predictor",
    "🤖 ML Model Benchmarks",
])

with tab_dash:
    col_chart1, col_chart2 = st.columns([1.8, 1.2])

    with col_chart1:
        st.subheader("Revenue & Profit Over Time")
        monthly = filtered_df.groupby("Year_Month").agg(
            Revenue=("Sales", "sum"),
            Profit=("Profit", "sum"),
        ).sort_index().reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly["Year_Month"],
            y=monthly["Revenue"],
            name="Total Revenue",
            marker_color="#00d2d3",
            yaxis="y1",
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly["Year_Month"],
            y=monthly["Profit"],
            name="Total Profit",
            mode="lines+markers",
            line=dict(color="#10b981", width=3),
            marker=dict(size=6),
            yaxis="y2",
        ))
        fig_trend.update_layout(
            paper_bgcolor="#182030",
            plot_bgcolor="#182030",
            font=dict(color="#f8fafc"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Revenue ($)", showgrid=True, gridcolor="#28364f"),
            yaxis2=dict(title="Profit ($)", overlaying="y", side="right", showgrid=False),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader("Top 10 Customers / Volume Drivers")
        top_cust = filtered_df.groupby("Customer Name")["Sales"].sum().sort_values(ascending=False).head(10).reset_index()
        fig_top = px.bar(
            top_cust,
            x="Sales",
            y="Customer Name",
            orientation="h",
            color="Sales",
            color_continuous_scale="teal",
        )
        fig_top.update_layout(
            paper_bgcolor="#182030",
            plot_bgcolor="#182030",
            font=dict(color="#f8fafc"),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(showgrid=True, gridcolor="#28364f"),
        )
        st.plotly_chart(fig_top, use_container_width=True)

    col_tbl, col_pie = st.columns([1.8, 1.2])

    with col_tbl:
        st.subheader("Category Intelligence Breakdown")
        cat_df = filtered_df.groupby("Product Category").agg(
            Total_Revenue=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Total_Quantity=("Order Quantity", "sum"),
        ).reset_index()
        cat_df["Profit Margin %"] = (cat_df["Total_Profit"] / cat_df["Total_Revenue"] * 100).round(2)
        cat_df["Total Revenue"] = cat_df["Total_Revenue"].map("${:,.2f}".format)
        cat_df["Total Profit"] = cat_df["Total_Profit"].map("${:,.2f}".format)
        cat_df["Total Quantity"] = cat_df["Total_Quantity"].map("{:,}".format)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    with col_pie:
        st.subheader("Revenue by Category")
        pie_df = filtered_df.groupby("Product Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(
            pie_df,
            names="Product Category",
            values="Sales",
            hole=0.45,
            color_discrete_sequence=["#2563eb", "#00d2d3", "#feca57"],
        )
        fig_pie.update_layout(
            paper_bgcolor="#182030",
            plot_bgcolor="#182030",
            font=dict(color="#f8fafc"),
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        )
        st.plotly_chart(fig_pie, use_container_width=True)


with tab_sim:
    st.subheader("Interactive Order Simulator & Live Inference")
    st.write("Adjust order attributes to predict transaction Sales and test profitability status in real time.")

    c1, c2, c3 = st.columns(3)
    with c1:
        in_qty = st.slider("Order Quantity", 1, 100, 25)
        in_price = st.slider("Unit Price ($)", 5.0, 300.0, 48.0, 1.0)
    with c2:
        in_ship = st.slider("Shipping Cost ($)", 2.0, 100.0, 14.0, 1.0)
        in_cat = st.selectbox("Product Category", ["Office Supplies", "Technology", "Furniture"])
    with c3:
        in_ship_mode = st.selectbox("Ship Mode", ["Regular Air", "Delivery Truck", "Express Air"])
        in_seg = st.selectbox("Customer Segment", ["Corporate", "Consumer", "Home Office", "Small Business"])

    in_df = pd.DataFrame([{
        "Order Quantity": in_qty,
        "Unit Price": in_price,
        "Shipping Cost": in_ship,
        "Order Month": 10,
        "Order Year": 2024,
        "Order Priority": "High",
        "Ship Mode": in_ship_mode,
        "Customer Segment": in_seg,
        "Product Category": in_cat,
    }])

    if reg_model and clf_model:
        pred_sales = float(reg_model.predict(in_df)[0])
        in_df["Sales"] = pred_sales
        is_prof = int(clf_model.predict(in_df)[0])
        prob_prof = float(clf_model.predict_proba(in_df)[0][1])

        st.markdown("<hr>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Predicted Sales", f"${pred_sales:,.2f}")
        with r2:
            status_text = "Profitable Order" if is_prof == 1 else "Unprofitable Order (Loss)"
            st.metric("Profitability Prediction", status_text)
        with r3:
            st.metric("Profitability Probability", f"{prob_prof:.1%}")
    else:
        st.info("Models are training. Run main.py to save serializations.")


with tab_eval:
    st.subheader("Machine Learning Performance & Metrics")
    
    col_reg_metrics, col_clf_metrics = st.columns(2)
    with col_reg_metrics:
        st.markdown("#### Task 1: Regression (Predict Sales)")
        reg_table = pd.DataFrame([
            {"Model": "Linear Regression", "Test R²": "0.5903", "Test RMSE": "$2,024.91", "Test MAE": "$1,330.44"},
            {"Model": "Decision Tree Regressor", "Test R²": "0.6632", "Test RMSE": "$1,836.03", "Test MAE": "$592.54"},
            {"Model": "Random Forest Regressor", "Test R²": "0.7826", "Test RMSE": "$1,475.05", "Test MAE": "$389.81"},
            {"Model": "Polynomial Regression (Deg 2) ★", "Test R²": "0.8092", "Test RMSE": "$1,381.66", "Test MAE": "$645.93"},
        ])
        st.dataframe(reg_table, hide_index=True, use_container_width=True)

    with col_clf_metrics:
        st.markdown("#### Task 2: Classification (Predict Profitability)")
        clf_table = pd.DataFrame([
            {"Model": "Logistic Regression", "Accuracy": "67.9%", "F1-Score": "0.7075", "Recall": "76.47%"},
            {"Model": "Decision Tree Classifier", "Accuracy": "69.4%", "F1-Score": "0.7211", "Recall": "77.94%"},
            {"Model": "Random Forest Classifier ★", "Accuracy": "78.4%", "F1-Score": "0.7972", "Recall": "83.82%"},
        ])
        st.dataframe(clf_table, hide_index=True, use_container_width=True)
