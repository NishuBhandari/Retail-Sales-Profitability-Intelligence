"""
RETAIL SALES & PROFITABILITY INTELLIGENCE
Machine Learning project:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Regression: predict Sales
- Classification: predict Profitability (profitable / unprofitable)
- Model evaluation and saved artifacts

Dataset expected:
SalesDatafor preprocessing(maneet kaur).csv

Run:
    pip install -r requirements.txt
    python retail_sales_profitability.py

Important:
This dataset has Order Date, Order Priority, Order Quantity, Sales, Ship Mode,
Profit, Unit Price, Shipping Cost, Customer Name, Customer Segment,
Product Category, and company.
It does NOT have discount or region fields, so this project does not invent them.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from main import main

if __name__ == "__main__":
    main()
