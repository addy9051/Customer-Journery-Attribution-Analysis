# Customer Journey Attribution Analysis Platform

A comprehensive Streamlit-based application for analyzing customer journeys and multi-touch attribution using real-world e-commerce data.

## 🎯 Overview

This platform processes real e-commerce transaction data to understand how different marketing channels contribute to conversions. It implements multiple attribution models and provides interactive visualizations to help marketing teams optimize their channel strategies.

## ✨ Features

### 📊 Data Processing
- **Real-world data support**: Upload and process actual e-commerce transaction datasets
- **Intelligent data cleaning**: Automatic handling of missing values, duplicates, and data quality issues
- **Journey inference**: Creates realistic customer journey touchpoints from transaction data
- **Flexible file support**: CSV and Excel file formats

### 🎯 Attribution Models
- **First-Touch Attribution**: Credits the first touchpoint in the customer journey
- **Last-Touch Attribution**: Credits the last touchpoint before conversion
- **Linear Attribution**: Distributes credit equally across all touchpoints
- **Time-Decay Attribution**: Gives more weight to touchpoints closer to conversion
- **Position-Based Attribution**: Emphasizes first and last touchpoints (U-shaped)
- **Model Comparison**: Side-by-side analysis of all attribution models

### 📈 Visualizations & Analytics
- **Interactive dashboards**: Built with Plotly for rich, interactive charts
- **Journey path analysis**: Understand common customer journey patterns
- **Channel performance metrics**: Conversion rates, costs, and efficiency analysis
- **Attribution comparison charts**: Compare how different models value each channel
- **Customer journey flow**: Visual representation of touchpoint sequences

### 💡 Insights & Recommendations
- **Automated insights**: AI-generated observations about channel performance
- **Actionable recommendations**: Specific suggestions for marketing optimization
- **Cost efficiency analysis**: ROI and cost-per-conversion metrics
- **Journey optimization tips**: Recommendations for improving customer paths

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Streamlit
- pandas, numpy, plotly, scipy, networkx, scikit-learn

### Installation & Usage

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install streamlit pandas numpy plotly scipy networkx scikit-learn matplotlib
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py --server.port 5000
   ```

4. **Access the platform**: Open your browser to `http://localhost:5000`

### Data Requirements

Your e-commerce dataset should include:
- **CustomerID**: Unique identifier for each customer
- **InvoiceDate**: Timestamp of each transaction
- **Optional fields**: InvoiceNo, Quantity, UnitPrice, Description

**Supported formats**: CSV (.csv), Excel (.xlsx, .xls)

