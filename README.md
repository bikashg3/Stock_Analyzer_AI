# Stock Analysis and Prediction Application 📊

This repository hosts a Streamlit-based web application designed to help users analyze stock data, financials, technical indicators, sentiment analysis, and price predictions. The app provides actionable insights and recommendations for traders and investors.

---

## Features ✨

### 1. **Stock Overview**
- Displays the current price, price changes, and a 52-week range.
- Highlights undervaluation or overvaluation based on price proximity to the 52-week range.

### 2. **Company Financials**
- **Income Statement**, **Balance Sheet**, and **Cash Flow**:
  - Quarterly data with color-coded quarter-over-quarter changes.
  - Analysis of key financial health metrics like revenue, assets, and liabilities.

### 3. **Sentiment Analysis**
- Extracts news headlines and summaries from financial news sources.
- Computes sentiment polarity scores to gauge market sentiment toward a stock.

### 4. **Technical Analysis**
- Key indicators:
  - Relative Strength Index (RSI)
  - Simple Moving Averages (SMA-50 and SMA-200)
  - Bollinger Bands
- Interactive charts for detailed insights into stock trends and conditions.

### 5. **Price Prediction**
- LSTM-based advanced stock price prediction for the next 30 days.
- Provides historical predictions to verify the model's accuracy.

### 6. **Recommendations**
- Provides buy, sell, or hold recommendations based on:
  - RSI
  - Bollinger Bands
  - Analyst Ratings
  - Predicted Prices
  - 52-week range proximity

### 7. **Export and Analysis**
- Detailed JSON export of all data and analysis for offline use.
- Integrates web-scraped data from Screener.in to enrich company details.

---

## Installation 🛠️

1. Clone this repository:
   ```bash
   git clone https://github.com/bikashg3/Stock_Analyzer_AI.git
   cd Stock_Analyzer_AI
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py


## Usage 📋
1. **Open the app** in your browser.
2. **Enter the stock ticker symbol** (e.g., `RELIANCE.NS` for NSE stocks).
3. **Explore various sections** like:
   - Company financials
   - Sentiment analysis
   - Technical indicators
   - Predictions
4. **Download the analysis** as a JSON file for further use.

---

## Key Technologies 🛠️

### **Python Libraries**
- `streamlit`: For building the web app interface
- `yfinance`: For fetching stock market data
- `plotly`: For interactive visualizations
- `textblob`: For sentiment analysis
- `tensorflow`: For building LSTM models
- `neuralprophet`: For time-series forecasting
- `pmdarima`: For auto ARIMA modeling

### **Data Sources**
- **Yahoo Finance API**: For real-time stock data
- **Screener.in Web Scraping**: For detailed financial data

### **Machine Learning**
- **LSTM**: Used for price predictions
- **TextBlob**: Used for sentiment analysis

### **Visualization**
- **Plotly**: Interactive plots for an engaging user experience

---

## Screenshots 🌟

### Dashboard Overview:
*(Add a screenshot here)*

### Price Prediction:
*(Add a screenshot here)*

### Financial Analysis:
*(Add a screenshot here)*

---

## Future Enhancements 🚀
- Add **robust error handling** for data retrieval.
- Incorporate **additional financial metrics** and peer comparisons.
- Extend the **LSTM model** for multi-stock prediction.
- Enhance user experience with **more interactive features**.

---

## Contribution 🤝
Contributions are welcome! Follow these steps to contribute:

1. **Fork the repository.**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature-name

