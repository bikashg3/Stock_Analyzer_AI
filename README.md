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


