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
![image](https://github.com/user-attachments/assets/582957ca-5594-493d-bf71-ffa352ae86d9)

### Financial Analysis:
![image](https://github.com/user-attachments/assets/5e760acf-a839-4f2b-8ae1-b5d8f40c1447)

![image](https://github.com/user-attachments/assets/b8180f72-f956-4d8c-8679-94d8a01811e3)

![image](https://github.com/user-attachments/assets/7fc25d6b-c570-47ae-839f-42c0d5339843)



### Price Prediction:
![image](https://github.com/user-attachments/assets/7d647925-b4ed-4958-b706-8ac90db083d7)

![image](https://github.com/user-attachments/assets/7403a6e2-4203-4149-8d1c-89a501cace66)

### Analysis & Recommendation:
![image](https://github.com/user-attachments/assets/7b391788-c801-427b-8688-c79edb12b2d4)

![image](https://github.com/user-attachments/assets/72eca7a4-237c-4d02-8bd4-ce6bf088eb6c)

![image](https://github.com/user-attachments/assets/de3b0f14-797b-4aa3-9d0e-7d3399322249)

![image](https://github.com/user-attachments/assets/6fc03809-f2ef-4e95-8b0c-b32cd45c19c1)

![image](https://github.com/user-attachments/assets/ce7729be-68e9-408b-baad-dab52ea75eed)




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
3. **Commit your changes.**
4. **Push to the branch and create a pull request.**
   
---
## License 📜
This project is licensed under the [MIT License](LICENSE). For more details, please refer to the LICENSE file.

## Acknowledgments 🙌
- **[Streamlit](https://streamlit.io/):** For app development.
- **[Yahoo Finance API](https://finance.yahoo.com/):** For stock data retrieval.
- **[TextBlob](https://textblob.readthedocs.io/):** For sentiment analysis.
- **[ChatGPT](https://openai.com/chatgpt):** For AI-driven insights and competitor analysis.


