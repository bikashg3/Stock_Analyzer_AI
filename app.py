import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from datetime import timedelta
import json
from pmdarima import auto_arima
from neuralprophet import NeuralProphet
import requests
from bs4 import BeautifulSoup
from g4f.client import Client
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import copy

def scrape_screener_complete(url, output_file="screener_full_data.json"):
    """
    Scrapes all data from Screener.in, ensuring all ranges-table elements and tables are fully captured.

    Args:
        url (str): The URL of the Screener.in company page.
        output_file (str): Path to save the scraped JSON data. Default is 'screener_full_data.json'.
    """
    try:
        # Fetch the webpage content
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Initialize the data dictionary
        data = {}

        # Extract company name
        company_name = soup.find('h1', class_='margin-0')
        if company_name:
            data['company_name'] = company_name.get_text(strip=True)

        # Extract all ranges-table data
        ranges_data = {}
        all_ranges_tables = soup.find_all('table', class_='ranges-table')
        for i, ranges_table in enumerate(all_ranges_tables, start=1):
            # Use <th> tag if available for table title, else use a counter
            table_title = ranges_table.find('th').get_text(strip=True) if ranges_table.find('th') else f"Range Table {i}"
            table_data = {}
            rows = ranges_table.find_all('tr')[1:]  # Skip the header row
            for row in rows:
                cells = row.find_all('td')
                if len(cells) == 2:
                    key = cells[0].get_text(strip=True).replace(':', '')
                    value = cells[1].get_text(strip=True)
                    table_data[key] = value
            if table_data:
                ranges_data[table_title] = table_data
        if ranges_data:
            data['ranges_data'] = ranges_data

        # Extract all other tables
        all_tables_data = {}
        tables = soup.find_all('table')
        for idx, table in enumerate(tables, start=1):
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                if len(cells) == len(headers):  # Match cells with headers
                    rows.append(dict(zip(headers, cells)))
            if rows:
                all_tables_data[f"table_{idx}"] = rows
        if all_tables_data:
            data['financial_tables'] = all_tables_data

        # # Extract key metrics
        # key_metrics = {}
        # metrics_sections = soup.find_all('li', class_='flex flex-space-between')
        # for section in metrics_sections:
        #     key = section.find('span').get_text(strip=True)
        #     value = section.find('span', class_='number').get_text(strip=True) if section.find('span', class_='number') else None
        #     if key and value:
        #         key_metrics[key] = value
        # if key_metrics:
        #     data['key_metrics'] = key_metrics

        key_metrics = {}
        metrics_sections = soup.find_all('li', class_='flex flex-space-between')

        for section in metrics_sections:
            key = section.find('span', class_='name').get_text(strip=True)  # Extract the name/span text
            value_span = section.find('span', class_='nowrap value')  # Locate the value container

            if value_span:
                # Special handling for multiple numbers in the 'nowrap' span
                numbers = value_span.find_all('span', class_='number')
                if len(numbers) == 2:
                    value = f"{numbers[0].get_text(strip=True)} / {numbers[1].get_text(strip=True)}"
                else:
                    value = numbers[0].get_text(strip=True) if numbers else None
            else:
                value = None

            if key and value:
                key_metrics[key] = value

        # Optional: Add the key metrics to your data dictionary
        if key_metrics:
            data['key_metrics'] = key_metrics

        # Save the data to a JSON file
        # with open(output_file, 'w', encoding='utf-8') as json_file:
        #     json.dump(data, json_file, ensure_ascii=False, indent=4)

        # print(f"Scraped data has been saved to {output_file}")
        return data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
#scraped_data = scrape_screener_complete("https://www.screener.in/company/ZOMATO/")



# -------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------

def styled_comment(text, color):
    """Renders a styled comment box in Streamlit."""
    st.markdown(
        f'<div style="background-color:{color}; padding:10px; border-radius:5px; font-size:16px;">{text}</div>',
        unsafe_allow_html=True,
    )

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20):
    sma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (2 * std)
    lower_band = sma - (2 * std)
    return sma, upper_band, lower_band


def extract_news_info(news_data):
    """
    Extract title and summary from a list of news articles.
    
    Args:
        news_data (list): List of dictionaries containing news articles
        
    Returns:
        list: List of dictionaries containing title and summary for each article
    """
    results = []
    
    for article in news_data:
        content = article['content']
        article_info = {
            'title': content['title'],
            'summary': content['summary']
        }
        results.append(article_info)
    
    return results

def analyze_sentiment(news):
    sentiments = []
    for article in news:
        sentiment = TextBlob(article).sentiment.polarity
        sentiments.append(sentiment)
    return np.mean(sentiments) if sentiments else 0



def predict_prices_advanced(data2, days=30):
    """
    Predict stock prices for the entire data range and the next `days` using an LSTM model.
    
    Parameters:
    - ticker (str): Stock ticker symbol.
    - days (int): Number of days to predict into the future.
    
    Returns:
    - historical_predictions (list): Predicted prices for historical data.
    - future_predictions (list): Predicted stock prices for the next `days`.
    - future_dates (list): Corresponding future dates for the predictions.
    - data (DataFrame): The historical data used for predictions.
    """
    # Fetch historical stock data
    #data = yf.download(ticker, period="5y", interval="1d")
    data = copy.deepcopy(data2)
    data.reset_index(inplace=True)
    close_prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Prepare data for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Build LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)  # Output layer for price prediction
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, batch_size=32, epochs=20, verbose=0)
    
    # Predict historical data
    historical_predictions = model.predict(X, verbose=0)
    historical_predictions = scaler.inverse_transform(historical_predictions).flatten()
    
    # Predict future prices
    future_predictions = []
    last_sequence = scaled_data[-sequence_length:]  # Start with the last sequence
    for _ in range(days):
        input_data = last_sequence.reshape((1, sequence_length, 1))
        predicted_price = model.predict(input_data, verbose=0)
        future_predictions.append(predicted_price[0, 0])
        # Update sequence with the predicted price
        last_sequence = np.append(last_sequence[1:], predicted_price, axis=0)
    
    # Inverse transform predictions back to original scale
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    
    # Generate future dates
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    return historical_predictions, future_predictions, future_dates, data2


# Example Sector/Industry Benchmarks (Mock Data)
BENCHMARKS = {
    "Technology": {
        "PE": 20,
        "EPS": 2.0,
        "RevenueGrowth": 0.10,
    },
    "Financial": {
        "PE": 15,
        "EPS": 1.5,
        "RevenueGrowth": 0.05,
    },
    "Energy": {
        "PE": 12,
        "EPS": 5.0,
        "RevenueGrowth": 0.03,
    },
}

def get_benchmarks(sector):
    return BENCHMARKS.get(sector, {"PE": 15, "EPS": 2.0, "RevenueGrowth": 0.05})

def highlight_financials(val):
    """Highlight negative values in red, positive in green (basic highlight)."""
    if pd.isnull(val):
        return ""
    try:
        val_float = float(val)
        if val_float > 0:
            color = 'lightgreen'
        else:
            color = 'tomato'
        return f'background-color: {color}'
    except:
        return ""


def highlight_qoq_changes(df, threshold_up=0.2, threshold_down=-0.2):
    """
    Color cells based on quarter-over-quarter percentage change,
    assuming row 0 is the most recent data and row n is the oldest.

    Green if soared >= +20%, Red if <= -20%.
    """
    # Create an empty style DataFrame to store background-color instructions
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Iterate from row=0 (most recent) down to the second-last row
    # We'll compare row r with row r+1 (which is 1 quarter older)
    for r in range(len(df) - 1):
        for c in range(len(df.columns)):
            current_val = df.iloc[r, c]
            prev_val = df.iloc[r+1, c]
            
            # Skip highlighting if any value is NaN or zero (to avoid division by zero)
            if pd.isnull(current_val) or pd.isnull(prev_val) or prev_val == 0:
                continue
            
            # Calculate the percentage change from previous quarter to current (most recent)
            pct_change = (current_val - prev_val) / abs(prev_val)
            
            # Apply highlighting based on thresholds
            if pct_change >= threshold_up:
                styles.iloc[r, c] = 'background-color: lightgreen'
            elif pct_change <= threshold_down:
                styles.iloc[r, c] = 'background-color: tomato'
    
    # Return a styled DataFrame using the 'styles' for background-color
    return df.style.apply(lambda col: styles[col.name], axis=0)


# -------------------------------------
# 2. STREAMLIT APP
# -------------------------------------

def main():
    st.title("Stock Analysis - Bikash 📊📈")
    st.caption("Analyze stock prices, metrics, financials, sentiment, and predictions with actionable recommendations.")

    stock_symbol = st.text_input("Enter the stock symbol (e.g., RELIANCE.NS for NSE stocks)")
    stock_symbol = stock_symbol.upper().strip()   

    if stock_symbol:
        # Collect all info for JSON export
        export_data = {
            "stock_symbol": stock_symbol,
            "company_info": {},
            "financials_analysis": {},
            "sentiment_analysis": {},
            "technical_analysis": {},
            "analyst_ratings": {},
            "prediction": {},
            "recommendation": {},
            "bot_analysis_suggestions": {}
        }

        # A) Fetch historical data
        try:
            stock_data = yf.Ticker(stock_symbol)
            stock_hist = stock_data.history(period="5y")
            if stock_hist.empty:
                st.error("No historical data found. Please check the symbol or try another.")
                return

            current_price = stock_hist['Close'].iloc[-1]
            previous_close = stock_hist['Close'].iloc[-2]
            price_change = current_price - previous_close
            percentage_change = (price_change / previous_close) * 100
            price_color = "green" if price_change > 0 else "red"

            st.header(f"Current Price of {stock_symbol}: {current_price:.2f} INR")
            st.markdown(
                f'<p style="color:{price_color};">Price Change: {price_change:.2f} INR ({percentage_change:.2f}%)</p>',
                unsafe_allow_html=True
            )

            export_data["company_info"]["current_price"] = current_price
            export_data["company_info"]["previous_close"] = previous_close
            export_data["company_info"]["price_change"] = price_change
            export_data["company_info"]["percentage_change"] = percentage_change

        except Exception as e:
            st.error(f"Failed to fetch stock data for {stock_symbol}. Error: {e}")
            return

        # B) Company Overview
        try:
            st.header("Company Overview")
            info = stock_data.info
            name = info.get('longName', 'N/A')
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            country = info.get('country', 'N/A')
            low_52w = info.get('fiftyTwoWeekLow', np.nan)
            high_52w = info.get('fiftyTwoWeekHigh', np.nan)
            eps = info.get("trailingEps", np.nan)
            pe_ratio = info.get("trailingPE", np.nan)

            st.write(f"**Name**: {name}")
            st.write(f"**Sector**: {sector}")
            st.write(f"**Industry**: {industry}")
            st.write(f"**Country**: {country}")
            st.write(f"**EPS**: {eps}")
            st.write(f"**P/E Ratio**: {pe_ratio}")
            st.write(f"**52 Week Range**: {low_52w} - {high_52w} INR")

            # 52 week range comment
            if pd.notnull(low_52w) and pd.notnull(high_52w):
                if current_price < (low_52w + (high_52w - low_52w) * 0.25):
                    styled_comment("Near 52-week low -> Potential undervaluation.", "red")
                elif current_price > (low_52w + (high_52w - low_52w) * 0.75):
                    styled_comment("Near 52-week high -> Potential overvaluation.", "red")
                else:
                    styled_comment("Trading in mid 52-week range.", "yellow")


            # Save to export data
            export_data["company_info"].update({
                "long_name": name,
                "sector": sector,
                "industry": industry,
                "country": country,
                "EPS": eps,
                "PE_ratio": pe_ratio,
                "52w_low": low_52w,
                "52w_high": high_52w,
            })

        except Exception as e:
            st.warning("Unable to fetch company overview.")
            st.error(f"Error: {e}")

        # C) Financial Statements
        st.header("Financial Statements")

        def process_financial_statement(df, statement_name):
            """
            1) Display with color-coded quarter-over-quarter changes
            2) Return a dictionary with the raw data
            """
            if df is None or df.empty:
                st.subheader(f"{statement_name}: No data available.")
                return {}

            # Convert index to string for easier display
            df_display = df.copy()
            df_display.index = df_display.index.astype(str)

            # Display with quarter-over-quarter color highlighting
            styled_df = highlight_qoq_changes(df_display)
            st.subheader(f"{statement_name} (Last 5 Years)")
            st.dataframe(styled_df, height=300)

            # Return dictionary for JSON export
            return df_display.to_dict()

        # Income Statement
        try:
            income_statement = stock_data.financials
            if income_statement is not None and not income_statement.empty:
                income_statement = income_statement.T
                income_statement_dict = process_financial_statement(income_statement, "Income Statement")

                # Additional example checks (e.g., total revenue sum)
                if "Total Revenue" in income_statement.columns:
                    total_revenue_series = income_statement["Total Revenue"]
                    if total_revenue_series.sum() > 0:
                        styled_comment("Sum of last 4 quarters revenue is positive.", "lightgreen")
                    else:
                        styled_comment("Total revenue is negative or missing. Investigate further.", "red")
            else:
                st.write("No income statement data available.")
                income_statement_dict = {}

            export_data["financials_analysis"]["income_statement"] = {
                "data": income_statement_dict
            }
        except Exception as e:
            st.warning("Unable to fetch income statement.")
            st.error(f"Error: {e}")

        # Balance Sheet
        try:
            balance_sheet = stock_data.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                balance_sheet = balance_sheet.T
                balance_sheet_dict = process_financial_statement(balance_sheet, "Balance Sheet")

                if ("Total Assets" in balance_sheet.columns and
                        "Total Liabilities Net Minority Interest" in balance_sheet.columns):
                    assets_sum = balance_sheet["Total Assets"].sum()
                    liab_sum = balance_sheet["Total Liabilities Net Minority Interest"].sum()
                    if assets_sum > liab_sum:
                        styled_comment("Assets exceed liabilities -> Good sign.", "lightgreen")
                    else:
                        styled_comment("Liabilities exceed assets -> Potential risk.", "red")
            else:
                st.write("No balance sheet data available.")
                balance_sheet_dict = {}

            export_data["financials_analysis"]["balance_sheet"] = {
                "data": balance_sheet_dict
            }
        except Exception as e:
            st.warning("Unable to fetch balance sheet.")
            st.error(f"Error: {e}")

        # Cash Flow Statement
        try:
            cash_flow = stock_data.cashflow
            if cash_flow is not None and not cash_flow.empty:
                cash_flow = cash_flow.T
                cash_flow_dict = process_financial_statement(cash_flow, "Cash Flow")

                if "Total Cash From Operating Activities" in cash_flow.columns:
                    cfo_sum = cash_flow["Total Cash From Operating Activities"].sum()
                    if cfo_sum > 0:
                        styled_comment("Positive sum of cash flow from operations. Good sign.", "lightgreen")
                    else:
                        styled_comment("Negative or zero sum of operating cash flow -> Potential issues.", "red")
            else:
                st.write("No cash flow data available.")
                cash_flow_dict = {}

            export_data["financials_analysis"]["cash_flow"] = {
                "data": cash_flow_dict
            }
        except Exception as e:
            st.warning("Unable to fetch cash flow statement.")
            st.error(f"Error: {e}")

        # D) News Sentiment
        try:
            st.header("News Sentiment Analysis")
            if stock_data.news:
                stock_data_news = extract_news_info(stock_data.news)
                headlines = [article['title'] for article in stock_data_news]
                summary = [article['summary'] for article in stock_data_news]
                #print(headlines)
                #st.write(f"**Headlines**: {headlines}")
                sentiment_score = analyze_sentiment(summary)
                st.write(f"**Sentiment Score**: {sentiment_score:.2f}")
                st.write("**Recent News**:")
                for k in range(min(len(headlines),5)):
                    st.write(f"- {headlines[k]}")
                    st.write(f"<i>{summary[k]}</i>", unsafe_allow_html=True)

                if sentiment_score > 0.1:
                    styled_comment("Positive sentiment indicates optimism.", "lightgreen")
                elif sentiment_score < -0.1:
                    styled_comment("Negative sentiment suggests concerns.", "red")
                else:
                    styled_comment("Neutral sentiment indicates no strong view.", "yellow")

                export_data["sentiment_analysis"]["score"] = sentiment_score
                export_data["sentiment_analysis"]["recent_headlines"] = headlines
            else:
                st.write("No news available for sentiment analysis.")
        except Exception as e:
            st.warning("Unable to fetch or analyze sentiment.")
            st.error(f"Error: {e}")

        # E) Technical Indicators
        try:
            st.header("Technical Indicators (Interactive Plot)")
            stock_hist['RSI'] = calculate_rsi(stock_hist)
            stock_hist['SMA_50'] = stock_hist['Close'].rolling(window=50).mean()
            stock_hist['SMA_200'] = stock_hist['Close'].rolling(window=200).mean()
            stock_hist['Bollinger_SMA'], stock_hist['Upper_Band'], stock_hist['Lower_Band'] = calculate_bollinger_bands(stock_hist)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Close'], 
                                     mode='lines', name='Close Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Upper_Band'],
                                     mode='lines', name='Upper Band', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['Lower_Band'],
                                     mode='lines', name='Lower Band', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['SMA_50'],
                                     mode='lines', name='SMA 50', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=stock_hist.index, y=stock_hist['SMA_200'],
                                     mode='lines', name='SMA 200', line=dict(color='green')))

            # Overbought/Oversold points
            overbought = stock_hist[stock_hist['RSI'] > 70]
            oversold = stock_hist[stock_hist['RSI'] < 30]
            fig.add_trace(go.Scatter(x=overbought.index, y=overbought['Close'],
                                     mode='markers', name='Overbought (RSI>70)',
                                     marker=dict(color='red', size=8)))
            fig.add_trace(go.Scatter(x=oversold.index, y=oversold['Close'],
                                     mode='markers', name='Oversold (RSI<30)',
                                     marker=dict(color='green', size=8)))

            fig.update_layout(
                title=f"{stock_symbol} Price Chart with Technical Indicators",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                legend_title="Indicators",
                template="plotly_white",
            )
            st.plotly_chart(fig)

            last_rsi = stock_hist['RSI'].iloc[-1]
            if last_rsi > 70:
                styled_comment("RSI > 70 = overbought condition.", "red")
            elif last_rsi < 30:
                styled_comment("RSI < 30 = oversold condition.", "lightgreen")
            else:
                styled_comment("RSI in neutral range.", "yellow")

            export_data["technical_analysis"]["latest_RSI"] = float(last_rsi)
        except Exception as e:
            st.warning("Unable to fetch technical indicators.")
            st.error(f"Error: {e}")

        # F) Analyst Ratings (Using the new approach with period == '0m')
        try:
            st.header("Analyst Ratings and Trends")
            analyst_data = stock_data.recommendations
            if analyst_data is not None and not analyst_data.empty:
                st.dataframe(analyst_data.tail(10))  # Show the most recent 10 ratings

                # Calculate the most recent ratings (current period, i.e., 0m)
                current_ratings = analyst_data.loc[analyst_data['period'] == '0m']

                if not current_ratings.empty:
                    # Extract rating counts
                    strong_buy = current_ratings['strongBuy'].values[0]
                    buy = current_ratings['buy'].values[0]
                    hold = current_ratings['hold'].values[0]
                    sell = current_ratings['sell'].values[0]
                    strong_sell = current_ratings['strongSell'].values[0]

                    # Generate comments based on the ratings
                    total_ratings = strong_buy + buy + hold + sell + strong_sell
                    buy_ratio = (strong_buy + buy) / total_ratings * 100 if total_ratings else 0
                    sell_ratio = (sell + strong_sell) / total_ratings * 100 if total_ratings else 0

                    if buy_ratio > 70:
                        styled_comment(
                            f"Analysts are overwhelmingly positive about this stock with {buy_ratio:.2f}% of ratings being 'Buy' or 'Strong Buy'.",
                            "lightgreen"
                        )
                    elif 50 <= buy_ratio <= 70:
                        styled_comment(
                            f"Analysts generally favor buying this stock with {buy_ratio:.2f}% of ratings being 'Buy' or 'Strong Buy'.",
                            "lightgreen"
                        )
                    elif sell_ratio > 50:
                        styled_comment(
                            f"Analysts recommend caution with {sell_ratio:.2f}% of ratings being 'Sell' or 'Strong Sell'.",
                            "red"
                        )
                    elif hold > max(buy, sell):
                        styled_comment(
                            f"Most analysts recommend holding this stock, with {hold} 'Hold' ratings.",
                            "yellow"
                        )
                    else:
                        styled_comment("Analyst recommendations are mixed. Review other metrics before making a decision.", "yellow")
                    
                    # Save to export
                    export_data["analyst_ratings"].update({
                        "strong_buy": int(strong_buy),
                        "buy": int(buy),
                        "hold": int(hold),
                        "sell": int(sell),
                        "strong_sell": int(strong_sell),
                    })
                else:
                    st.write("No current analyst ratings available for analysis.")
        except Exception as e:
            st.warning("Unable to process analyst ratings.")
            st.error(f"Error: {e}")

        # G) Future Price Predictions
        try:
            st.header("Future Price Predictions (Interactive Plot)")
            #predictions, future_dates = predict_prices(stock_hist)
            historical_predictions, predictions, future_dates, stock_hist = predict_prices_advanced(stock_hist)

            #future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
            # Create future dataframe
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
            st.dataframe(future_df)

            pred_fig = go.Figure()
            pred_fig.add_trace(go.Scatter(
                x=stock_hist.index,
                y=stock_hist['Close'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue')
            ))
            pred_fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines',
                name='Predicted Prices',
                line=dict(color='orange', dash='dash')
            ))

            # Historical predictions
            pred_fig.add_trace(go.Scatter(
                x=stock_hist.index[60:],  # Exclude the first 60 days since predictions start after this
                y=historical_predictions,
                mode='lines',
                name='Predicted (Historical)',
                line=dict(color='green')
            ))

            pred_fig.update_layout(
                title=f"{stock_symbol} Price Prediction (Next 30 Days)",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                legend_title="Data",
                template="plotly_white"
            )
            st.plotly_chart(pred_fig)

            if predictions[-1] > current_price:
                styled_comment("Model suggests potential price increase.", "lightgreen")
            else:
                styled_comment("Model suggests potential price decline.", "red")

            export_data["prediction"]["next_30_days"] = list(map(float, predictions))
            export_data["prediction"]["dates"] = [str(d) for d in future_dates]
        except Exception as e:
            st.warning("Unable to predict future prices.")
            st.error(f"Error: {e}")

        # H) Buy/Sell Recommendation
        try:
            st.header("Buy/Sell Recommendations")
            weights = {
                "RSI": 0.25,
                "Bollinger_Bands": 0.25,
                "Predictions": 0.1,
                "52_Week_Range": 0.25,
                "Analyst_Ratings": 0.15,
            }
            recommendation_score = 0
            positive_factors = []
            negative_factors = []

            # RSI
            last_rsi = stock_hist['RSI'].iloc[-1]
            if last_rsi > 70:
                recommendation_score -= weights["RSI"]
                negative_factors.append("Overbought (RSI>70)")
            elif last_rsi < 30:
                recommendation_score += weights["RSI"]
                positive_factors.append("Oversold (RSI<30)")

            # Bollinger
            if ('Upper_Band' in stock_hist and 'Lower_Band' in stock_hist):
                if current_price > stock_hist['Upper_Band'].iloc[-1]:
                    recommendation_score -= weights["Bollinger_Bands"]
                    negative_factors.append("Above upper Bollinger Band -> Overbought")
                elif current_price < stock_hist['Lower_Band'].iloc[-1]:
                    recommendation_score += weights["Bollinger_Bands"]
                    positive_factors.append("Below lower Bollinger Band -> Oversold")

            # Predictions
            if predictions[-1] > current_price:
                recommendation_score += weights["Predictions"]
                positive_factors.append("Positive future price prediction")
            else:
                recommendation_score -= weights["Predictions"]
                negative_factors.append("Negative future price prediction")

            # 52w range
            if pd.notnull(low_52w) and pd.notnull(high_52w):
                if current_price < (low_52w + (high_52w - low_52w) * 0.25):
                    recommendation_score += weights["52_Week_Range"]
                    positive_factors.append("Near 52-week low (undervalued)")
                elif current_price > (low_52w + (high_52w - low_52w) * 0.75):
                    recommendation_score -= weights["52_Week_Range"]
                    negative_factors.append("Near 52-week high (overvalued)")

            # Analyst Ratings
            try:
                analyst_data = stock_data.recommendations
                if analyst_data is not None and not analyst_data.empty:
                    cr = analyst_data.loc[analyst_data['period'] == '0m']
                    if not cr.empty:
                        sb = cr['strongBuy'].values[0]
                        b = cr['buy'].values[0]
                        h = cr['hold'].values[0]
                        s = cr['sell'].values[0]
                        ss = cr['strongSell'].values[0]
                        total = sb + b + h + s + ss
                        if total > 0:
                            buy_ratio = (sb + b) / total * 100
                            if buy_ratio > 70:
                                recommendation_score += weights["Analyst_Ratings"]
                                positive_factors.append(f"High Buy Ratio ({buy_ratio:.1f}%)")
                            elif buy_ratio < 30:
                                recommendation_score -= weights["Analyst_Ratings"]
                                negative_factors.append(f"High Sell Ratio ({100 - buy_ratio:.1f}%)")
            except:
                pass

            if recommendation_score > 0.5:
                final_recommendation = "Strong Buy"
                color = "lightgreen"
            elif 0 < recommendation_score <= 0.5:
                final_recommendation = "Buy"
                color = "lightgreen"
            elif -0.5 <= recommendation_score <= 0:
                final_recommendation = "Hold"
                color = "yellow"
            else:
                final_recommendation = "Sell"
                color = "red"

            styled_comment(f"**Recommendation**: <b>{final_recommendation}</b> (Score: {recommendation_score:.2f})", color)

            st.subheader("Factors Contributing Positively")
            if positive_factors:
                for pf in positive_factors:
                    styled_comment(f"+ {pf}", "lightgreen")
            else:
                st.write("No strong positive factors identified.")

            st.subheader("Factors Contributing Negatively")
            if negative_factors:
                for nf in negative_factors:
                    styled_comment(f"- {nf}", "red")
            else:
                st.write("No strong negative factors identified.")

            export_data["recommendation"] = {
                "recommendation_score": recommendation_score,
                "verdict": final_recommendation,
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
            }

        except Exception as e:
            st.warning("Unable to generate recommendations.")
            st.error(f"Error: {e}")

        # ---------------------------------
        # I) Additional Bot Analysis Suggestions
        # ---------------------------------
        #st.header("Additional Bot Analysis Suggestions")

        # Example of how you might populate this with a final summary
        # We'll do a minimal example, but you can customize the logic to your liking.
        bot_analysis = {}

        bot_analysis["Key Highlights"] = {
            "Current Price": f"₹{current_price:.2f}",
            "52-Week Range": f"₹{low_52w} - ₹{high_52w}" if pd.notnull(low_52w) and pd.notnull(high_52w) else "N/A",
            "PE Ratio": f"{pe_ratio:.2f}" if pd.notnull(pe_ratio) else "N/A",
            "EPS": f"₹{eps:.2f}" if pd.notnull(eps) else "N/A",
            "Sector": sector,
            "Industry": industry,
        }


        export_data['screener_webscrapped_data'] = scrape_screener_complete(f"https://www.screener.in/company/{stock_symbol[ 0 : stock_symbol.index('.')]}/")

        client = Client()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an Indian Stock Market Expert assistant that analyzes stock data to help analyze the given stock and share complete recommendations."},
                {"role": "user", "content": f"Analyze the following stock as an expert and share complete trading recommendations along with its financial health in detail explaining terms as well (also highlight main pointers from screener_webscrapped_data).\n Also, add more detailed peer comparison (very detailed comparison using bullet points) or risk-adjusted portfolio suggestions:\n\n{export_data}"}
            ],
            #messages=[{"role": "user", "content": f"Please help analyze the given stock and share complete recommendations acting as an indian stock market expert. \n {data}"}],
            web_search = True
        )
        st.header("Analysis by ChatGPT")
        st.write(response.choices[0].message.content)

        # ---------------------------------
        # J) Export All to JSON
        # ---------------------------------
        st.header("Export Analysis to JSON")
        st.write("You can download the entire analysis as a JSON file and use it elsewhere (e.g. ChatGPT).")

        # Convert everything to JSON (stringify data properly)
        json_data = json.dumps(export_data, indent=4, default=str)

        st.download_button(
            label="Download Analysis JSON",
            data=json_data,
            file_name=f"{stock_symbol}_analysis.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
