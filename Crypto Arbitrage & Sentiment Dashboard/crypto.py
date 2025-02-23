import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# Initialize sentiment analyzer (singleton to avoid repeated initialization)
@st.experimental_singleton
def init_sentiment():
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

sid = init_sentiment()

# Cached function to fetch crypto prices with a TTL of 60 seconds
@st.experimental_memo(ttl=60)
def fetch_price(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Handle different JSON formats (Binance returns a 'price' key,
        # Coinbase returns data['data']['amount'])
        if 'price' in data:
            return float(data['price'])
        elif 'data' in data and 'amount' in data['data']:
            return float(data['data']['amount'])
        else:
            st.error("Unexpected JSON format.")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to select API endpoints based on user-selected cryptocurrency
def get_api_urls(pair):
    if pair == "BTC":
        binance = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        coinbase = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    elif pair == "ETH":
        binance = "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT"
        coinbase = "https://api.coinbase.com/v2/prices/ETH-USD/spot"
    else:
        binance = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        coinbase = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    return binance, coinbase

# Function to update price history DataFrame
def update_price_history(history, exchange, price):
    timestamp = datetime.now().strftime("%H:%M:%S")
    new_entry = pd.DataFrame({"Time": [timestamp], "Exchange": [exchange], "Price": [price]})
    if history is None:
        return new_entry
    else:
        return pd.concat([history, new_entry], ignore_index=True)

def main():
    st.title("Crypto Arbitrage & Sentiment Dashboard")

    # User selection for cryptocurrency pair
    crypto_choice = st.selectbox("Select Cryptocurrency", ["BTC", "ETH"])
    binance_url, coinbase_url = get_api_urls(crypto_choice)
    
    # Retrieve stored price history from session_state (if available)
    history_binance = st.session_state.get("history_binance", None)
    history_coinbase = st.session_state.get("history_coinbase", None)
    
    # Fetch current prices
    price_binance = fetch_price(binance_url)
    price_coinbase = fetch_price(coinbase_url)
    
    # Update price history with the new data
    history_binance = update_price_history(history_binance, "Binance", price_binance)
    history_coinbase = update_price_history(history_coinbase, "Coinbase", price_coinbase)
    st.session_state["history_binance"] = history_binance
    st.session_state["history_coinbase"] = history_coinbase

    # Display current prices in a table
    prices_df = pd.DataFrame({
        "Exchange": ["Binance", "Coinbase"],
        "Price": [price_binance, price_coinbase]
    })
    st.header("Live Cryptocurrency Prices")
    st.dataframe(prices_df)

    # Calculate arbitrage opportunity
    if price_binance is not None and price_coinbase is not None:
        price_diff = abs(price_binance - price_coinbase)
        percentage_diff = (price_diff / min(price_binance, price_coinbase)) * 100
        st.subheader("Arbitrage Opportunity")
        st.write(f"Price Difference: ${price_diff:.2f} ({percentage_diff:.2f}%)")
        
        # Let user set a threshold to trigger an arbitrage alert
        threshold = st.number_input("Set Arbitrage Alert Threshold (%)", value=1.0, step=0.1)
        if percentage_diff >= threshold:
            st.warning("Arbitrage Opportunity Detected!")

    # Plot historical price data using Plotly for interactive visualization
    st.subheader("Price History")
    combined_history = pd.concat([
        history_binance.assign(Exchange="Binance"),
        history_coinbase.assign(Exchange="Coinbase")
    ])
    fig = px.line(combined_history, x="Time", y="Price", color="Exchange", title="Price History")
    st.plotly_chart(fig)

    #  Analysis Section
    st.subheader("Market Sentiment")
    headline = st.text_input("Enter a headline for sentiment analysis", 
                               "Bitcoin rallies as investors anticipate new market trends")
    if headline:
        sentiment_score = sid.polarity_scores(headline)['compound']
        st.write(f"Sentiment Score: {sentiment_score}")
        if sentiment_score >= 0.05:
            st.success("Overall Positive Sentiment")
        elif sentiment_score <= -0.05:
            st.error("Overall Negative Sentiment")
        else:
            st.info("Neutral Sentiment")
    
    #re-runs every 60 seconds.
    auto_refresh = st.checkbox("Auto-refresh every 60 seconds")
    if auto_refresh:
        time.sleep(60)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
