import yfinance as yf
import pandas as pd


# Define the ticker symbol and the time period
ticker_symbol = "^SPX"  # Example: Apple Inc. You can change it to any other stock symbol
end_date = pd.to_datetime("today")


# Fetch data using yfinance with 15-minute intervals for the last 7 days
data = yf.download(ticker_symbol, period="500d", interval="1d")


# Save the data to a CSV file
data.to_csv(f"{ticker_symbol}_data.csv")


print(f"Data saved to {ticker_symbol}_data.csv")


