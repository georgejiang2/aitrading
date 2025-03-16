import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace '^SPX' with the stock symbol you're interested in
symbol = '^SPX'

# Create a ticker object for the stock
ticker = yf.Ticker(symbol)

# Get available expiration dates
expiration_dates = ticker.options

# Ensure March 17th, 2025 is available in the expiration dates list
expiration_date = '2025-03-18'

def calc(s=symbol, ed=expiration_date):
    symbol = s
    expiration_date = ed
    matplotlib.use('Agg')
    if expiration_date in expiration_dates:
        # Get the options chain for the expiration date
        options_chain = ticker.option_chain(expiration_date)

        # List of columns to remove
        columns_to_remove = ['contractSymbol', 'lastTradeDate', 'inTheMoney', 'contractSize', 'currency', 'percentChange', 'change']

        # Drop the specified columns from calls and puts DataFrames
        calls_df = options_chain.calls.drop(columns=columns_to_remove)
        puts_df = options_chain.puts.drop(columns=columns_to_remove)

        # Get the current stock price
        current_price = ticker.history(period="1d")['Close'].iloc[-1]
        
        # Filter based on strike prices around current price
        x_margin = 250  # Margin to extend above and below the current price
        calls_df = calls_df[(calls_df['strike'] >= current_price - x_margin) & (calls_df['strike'] <= current_price + x_margin)]
        puts_df = puts_df[(puts_df['strike'] >= current_price - x_margin) & (puts_df['strike'] <= current_price + x_margin)]

        # Create bins with a width of 5
        min_strike = np.floor(min(calls_df['strike'].min(), puts_df['strike'].min()) / 5) * 5
        max_strike = np.ceil(max(calls_df['strike'].max(), puts_df['strike'].max()) / 5) * 5
        bin_edges = np.arange(min_strike, max_strike + 5, 5)  # +5 to include the upper bound
        
        # Plotting the bar chart
        plt.figure(figsize=(14, 8))

        # First plot the calls (in the background) with full width
        n_calls, bins, patches_calls = plt.hist(calls_df['strike'], 
                bins=bin_edges, 
                weights=calls_df['volume'], 
                label='Call',
                color='blue',
                edgecolor='black',
                alpha=1)

        # Calculate bin centers for plotting thinner put bars
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]  # Should be 5
        put_bar_width = bin_width * 0.6  # Make put bars 60% of the width of call bars
        
        # Create histogram data for puts manually
        put_hist, _ = np.histogram(puts_df['strike'], bins=bin_edges, weights=puts_df['volume'])
        
        # Then plot puts as centered, thinner bars
        plt.bar(bin_centers, put_hist, width=put_bar_width, color='darkorange', 
                edgecolor='black', alpha=1, label='Put')
                
        # Add a vertical line for the current price
        plt.axvline(x=current_price, color='green', linestyle='--', linewidth=2, 
                    label=f'Current Price: {current_price:.2f}')

        # Adding titles and labels
        plt.title(f"Options Volume by Strike Price for {symbol} Expiring on {expiration_date}", fontsize=16)
        plt.xlabel('Strike Price', fontsize=14)
        plt.ylabel('Volume', fontsize=14)
        plt.legend(title='Option Type', fontsize=12)
        
        # Set the x-axis limits to precise multiples of 5
        x_min = np.floor((current_price - x_margin) / 5) * 5
        x_max = np.ceil((current_price + x_margin) / 5) * 5
        plt.xlim(x_min, x_max)

        x_ticks = np.arange(x_min, x_max + 5, 15)
        plt.xticks(x_ticks)
        
        # Rotate tick labels for better readability
        plt.xticks(rotation=45)

        # Show the plot
        plt.tight_layout()
        image_path = os.path.join(os.path.dirname(__file__), 'images', 'test.png')
        plt.savefig(image_path, dpi=300)
    else:
        print(f"Expiration date {expiration_date} is not available for {symbol}.")


if __name__ == "__main__":
    calc()