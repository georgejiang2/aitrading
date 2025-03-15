import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from tqdm import tqdm  # Import tqdm for progress bars
from sklearn.metrics import mean_absolute_error, mean_squared_error

def parse_stock_data(file_path):
    # mostly for making sure if your data is formatted properly
    try:
        # Read the CSV, skipping the first two rows
        df = pd.read_csv(file_path)  

        # Convert relevant columns to numeric
        # essentially forcing these columns into numbers
        # already should be this way
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows with NaN values after conversion (if data was corrupted)
        # there shouldnt be any
        df.dropna(subset=numeric_columns, inplace=True)

        # Convert 'Datetime' column to datetime format if it exists
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)
        
        # Sort by datetime if available
        if 'Datetime' in df.columns or isinstance(df.index, pd.DatetimeIndex):
            df.sort_index(inplace=True)

        return df

    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        return None


# def prepare_features(df, days_lookback=5):
#     data = df.copy()

#     # aggregates data into one day
#     # ex: 8 1 hour data points into 1 1 day data point
#     if isinstance(data.index, pd.DatetimeIndex):
#         daily_data = data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
#     else:
#         daily_data = data
#     daily_data['Return'] = daily_data['Close'].pct_change()
#     daily_data['Range'] = daily_data['High'] - daily_data['Low']
#     daily_data['PrevClose'] = daily_data['Close'].shift(1)

#     for i in tqdm(range(1, days_lookback + 1), desc="Generating lag features"):
#         daily_data[f'Open_lag_{i}'] = daily_data['Open'].shift(i)
#         daily_data[f'Close_lag_{i}'] = daily_data['Close'].shift(i)
#         daily_data[f'High_lag_{i}'] = daily_data['High'].shift(i)
#         daily_data[f'Low_lag_{i}'] = daily_data['Low'].shift(i)
#         daily_data[f'Volume_lag_{i}'] = daily_data['Volume'].shift(i)
#         daily_data[f'Return_lag_{i}'] = daily_data['Return'].shift(i)
#         daily_data[f'Range_lag_{i}'] = daily_data['Range'].shift(i)

#     daily_data.dropna(inplace=True)
#     return daily_data

import pandas as pd
from tqdm import tqdm

def prepare_features(df, days_lookback):
    data = df.copy()

    # Aggregate intraday data into daily data
    if isinstance(data.index, pd.DatetimeIndex):
        daily_data = data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    else:
        daily_data = data
    
    daily_data['Return'] = daily_data['Close'].pct_change()
    daily_data['Range'] = daily_data['High'] - daily_data['Low']
    daily_data['PrevClose'] = daily_data['Close'].shift(1)
    
    # Generate lag features
    lag_features = []
    for i in tqdm(range(0, days_lookback), desc="Generating lag features"):
        lag_features.append(daily_data[['Open', 'Close', 'High', 'Low', 'Volume', 'Return', 'Range']].shift(i).add_suffix(f'_lag_{i}'))
    
    # Concatenate all lag features into daily_data
    daily_data = pd.concat([daily_data] + lag_features, axis=1)
    
    daily_data.dropna(inplace=True)
    return daily_data


def train_and_predict(features_df, days_to_predict=1):
    # X is now only lag 
    X = features_df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Range', 'PrevClose'], axis=1)
    print(X)
    y_open = features_df['Open']
    y_close = features_df['Close']
    y_high = features_df['High']
    y_low = features_df['Low']

    scaler_X = MinMaxScaler()
    scaler_open = MinMaxScaler()
    scaler_close = MinMaxScaler()
    scaler_low = MinMaxScaler()
    scaler_high = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_open_scaled = scaler_open.fit_transform(y_open.values.reshape(-1, 1))
    y_close_scaled = scaler_close.fit_transform(y_close.values.reshape(-1, 1))
    y_high_scaled = scaler_high.fit_transform(y_high.values.reshape(-1, 1))
    y_low_scaled = scaler_low.fit_transform(y_low.values.reshape(-1, 1))


    # Create weights: Newer dates get higher weights
    decay_factor = 0.98  # Adjust this to control the weighting decay
    num_samples = len(features_df)
    weights = np.array([decay_factor**(num_samples - i) for i in range(num_samples)])

    # Normalize weights to sum to 1 (optional)
    weights /= weights.sum()


    model_open = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)

#    for _ in tqdm(range(100), desc="Training Open Model"):
    model_open.fit(X_scaled, y_open_scaled.ravel(), sample_weight=weights)

#    for _ in tqdm(range(100), desc="Training Close Model"):
    model_close.fit(X_scaled, y_close_scaled.ravel(),sample_weight=weights)
    model_high.fit(X_scaled, y_high_scaled.ravel(),sample_weight=weights)
    model_low.fit(X_scaled, y_low_scaled.ravel(),sample_weight=weights)

    X_pred = X_scaled[-1:].reshape(1, -1)
    pred_open_scaled = model_open.predict(X_pred)
    pred_close_scaled = model_close.predict(X_pred)
    
    pred_high_scaled = model_high.predict(X_pred)
    pred_low_scaled = model_low.predict(X_pred)
    
    pred_open = scaler_open.inverse_transform(pred_open_scaled.reshape(-1, 1))
    pred_close = scaler_close.inverse_transform(pred_close_scaled.reshape(-1, 1))
    
    pred_high = scaler_open.inverse_transform(pred_high_scaled.reshape(-1, 1))
    pred_low = scaler_close.inverse_transform(pred_low_scaled.reshape(-1, 1))
    
    last_date = features_df.index[-1] if isinstance(features_df.index, pd.DatetimeIndex) else "Last Date"
    next_date = last_date + timedelta(days=days_to_predict) if isinstance(features_df.index, pd.DatetimeIndex) else "Next Day"
    
    prediction = {'Date': next_date, 'Predicted_Open': pred_open[0][0], 'Predicted_Close': pred_close[0][0], 'Predicted_High': pred_high[0][0], 'Predicted_Low': pred_low[0][0]}
    return prediction, model_open, model_close, X, scaler_X, scaler_open, scaler_close

def predict_stock_prices(file_path, output_path='prediction_results.csv'):
    df = parse_stock_data(file_path) # parse data
    if df is None:
        return "Failed to parse the input CSV file."
    
    # days look back is how many trading days you want to predict from
    # prepares df for training
    features_df = prepare_features(df, days_lookback=30)

    prediction, _, _, _, _, _, _ = train_and_predict(features_df)
    
    print(f"Prediction for {prediction['Date']}:\nOpening Price: ${prediction['Predicted_Open']:.2f}\nClosing Price: ${prediction['Predicted_Close']:.2f}\nHigh: ${prediction['Predicted_High']:.2f}\nLow: ${prediction['Predicted_Low']:.2f}")
    
    pd.DataFrame([prediction]).to_csv(output_path, index=False)
    print(f"Prediction saved to {output_path}")
    
    return prediction



if __name__ == "__main__":
    file_path = "^SPX_data.csv"
    #predict_stock_prices(file_path)
    df = parse_stock_data(file_path) # parse data
    backtest_model(df, train_size=0.8, days_lookback=30)

