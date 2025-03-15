import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def parse_stock_data(file_path):
    # mostly for making sure if your data is formatted properly
    try:
        # Read the CSV, skipping the first two rows
        df = pd.read_csv(file_path)  

        # Convert relevant columns to numeric
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop any rows with NaN values after conversion (if data was corrupted)
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
def prepare_features(df, days_lookback, end_date=None):
    """
    Prepare features for the model, optionally using data only up to a specific end date.
    
    Parameters:
    df: DataFrame containing the stock data
    days_lookback: Number of previous days to use as features
    end_date: Optional cutoff date; if provided, only use data up to this date
    
    Returns:
    DataFrame with features prepared for training/prediction
    """
    data = df.copy()
    
    # If an end date is provided, filter the data
    if end_date is not None and isinstance(data.index, pd.DatetimeIndex):
        data = data[data.index <= end_date]
    
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
    for i in range(0, days_lookback):
        lag_features.append(daily_data[['Open', 'Close', 'High', 'Low', 'Volume', 'Return', 'Range']].shift(i).add_suffix(f'_lag_{i}'))
    
    # Concatenate all lag features into daily_data
    daily_data = pd.concat([daily_data] + lag_features, axis=1)
    
    # Drop rows with NaN values in the lag features
    daily_data.dropna(inplace=True)

    # Ensure the length is exactly `days_lookback + 1`
    if len(daily_data) > days_lookback:
        daily_data = daily_data[-days_lookback:]  # Keep only the last `days_lookback` rows

    return daily_data


def train_and_predict(features_df, days_to_predict=1):
    """Train models and make predictions for future days."""
    # X is now only lag features
    X = features_df.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Range', 'PrevClose'], axis=1)
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
    decay_factor = 0.94  # Adjust this to control the weighting decay
    num_samples = len(features_df)
    weights = np.array([decay_factor**(num_samples - i) for i in range(num_samples)])

    # Normalize weights to sum to 1
    weights /= weights.sum()

    model_open = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_high = RandomForestRegressor(n_estimators=100, random_state=42)
    model_low = RandomForestRegressor(n_estimators=100, random_state=42)

    model_open.fit(X_scaled, y_open_scaled.ravel(), sample_weight=weights)
    model_close.fit(X_scaled, y_close_scaled.ravel(), sample_weight=weights)
    model_high.fit(X_scaled, y_high_scaled.ravel(), sample_weight=weights)
    model_low.fit(X_scaled, y_low_scaled.ravel(), sample_weight=weights)

    X_pred = X_scaled[-1:].reshape(1, -1)
    pred_open_scaled = model_open.predict(X_pred)
    pred_close_scaled = model_close.predict(X_pred)
    pred_high_scaled = model_high.predict(X_pred)
    pred_low_scaled = model_low.predict(X_pred)
    
    pred_open = scaler_open.inverse_transform(pred_open_scaled.reshape(-1, 1))
    pred_close = scaler_close.inverse_transform(pred_close_scaled.reshape(-1, 1))
    pred_high = scaler_high.inverse_transform(pred_high_scaled.reshape(-1, 1))
    pred_low = scaler_low.inverse_transform(pred_low_scaled.reshape(-1, 1))
    
    last_date = features_df.index[-1] if isinstance(features_df.index, pd.DatetimeIndex) else "Last Date"
    next_date = last_date + timedelta(days=days_to_predict) if isinstance(features_df.index, pd.DatetimeIndex) else "Next Day"
    
    prediction = {'Date': next_date, 'Predicted_Open': pred_open[0][0], 'Predicted_Close': pred_close[0][0], 'Predicted_High': pred_high[0][0], 'Predicted_Low': pred_low[0][0]}
    return prediction, model_open, model_close, X, scaler_X, scaler_open, scaler_close

def calculate_metrics(predictions_df):
    """Calculate accuracy metrics for the predictions."""
    metrics = {}
    
    # Calculate metrics for each price type
    for price_type in ['Open', 'Close', 'High', 'Low']:
        actual = predictions_df[f'Actual_{price_type}']
        predicted = predictions_df[f'Predicted_{price_type}']
        
        # Error metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Direction accuracy (for open and close)
        if price_type in ['Open', 'Close']:
            actual_diffs = actual.diff().dropna()
            predicted_diffs = predicted[1:] - actual[:-1].values
            direction_accuracy = np.mean((actual_diffs > 0) == (predicted_diffs > 0)) * 100
        else:
            direction_accuracy = None
        
        metrics[price_type] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy
        }
    
    return metrics

def print_metrics(metrics):
    """Print formatted metrics."""
    print("\n=== MODEL PERFORMANCE METRICS ===")
    for price_type, metric_dict in metrics.items():
        print(f"\n{price_type} Price Predictions:")
        print(f"Mean Absolute Error (MAE): ${metric_dict['MAE']:.2f}")
        print(f"Root Mean Squared Error (RMSE): ${metric_dict['RMSE']:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metric_dict['MAPE']:.2f}%")
        print(f"R² Score: {metric_dict['R2']:.4f}")
        
        if metric_dict['Direction_Accuracy'] is not None:
            print(f"Direction Accuracy: {metric_dict['Direction_Accuracy']:.2f}%")

def plot_predictions(predictions_df, price_type='Close', output_path=None):
    """Plot actual vs predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df['Date'], predictions_df[f'Actual_{price_type}'], label=f'Actual {price_type}')
    plt.plot(predictions_df['Date'], predictions_df[f'Predicted_{price_type}'], label=f'Predicted {price_type}')
    plt.title(f'Actual vs Predicted {price_type} Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def rolling_window_backtest(df, min_train_size=60, days_lookback=14, step=1):
    """
    Perform backtesting using a rolling window approach.
    
    Parameters:
    df: DataFrame containing stock data
    min_train_size: Minimum number of days required for initial training
    days_lookback: Number of days to look back for feature generation
    step: Step size for the rolling window (1 = predict every day)
    
    Returns:
    DataFrame with predictions and actual values
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for rolling window backtesting")
    
    # First, we need to ensure we have enough data for the initial training
    # We need at least min_train_size + days_lookback days
    min_required_days = min_train_size + days_lookback
    if len(df) < min_required_days:
        raise ValueError(f"Not enough data for backtesting. Need at least {min_required_days} days.")
    
    # Get a list of unique dates to use for the rolling window
    dates = df.index.unique()
    dates = sorted(dates)
    
    # We'll start predicting from date index min_train_size onwards
    prediction_results = []
    
    print("Performing rolling window backtesting...")
    for i in tqdm(range(min_train_size, len(dates), step)):
        # Current date up to which we'll use data for training
        current_date = dates[i]
        
        # If we can't look ahead (which might happen at the end of the dataset), skip
        if i + 1 >= len(dates):
            continue
        
        # The next date is what we want to predict
        next_date = dates[i + 1]
        
        # Prepare features using only data up to the current date
        features_df = prepare_features(df, days_lookback=days_lookback, end_date=current_date)
        # print(len(features_df))
        # If we don't have enough data after feature preparation, skip
        if len(features_df) < days_lookback:  # Arbitrary small number to ensure we have enough data
            continue
        
        # Train model and make prediction
        prediction, _, _, _, _, _, _ = train_and_predict(features_df)
        
        # Get actual values for the next day
        # We need to handle the case where the next_date might not be in the dataset
        if next_date in df.index:
            next_day_data = df.loc[next_date]
            # If multiple records for the same date, take the first one
            if isinstance(next_day_data, pd.DataFrame):
                next_day_data = next_day_data.iloc[0]
            
            actual_open = next_day_data['Open']
            actual_close = next_day_data['Close']
            actual_high = next_day_data['High']
            actual_low = next_day_data['Low']
            
            # Store the prediction along with actual values
            result = {
                'Date': next_date,
                'Predicted_Open': prediction['Predicted_Open'],
                'Actual_Open': actual_open,
                'Predicted_Close': prediction['Predicted_Close'],
                'Actual_Close': actual_close,
                'Predicted_High': prediction['Predicted_High'],
                'Actual_High': actual_high,
                'Predicted_Low': prediction['Predicted_Low'],
                'Actual_Low': actual_low
            }
            prediction_results.append(result)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(prediction_results)
    
    # Calculate metrics
    if len(predictions_df) > 0:
        metrics = calculate_metrics(predictions_df)
        print_metrics(metrics)
        
        # Plot results
        for price_type in ['Open', 'Close', 'High', 'Low']:
            plot_path = f"rolling_{price_type.lower()}_predictions.png"
            plot_predictions(predictions_df, price_type, output_path=plot_path)
        
        # Save predictions to CSV
        predictions_path = "rolling_backtest_predictions.csv"
        predictions_df.to_csv(predictions_path)
        print(f"\nPredictions saved to {predictions_path}")
        
        return predictions_df, metrics
    else:
        print("No predictions were generated. Check your data and parameters.")
        return None, None

def predict_stock_prices(file_path, output_path='prediction_results.csv', backtest_first=True):
    """Parse data, train models, and make predictions."""
    df = parse_stock_data(file_path)
    if df is None:
        return "Failed to parse the input CSV file."
    
    # Backtest the model first to show accuracy
    if backtest_first:
        print("\n=== RUNNING ROLLING WINDOW BACKTESTING ===")
        backtest_df, metrics = rolling_window_backtest(df, min_train_size=60, days_lookback=30, step=1)
        
        if backtest_df is None:
            print("Backtesting failed. Proceeding with prediction anyway.")
    
    # Now prepare full dataset for final prediction
    print("\n=== MAKING PREDICTION FOR NEXT TRADING DAY ===")
    features_df = prepare_features(df, days_lookback=30)
    
    prediction, _, _, _, _, _, _ = train_and_predict(features_df)
    
    print(f"\nPrediction for {prediction['Date']}:")
    print(f"Opening Price: ${prediction['Predicted_Open']:.2f}")
    print(f"Closing Price: ${prediction['Predicted_Close']:.2f}")
    print(f"High: ${prediction['Predicted_High']:.2f}")
    print(f"Low: ${prediction['Predicted_Low']:.2f}")
    
    # Add confidence metrics to the prediction if we have them
    if backtest_first and 'metrics' in locals() and metrics is not None:
        for price_type in ['Open', 'Close', 'High', 'Low']:
            prediction[f'{price_type}_MAPE'] = metrics[price_type]['MAPE']
            if price_type in ['Open', 'Close']:
                prediction[f'{price_type}_Direction_Accuracy'] = metrics[price_type]['Direction_Accuracy']
    
    pd.DataFrame([prediction]).to_csv(output_path, index=False)
    print(f"Prediction saved to {output_path}")
    
    return prediction

if __name__ == "__main__":
    file_path = "^SPX_data.csv"
    predict_stock_prices(file_path, backtest_first=False)