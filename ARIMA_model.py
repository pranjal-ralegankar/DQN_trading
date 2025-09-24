# =========================
# Import necessary libraries
# =========================
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
from trading_env_class import stock
from config import threshold_array, window_array
# =========================

# =========================
# Load and prepare S&P 500 data
# =========================
sp500_data = pd.read_csv('SP500.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])  # Convert Date column to datetime
price_history = {"^GSPC": sp500_data.set_index('Date')["Close"]}  # Set Date as the index for easy lookup
# =========================

# =========================
# Calculate log of daily returns
# =========================
log_daily_return = np.log(price_history["^GSPC"] / price_history["^GSPC"].shift(1))
log_return_df = pd.DataFrame({
        'Date': price_history["^GSPC"].index[:],
        'Log_Daily_Return': log_daily_return[:]
    }).reset_index(drop=True)
log_return_df.dropna(inplace=True)  # Drop NaN values resulting from the shift operation
# =========================

# =========================
# Define ARIMA action function
# =========================
def arima_action(SP_arima, window_size, threshold, date_index, p=1, q=1):
    # Create a DataFrame with the date and log of daily returns for the training data only
    date = SP_arima.all_dates[date_index]
    window_mask = (log_return_df['Date'] <= date) & (log_return_df['Date'] > date - pd.Timedelta(days=window_size))
    windowed = log_return_df.loc[window_mask, ['Date', 'Log_Daily_Return']]
    # Set DatetimeIndex
    windowed_returns = windowed.set_index('Date')['Log_Daily_Return'].asfreq('B')

    # Skip if not enough data for ARIMA
    if len(windowed_returns) < 10:
        return None, None

    ar1 = ARIMA(windowed_returns, order=(p, 0, q)).fit()
    predicted_return = ar1.forecast(steps=1).values[0]

    # Calculate the actual return for the next day
    if date_index + 1 < len(SP_arima.all_dates):
        actual_return = log_daily_return.iloc[date_index + 1]
        date2 = SP_arima.all_dates[date_index + 1]
        if predicted_return > threshold:
            SP_arima.buy(date, 1)  # Buy the share    
            SP_arima.sell(date2, 1)  # Sell the next day
        elif predicted_return < -threshold:
            SP_arima.sell(date, 1)  # Short the share
            SP_arima.buy(date2, 1)  # Buy the next day

    return predicted_return, actual_return
# =========================

# =========================
# Define ARIMA profit calculation function
# =========================
def arima_profit(SP_arima, date_index_start, date_index_end, window_size, threshold, p=1, q=1):
    profit = []
    for i in range(date_index_start, date_index_end):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arima_action(SP_arima, window_size, threshold, i, p, q)
        profit.append(SP_arima.realized_profit)
    SP_arima.close(SP_arima.all_dates[i + 1])
    profit.append(SP_arima.realized_profit)
    return profit
# =========================

# =========================
# Define ARIMA training function
# =========================
def train_arima(ticker, Ntrain):
    max_profit = [0,]
    best_threshold = threshold_array[0]
    best_window_size = window_array[0]
    for threshold in threshold_array:
        for window_size in window_array:
            SP_arima = stock(ticker)  # Reset the stock instance for each parameter combination
            profit = arima_profit(SP_arima, 0, Ntrain, window_size=window_size, threshold=threshold, p=1, q=1)
            print(f"Threshold: {threshold}, Window Size: {window_size}, Profit: {profit}")
            if profit[-1] > max_profit[-1]:
                max_profit = profit
                best_threshold = threshold
                best_window_size = window_size

    return max_profit, best_threshold, best_window_size
# =========================

