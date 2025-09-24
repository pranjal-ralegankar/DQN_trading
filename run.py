# Import necessary libraries and modules
print("Importing libraries...")
from trading_env_class import stock
from DQN_model import train_DQN, test_DQN 
from ARIMA_model import train_arima, arima_profit
from config import train_fraction, arima_p, arima_q, epochs, window_size

# Initialize stock data for S&P 500
ticker = "^GSPC"
SP = stock(ticker)
Ndata = len(SP.all_dates) - 1  # Subtract 1 because data is price return and hence is 1 smaller than price history

# Test-train split
Ntrain = int(Ndata * train_fraction)

# Train DQN model and get results
print("Training DQN model...")
Q, profit, q_values_over_epochs, profit_epochs = train_DQN(ticker, window_size=window_size, epochs=epochs, Ntrain=Ntrain)

# Initialize a new stock instance for testing
SP_test = stock(ticker)
test_profit = test_DQN(SP_test, Q, window_size, Ntrain, Ndata)

# ARIMA training
print("Training ARIMA model...")
max_profit_arima, best_threshold, best_window_size = train_arima(ticker, Ntrain)

# ARIMA testing
SP_arima_test = stock(ticker)
arima_test_profit = arima_profit(SP_arima_test, Ntrain, Ndata, window_size=best_window_size, threshold=best_threshold, p=arima_p, q=arima_q)

# Save the results to a pickle file
import pickle
with open('dqn_trading_results.pkl', 'wb') as f:
    pickle.dump({
        'DQN_test_profit': test_profit,
        'DQN_training_profit': profit,
        'DQN_profit_epochs': profit_epochs,
        'q_values_over_epochs': q_values_over_epochs,
        'Qmodel': Q,
        'ARIMA_test_profit': arima_test_profit,
        'ARIMA_best_threshold': best_threshold,
        'ARIMA_best_window_size': best_window_size,
        'ARIMA_training_profit': max_profit_arima,
        'train_dates': SP.all_dates[:Ntrain],
        'test_dates': SP.all_dates[Ntrain:Ndata]
    }, f)