# Deep Q-Learning vs. ARIMA for S&P 500 Trading

This project implements and compares two different strategies for stock trading on the S&P 500 index:
1.  **Deep Q-Network (DQN):** A reinforcement learning model that learns a trading policy based on historical price returns.
2.  **ARIMA Model:** A classical statistical time-series model used to predict price movements and execute trades based on a prediction threshold.

## Project Structure

*   [run.py](run.py): The main entry point for the analysis. It handles training, testing, and result saving for both models.
*   [config.py](config.py): Centralized configuration file for hyperparameters such as window sizes, learning rates, epochs, and ARIMA parameters.
*   [DQN_model.py](DQN_model.py): Implementation of the Deep Q-Network training and testing logic.
*   [ARIMA_model.py](ARIMA_model.py): Implementation of the ARIMA training (grid search for optimal hyperparameters) and testing logic.
*   [trading_env_class.py](trading_env_class.py): Defines the `stock` class, which simulates a trading environment including long/short positions and profit calculation.
*   [SP500.csv](SP500.csv): Historical price data for the S&P 500 index.
*   [results.ipynb](results.ipynb): Notebook for visualizing the results saved from the main analysis.

## How it Works

### DQN Strategy
The DQN model takes a window of price returns as its state and learns to choose between three actions: Buy, Sell, or Hold. The model is trained over multiple epochs to maximize total realized profit.

### ARIMA Strategy
The ARIMA model $(p, d, q)$ is used to forecast the next day's return. A grid search is performed during the "training" phase to find the best `window_size` and `threshold` (the predicted return required to trigger a trade) that maximizes profit on the training set.

## Usage

### 1. Prerequisites
Ensure you have the necessary Python libraries installed. Common requirements include:
- `numpy`
- `pandas`
- `statsmodels` (for ARIMA)
- `torch` (for DQN)
- `pickle`

### 2. Configuration
You can adjust parameters such as the train-test split ratio, DQN architecture, or ARIMA grid search values in [config.py](config.py).

### 3. Running the Analysis
To run the full training and testing pipeline, execute:
```bash
python run.py
```
This script will:
1. Load the data from [SP500.csv](SP500.csv).
2. Train the DQN model.
3. Perform a grid search to find the best ARIMA parameters.
4. Evaluate both models on the test (out-of-sample) data.
5. Save all results, model weights, and performance metrics to `dqn_trading_results.pkl`.

### 4. Visualizing Results
After running the analysis, you can open [results.ipynb](results.ipynb) to visualize the training progress, compare cumulative profits, and analyze the trading behavior of both strategies.

## Data Source
The data used is the historical Close prices of the S&P 500 index (`^GSPC`), provided in [SP500.csv](SP500.csv).
