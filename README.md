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

#### Network Architecture
The reinforcement learning agent uses a Deep Q-Network with the following structure:
- **Input Layer:** Size depends on the `window_size` (number of past days' returns).
- **Hidden Layers:** Two fully connected layers with `ReLU` activations. The number of neurons (e.g., 128) is configurable in `config.py`.
- **Output Layer:** 3 neurons representing the Q-values for each action:
    - **Buy (`1`):** Purchase one share on the current day and close the position (Sell) the following day.
    - **Hold (`0`):** Take no action for the day.
    - **Sell/Short (`-1`):** Short-sell one share on the current day and close the position (Buy) the following day.

#### Training Details
- **Trading Logic:** The model operates on a daily horizon. Each trade is initiated at the market close of the current day and automatically closed at the market close of the very next trading day.
- **State Representation:** A vector of percentage price changes over the preceding `window_size` days.
- **Experience Replay:** The model stores a history of recent transitions $(s, a, r, s')$ and samples from this buffer to perform mini-batch updates every few iterations.
- **Target Network:** A separate target network is used to stabilize training by providing stationary Q-value targets. Its weights are synchronized with the primary behavior network at the end of each epoch.
- **Exploration Policy:** An epsilon-greedy strategy is employed where $\epsilon$ decays exponentially during training to shift from exploration to exploitation.
- **Reward Function:**
    - Taking a position (Buy or Sell) yields a reward equal to the percentage profit gained by closing that position on the following day.
    - Holding results in a small positive constant reward (e.g., `0.1`) to encourage the agent to stay in the market unless a clear profit opportunity is identified.
- **Optimizer:** Adam optimizer with Mean Squared Error (MSE) loss.

### ARIMA Strategy
The ARIMA model $(p, d, q)$ is used to forecast the next day's return. A grid search is performed during the "training" phase to find the best `window_size` and `threshold` (the predicted return required to trigger a trade) that maximizes profit on the training set.

- **Trading Logic:** Similar to the DQN model, the ARIMA strategy initiates a trade at the current day's close and closes it the next day.
    - If `predicted_return > threshold`: Buy today, Sell tomorrow.
    - If `predicted_return < -threshold`: Short today, Buy tomorrow.
    - Otherwise: No trade.

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
