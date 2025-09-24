# =========================
# Import necessary libraries
# =========================
import numpy as np

# =========================
# Training configuration
# =========================
train_fraction = 0.7

# =========================
# DQN model parameters
# =========================
window_size = 30  # The number of previous days of price return to be used as state
epochs = 20  # Define the number of epochs for training

dqn_layer1 = 128  # Number of neurons in the first layer of the DQN
dqn_layer2 = 128  # Number of neurons in the second layer of the DQN
learning_rate = 0.001  # Learning rate for the DQN optimizer

Q_history_size = 30  # Number of past experiences to store for training the Q model
Q_update_epochs = 20  # Number of epochs to train the Q model when updating
Q_history_update_freq = 10  # Update the history of Q_learning every Q_history_update_freq data point later

# decay rates for epsilon in epsilon-greedy policy: eps_decay = eps_decay0 + eps_decay1 * np.exp(-t)
eps_decay0 = 0.1
eps_decay1 = 0.9

# =========================
# ARIMA model parameters
# =========================
arima_p = 1
arima_q = 1

# =========================
# Parameters for grid search in ARIMA model
# =========================
threshold_array = np.array([0.001, 0.003, 0.01, 0.03])
window_array = np.array([10, 20, 30, 50])
