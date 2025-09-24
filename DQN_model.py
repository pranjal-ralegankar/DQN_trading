# =========================
# Imports
# =========================
import numpy as np
import torch
import torch.nn as nn
import math
from config import dqn_layer1, dqn_layer2, learning_rate, eps_decay0, eps_decay1, Q_history_size, Q_update_epochs, Q_history_update_freq

# =========================
# Neural Network Class Definition
# =========================
class NNQ():
    def __init__(self, state_size):
        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(state_size, dqn_layer1), 
            nn.ReLU(), 
            nn.Linear(dqn_layer1, dqn_layer2), 
            nn.ReLU(), 
            nn.Linear(dqn_layer2, 3)  # Outputs for three actions
        )
        self.target_model = nn.Sequential(
            nn.Linear(state_size, dqn_layer1), 
            nn.ReLU(), 
            nn.Linear(dqn_layer1, dqn_layer2), 
            nn.ReLU(), 
            nn.Linear(dqn_layer2, 3)
        )
        self.target_model.load_state_dict(self.model.state_dict())  # Copy weights from self.model
        self.target_model.eval()  # Set the target model to evaluation mode
        self.optimizers = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        self.actions = np.array([-1, 0, 1])  # Actions: -1 = sell, 0 = hold, 1 = buy
        self.loss_table = []
    
    # =========================
    # Q-value Retrieval
    # =========================
    def get(self, s, a):
        i = np.where(self.actions == a)[0][0]
        s_torch = torch.tensor([s], dtype=torch.float32)
        return self.model(s_torch).detach().numpy().item(i)
    
    # =========================
    # Target Calculation
    # =========================
    def target(self, data, discount):
        ans = np.zeros(len(data))
        for i in range(len(data)):
            s, a, r, s_prime = data[i]
            s_torch = torch.tensor([s_prime], dtype=torch.float32)
            max_q = np.max(self.target_model(s_torch).detach().numpy())
            ans[i] = r if s_prime is None else r + discount * max_q
        return ans
    
    # =========================
    # Model Update
    # =========================
    def update(self, data, epochs):
        t = self.target(data, 0.9)
        self.model.train()
        for _ in range(epochs):
            avg_loss = 0  
            states = torch.tensor([s for (s, a, r, s_prime) in data], dtype=torch.float32)
            action_indices = [np.where(self.actions == a)[0][0] for (s, a, r, s_prime) in data]
            targets = torch.tensor(t, dtype=torch.float32)

            self.optimizers.zero_grad()
            predictions = self.model(states)[range(len(action_indices)), action_indices]
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizers.step()

            avg_loss = loss.item()
            self.loss_table.append(avg_loss)

        print("average loss=", avg_loss)

# =========================
# Epsilon-Greedy Action Selection
# =========================
def epsilon_greedy(q, s, eps):
    if np.random.rand() < eps:
        return np.random.choice([-1, 0, 1])
    else:
        q_values = [q.get(s, a) for a in [-1, 0, 1]]
        return np.argmax(q_values) - 1
    
# =========================
# Sigmoid Function
# =========================
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# =========================
# State Representation
# =========================
def getState(stk, t, n):    
    d = t - n + 1
    block = stk.all_prices[d:t + 1] if d >= 0 else np.concatenate((np.full(-d, stk.all_prices[0]), stk.all_prices[0:t + 1]))  # Pad with t0
    res = np.array([block[i + 1] / block[i] - 1 for i in range(n - 1)]) * 100  # Percent changes for last n-1 days
    ans = res
    return ans

# =========================
# Action Execution
# =========================
def execute_action(a, stk, date_index):
    date = stk.all_dates[date_index]
    date2 = stk.all_dates[date_index + 1]
    if a == 1:
        stk.buy(date, 1)  # Buy the share
        stk.sell(date2, 1)  # Sell immediately the next date
        r = stk.immediate_profit_percent  # Percent of profit made
    elif a == -1:
        stk.sell(date, 1)  # Short the share
        stk.buy(date2, 1)  # Buy immediately the next date
        r = stk.immediate_profit_percent  # Percent of profit made
    else:
        r = 0.1 #I reward for holding so the agent learns to buy or sell only if it thinks it would be very profitable.
    return r

# =========================
# Training the DQN Model
# =========================
from trading_env_class import stock
def train_DQN(ticker, window_size=30, epochs=20, Ntrain=1000):
    state_size = window_size  # State size for the model
    random_states = np.random.rand(1000, state_size) * 2 - 1  # Generate random states
    Q = NNQ(state_size)  # Initialize the Q-network

    # Initialize a dictionary to store Q-values for each epoch
    q_values_over_epochs = {action: [] for action in [-1, 0, 1]}
    for action in [-1, 0, 1]:
        q_values_over_epochs[action].append(
            [Q.get(state, action) for state in random_states]
        )
    
    profit_epochs = []
    for epoch in range(epochs):
        stock_history = []
        profit = []
        stk = stock(ticker)  # Reset the stock instance for each epoch
        for i in range(Ntrain - 1):
            s = getState(stk, i, window_size + 1)  # Get the state
            eps_decay = eps_decay0 + eps_decay1 * np.exp(-i / Ntrain * 1)  # Epsilon decay
            a = epsilon_greedy(Q, s, eps_decay)
            r = execute_action(a, stk, i)
            s_prime = getState(stk, i + 1, window_size + 1) if i + 1 < Ntrain else None

            stock_history.append((s, a, r, s_prime))
            if i % Q_history_update_freq == 0 and i > 0:
                if len(stock_history) > Q_history_size:
                    update_history = stock_history[-Q_history_size:]  # Take the last Q_histroy_size samples
                    Q.update(update_history, Q_update_epochs)
                else:
                    Q.update(stock_history, Q_update_epochs)
                print(i, " ", stk.realized_profit, " ", a, r)
            
            if epoch == epochs - 1:
                profit.append(stk.realized_profit)  # Store profit only for the last epoch

        stk.close(stk.all_dates[i + 1])
        profit_epochs.append(stk.realized_profit)
        print(f"Epoch {epoch + 1}/{epochs} - Realized profit: {stk.realized_profit}")
        
        # Update the target model after each epoch
        Q.target_model.load_state_dict(Q.model.state_dict())  # Copy weights from self.model
        Q.target_model.eval()  # Set the target model to evaluation mode

        # Compute Q-values for random states after this epoch
        for action in [-1, 0, 1]:
            q_values_over_epochs[action].append(
                [Q.get(state, action) for state in random_states]
            )
    
    return Q, profit, q_values_over_epochs, profit_epochs

# =========================
# Testing the DQN Model
# =========================
def test_DQN(SP_test, Q, window_size, Ntrain, Ndata):
    test_profit = []
    # Loop through the test dataset
    for i in range(Ntrain, Ndata):
        s = getState(SP_test, i, window_size + 1)  # Get the state
        a = epsilon_greedy(Q, s, 0.0)  # Use the trained model (epsilon = 0 for greedy policy)
        execute_action(a, SP_test, i)

        test_profit.append(SP_test.realized_profit)

    # Close any open positions at the end of the test period
    SP_test.close(SP_test.all_dates[-1])
    test_profit.append(SP_test.realized_profit)

    return test_profit