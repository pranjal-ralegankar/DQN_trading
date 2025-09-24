# =========================
# Imports
# =========================
import numpy as np
import pandas as pd

# =========================
# Load S&P 500 data and prepare price history dictionary
# =========================
sp500_data = pd.read_csv('SP500.csv')
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])  # Convert Date column to datetime
price_history = {"^GSPC": sp500_data.set_index('Date')["Close"]}  # Set Date as the index for easy lookup

# =========================
# Stock class definition
# =========================
class stock:
    # -------------------------
    # Initialization
    # -------------------------
    def __init__(self, ticker):
        self.ticker = ticker
        self.history  = []
        self.long_position = 0
        self._long_history = np.zeros([0, 2]) #this keeps track of the amount of shares bought and at which prices. IF some shares are sold then they are removed from this list
        self.short_position=0
        self._short_history = np.zeros([0,2]) #the opposite of long_history
        self.invested=0 #this is the amount of money invested in the stock
        self.realized_profit = 0.0 #this is accumulated profit from all transactions
        self.immediate_profit=0.0 #this is profit from the last transaction
        self.immediate_profit_percent=0.0 #this is profit from the last transaction

        self.all_prices=price_history[self.ticker].values
        self.all_dates=price_history[self.ticker].index

    # -------------------------
    # Helper: Subtract from array for transaction history
    # -------------------------
    def _subtract_from_array(self, arr, X): #Used for removing shares from the history and calculating how much money was invested in those shares. Is useful for calculating profit of each transaction
        arr = np.array(arr, dtype=float)  # Ensure the array is a NumPy array and supports float operations
        cumulative_sum = np.cumsum(arr[:,0])  # Compute the cumulative sum of the array
        remaining = cumulative_sum - X   # Subtract X from the cumulative sum

        # print(arr,X)
        if remaining[-1]>0:
            # Find where the remaining amount becomes non-positive
            zeroed_indices = remaining <= 0
            #calculatenet money invested on the sold shares
            net_invested = arr[zeroed_indices,0].T@arr[zeroed_indices,1]
            # Update the array based on the remaining amount
            arr = arr[~zeroed_indices]
            net_invested+= (arr[0,0]-remaining[~zeroed_indices][0])*arr[0,1]
            arr[0,0] = remaining[~zeroed_indices][0]
        else:
            net_invested=arr[:,0].T@arr[:,1]
            arr=np.zeros([0, 2])
        # print(arr)
        return (arr,net_invested)
    
    # -------------------------
    # Buy shares (handles closing shorts)
    # -------------------------
    def buy(self, date, amount):
        self.history.append((price_history[self.ticker][date], amount, +1))        
        shorted=min(self.short_position, amount)
        bought=amount-shorted
        
        self.immediate_profit=0 # I reset this to 0 if no shares are shorted. So that when called from outside it tells no profit has been made or lost
        if shorted>0:
            self._short_history, net_borrowed = self._subtract_from_array(self._short_history, shorted) #remove the number of shorted shares from previous short history
            self.immediate_profit=net_borrowed - shorted * price_history[self.ticker][date]#profit from buying the shorted shares
            self.realized_profit += self.immediate_profit #accumulated profit
            self.short_position -= shorted
            self.immediate_profit_percent=self.immediate_profit/net_borrowed*100 if net_borrowed!=0 else 0

        self._long_history=np.append(self._long_history,np.array([[bought,price_history[self.ticker][date]]]), axis=0) #append the new buy to the long history
        self.long_position += bought

        self.invested += amount * price_history[self.ticker][date] #shorted shares also count towards investing because they cancel out the negative sum we invested in the begining
        
    
    # -------------------------
    # Sell shares (handles closing longs)
    # -------------------------
    def sell(self, date, amount):
        self.history.append((price_history[self.ticker][date], amount, -1))
        long_sold = min(self.long_position, amount)
        shorted = amount - long_sold

        self.immediate_profit=0 # I reset this to 0 if no shares are shorted. So that when called from outside it tells no profit has been made or lost
        if long_sold > 0:
            self._long_history, net_invested = self._subtract_from_array(self._long_history, long_sold)  # Remove the number of sold shares from previous long history
            self.immediate_profit= long_sold * price_history[self.ticker][date] - net_invested  # Profit from selling the long shares
            self.realized_profit += self.immediate_profit #accumulated profit
            self.long_position -= long_sold
            self.immediate_profit_percent=self.immediate_profit/net_invested*100 if net_invested!=0 else 0

        self._short_history = np.append(self._short_history, np.array([[shorted, price_history[self.ticker][date]]]), axis=0)  # Append the new short to the short history
        self.short_position += shorted

        self.invested -= amount * price_history[self.ticker][date] #invested amount can go negative if we are shorting as we are borrowing money.

    # -------------------------
    # Close all positions
    # -------------------------
    def close(self,date):
        if self.long_position > 0:
            self.sell(date, self.long_position)
        if self.short_position > 0:  
            self.buy(date, self.short_position)
        pass
    
    # -------------------------
    # Calculate unrealized profit
    # -------------------------
    def unrealized_profit(self, date):
        # Calculate unrealized profit based on the current price and the invested amount
        current_price = price_history[self.ticker][date]
        profit = self._long_history[:,0].T@(current_price-self._long_history[:,1])
        profit += self._short_history[:,0].T@(self._short_history[:,1]-current_price)
        
        return profit