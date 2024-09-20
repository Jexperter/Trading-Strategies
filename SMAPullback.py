import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.test import GOOG

class SMAPullback(Strategy):
    stop_loss = 0
    n1 = 20  # 20 SMA

    def init(self):
        price = self.data.Close
        self.sma = self.I(talib.SMA, price, self.n1)

    def next(self):
        price = self.data.Close[-1]
        low_price_prev_day = self.data.Low[-1]

        if price > self.sma[-1]:
            self.stop_loss = low_price_prev_day

            if price <= 2.4 * self.stop_loss:
                self.buy()

        


# Pull data
data = pd.read_csv(r'C:\Users\Jesann\Documents\trading\history data file\ETHUSDT_1d_1_Jan,_2023_to_18_Jul,_2024.csv', index_col=0, parse_dates=True)
data.columns = [column.capitalize() for column in data.columns]
bt = Backtest(data, SMAPullback, cash=10000, commission=.002)

#checks and optimizes which SMA is the best
output = bt.optimize(maximize="Equity Final [$]", n1=range(5,40,1))
print(output)
bt.plot()
