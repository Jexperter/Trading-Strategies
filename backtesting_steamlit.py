import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from binance.client import Client
import datetime
import streamlit as st
import time

# Initialize Binance Client (use your API key and secret)
api_key = os.getenv('API_KEY_BINANCE')
api_secret = os.getenv('API_SECRET_BINANCE')
client = Client(api_key, api_secret)

TIMEOUT_DURATION = 300  # 5 minutes in seconds

# Initialize session state for timeout tracking
if 'last_active' not in st.session_state:
    st.session_state.last_active = time.time()


# Function to check for inactivity timeout
def check_timeout():
    if time.time() - st.session_state.last_active > TIMEOUT_DURATION:
        st.warning("You have been inactive for 5 minutes. The app will stop shortly.")
        time.sleep(5)  # Wait for 5 seconds before stopping
        st.stop()

# Update last active time after any user input
def update_last_active():
    st.session_state.last_active = time.time()

# Fetch OHLCV data from Binance
def fetch_ohlcv_data(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    # Convert data to DataFrame
    ohlcv_data = pd.DataFrame(klines, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
        'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
        'Taker Buy Quote Asset Volume', 'Ignore'
    ])
    
    # Convert to proper datetime and numeric types
    ohlcv_data['Open Time'] = pd.to_datetime(ohlcv_data['Open Time'], unit='ms')
    ohlcv_data['Open Time'] = ohlcv_data['Open Time'].dt.floor('T')  # Floor to minute, ignoring seconds
    ohlcv_data.set_index('Open Time', inplace=True)
    ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']] = ohlcv_data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    
    return ohlcv_data

# Main logic to process liquidation data and check conditions
def backtest_all_trades(start_date, end_date, timeframe, starting_equity):
    # Load liquidation data
    liquidation_data = pd.read_csv(
        r'C:\Users\Jesann\Documents\trading\history data file\liq data\BTCUSDT (02.02.2023 to 13.03.2023) liq data.csv',
        parse_dates=['Datetime'],
        dayfirst=True  # Ensure day first for European date format
    )
    
    # Convert start_date and end_date to datetime objects at midnight
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # End of the day

    # Filter liquidation data based on provided dates
    liquidation_data = liquidation_data[
        (liquidation_data['Datetime'] >= start_datetime) & (liquidation_data['Datetime'] <= end_datetime)
    ]

    
    # Floor liquidation time to minutes (ignoring seconds)
    liquidation_data['Datetime'] = liquidation_data['Datetime'].dt.floor('T')

    # Fetch OHLCV data from Binance for the specified period and timeframe
    ohlcv_data = fetch_ohlcv_data('BTCUSDT', timeframe, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Initialize equity
    equity = starting_equity
    risk_per_trade = 0.05 * starting_equity  # Risking 5% of account per trade
    trading_fee_rate = 0.002

    # Initialize statistics variables
    num_trades = 0
    num_wins = 0
    num_losses = 0
    equity_history = [starting_equity]
    all_results = []

    # Iterate over each unique date
    unique_dates = liquidation_data['Datetime'].dt.date.unique()

    for trade_date in unique_dates:
        liquidation_data_filtered = liquidation_data[
            liquidation_data['Datetime'].dt.date == trade_date
        ]
        
        open_trade = None

        for index, liquidation in liquidation_data_filtered.iterrows():
            liquidation_time = liquidation['Datetime'].floor('T')

            if liquidation_time in ohlcv_data.index:
                liquidation_candle = ohlcv_data.loc[liquidation_time]
                next_candle_time = liquidation_time + pd.DateOffset(minutes=1)

                if open_trade:
                    if (liquidation['Order Type'] != open_trade['Order Type']):
                        open_trade['Exit Time'] = liquidation_time
                        open_trade['Exit Price'] = ohlcv_data.loc[liquidation_time]['Close']
                        open_trade['Duration'] = (open_trade['Exit Time'] - open_trade['Entry Time']).seconds / 60.0
                        
                        # Calculate profit/loss by percentage change
                        entry_price = open_trade['Entry Price']
                        exit_price = open_trade['Exit Price']
                        price_change_percentage = (exit_price - entry_price) / entry_price * 100

                        # Apply trading fee to both entry and exit
                        fee_adjustment = (1 - trading_fee_rate) ** 2
                        profit_loss = (price_change_percentage / 100) * open_trade['Trade Size'] * fee_adjustment

                        # Update equity based on trade result
                        equity += profit_loss
                        if profit_loss > 0:
                            open_trade['Win'] = True
                            num_wins += 1
                        else:
                            open_trade['Win'] = False
                            num_losses += 1
                        
                        open_trade['Profit/Loss'] = profit_loss
                        open_trade['Return (%)'] = price_change_percentage * fee_adjustment
                        open_trade['Equity Before'] = equity - profit_loss
                        open_trade['Equity After'] = equity
                        all_results.append(open_trade)
                        open_trade = None
                        num_trades += 1

                if next_candle_time in ohlcv_data.index:
                    next_candle = ohlcv_data.loc[next_candle_time]
                    order_type = liquidation['Order Type']
                    
                    if order_type == 'short':
                        open_trade = {
                            'Order Type': 'short',
                            'Entry Time': next_candle_time,
                            'Entry Price': next_candle['Close'],
                            'Trade Size': risk_per_trade,
                            'Win': False
                        }
                        
                    elif order_type == 'long':
                        open_trade = {
                            'Order Type': 'long',
                            'Entry Time': next_candle_time,
                            'Entry Price': next_candle['Close'],
                            'Trade Size': risk_per_trade,
                            'Win': False
                        }

            # Update equity history
            equity_history.append(equity)

    # Convert all results to DataFrame
    trades_df = pd.DataFrame(all_results)
    
    total_return_percentage = (equity - starting_equity) / starting_equity * 100
    buy_and_hold_return_percentage = (ohlcv_data.iloc[-1]['Close'] - ohlcv_data.iloc[0]['Close']) / ohlcv_data.iloc[0]['Close'] * 100
    exposure_percentage = (len(trades_df) / len(ohlcv_data)) * 100

    # Final summary
    st.write(f"Number of Trades: {num_trades}")
    st.write(f"Number of Wins: {num_wins}")
    st.write(f"Number of Losses: {num_losses}")

    if num_trades > 0:
        win_rate = (num_wins / num_trades) * 100
        st.write(f"Win Rate: {win_rate:.2f}%")
    else:
        st.write("Win Rate: N/A (No trades executed)")

    st.write(f"Starting Equity: {starting_equity:.2f}")
    st.write(f"Final Equity: {equity:.2f}")
    st.write(f"Total Return (%): {total_return_percentage:.2f}%")
    st.write(f"Buy and Hold Return (%): {buy_and_hold_return_percentage:.2f}%")

    return trades_df, equity, equity_history, total_return_percentage

# Function to plot trades with Plotly
def plot_trades_with_plotly(ohlcv_data, trades_df):
    fig = go.Figure(data=[go.Candlestick(x=ohlcv_data.index,
                                         open=ohlcv_data['Open'],
                                         high=ohlcv_data['High'],
                                         low=ohlcv_data['Low'],
                                         close=ohlcv_data['Close'],
                                         name="OHLCV")])

    long_trades = trades_df[trades_df['Order Type'] == 'long']
    short_trades = trades_df[trades_df['Order Type'] == 'short']

    fig.add_trace(go.Scatter(x=long_trades['Entry Time'],
                             y=long_trades['Entry Price'],
                             mode='markers',
                             marker=dict(symbol="triangle-up", size=10, color='green'),
                             name='Long Entries'))

    fig.add_trace(go.Scatter(x=short_trades['Entry Time'],
                             y=short_trades['Entry Price'],
                             mode='markers',
                             marker=dict(symbol="triangle-down", size=10, color='red'),
                             name='Short Entries'))

    fig.add_trace(go.Scatter(x=trades_df['Exit Time'],
                             y=trades_df['Exit Price'],
                             mode='markers',
                             marker=dict(symbol="x", size=10, color='yellow'),
                             name='Exits'))

    for i, trade in trades_df.iterrows():
        fig.add_trace(go.Scatter(x=[trade['Entry Time'], trade['Exit Time']],
                                 y=[trade['Entry Price'], trade['Exit Price']],
                                 mode='lines',
                                 line=dict(color='yellow', dash='dot'),
                                 name=f'Trade Line {i}',
                                 showlegend=False))

    fig.update_layout(title="Backtested Trades",
                      xaxis_title="Date",
                      yaxis_title="Price (USDT)",
                      template="plotly_dark")

    st.plotly_chart(fig)

def plot_equity_curve(equity_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(equity_history))),
                             y=equity_history,
                             mode='lines',
                             name='Equity Curve'))
    fig.update_layout(title='Equity Curve',
                      xaxis_title='Time',
                      yaxis_title='Equity')
    st.plotly_chart(fig)

# Streamlit app setup
st.title('Liquidation-Based Backtesting Strategy')

st.write("""
## Strategy Description
This strategy takes trades based on liquidation data. It opens trades after a certain amount of liquidation occurs and closes when the opposite liquidation occurs. 
For testing purposes, I have a small sample of liquidation data.
""")

# Set the minimum and maximum dates for selection
min_date = datetime.date(2023, 2, 2)
max_date = datetime.date(2023, 3, 13)

# Date input from user with restrictions
start_date = st.date_input("Select Start Date", min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("Select End Date", max_date, min_value=min_date, max_value=max_date)

# Equity input from user (minimum $100,000)
starting_equity = st.number_input("Enter Starting Equity (Minimum $100,000)", min_value=100000.0, value=100000.0)

st.session_state.last_active = time.time()

# Backtest button
if st.button('Run Backtest'):
    update_last_active()
    trades_df, final_equity, equity_history, total_return_percentage = backtest_all_trades(start_date, end_date, Client.KLINE_INTERVAL_1MINUTE, starting_equity)
    
    plot_trades_with_plotly(fetch_ohlcv_data('BTCUSDT', Client.KLINE_INTERVAL_1MINUTE, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')), trades_df)

    plot_equity_curve(equity_history)

check_timeout()