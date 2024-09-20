import pandas as pd
import numpy as np
import os
from binance.client import Client

# Initialize Binance Client (use your API key and secret)
api_key = os.getenv('API_KEY_BINANCE')
api_secret = os.getenv('API_SECRET_BINANCE')
client = Client(api_key, api_secret)

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
def backtest_all_trades(start_date, end_date):
    # Load liquidation data
    liquidation_data = pd.read_csv(
        r'C:\Users\Jesann\Documents\trading\history data file\liq data\BTCUSDT (02.02.2023 to 13.03.2023) liq data.csv',
        parse_dates=['Datetime'],
        dayfirst=True  # Ensure day first for European date format
    )
    
    # Filter liquidation data based on provided dates
    liquidation_data = liquidation_data[
        (liquidation_data['Datetime'] >= start_date) &
        (liquidation_data['Datetime'] <= end_date)
    ]
    
    # Floor liquidation time to minutes (ignoring seconds)
    liquidation_data['Datetime'] = liquidation_data['Datetime'].dt.floor('T')

    # Fetch OHLCV data from Binance for the whole period
    ohlcv_data = fetch_ohlcv_data('BTCUSDT', '1m', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Initialize equity
    starting_equity = 100000
    equity = starting_equity
    risk_per_trade = 0.05 * starting_equity  # Risking 5% of account per trade

    # Trading fee (0.2% per trade)
    trading_fee_rate = 0.002

    # Initialize statistics variables
    num_trades = 0
    num_wins = 0
    num_losses = 0

    # Initialize equity history
    equity_history = [starting_equity]

    # Prepare results list
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

    # Print details of each trade at the start
    print("\n--- Trade Details ---")
    for idx, trade in trades_df.iterrows():
        print(f"Trade {idx+1}:")
        print(f"  Order Type: {trade['Order Type']}")
        print(f"  Entry Time: {trade['Entry Time']}")
        print(f"  Entry Price: {trade['Entry Price']:.2f}")
        print(f"  Exit Time: {trade['Exit Time']}")
        print(f"  Exit Price: {trade['Exit Price']:.2f}")
        print(f"  Duration (minutes): {trade['Duration']:.2f}")
        print(f"  Profit/Loss: {trade['Profit/Loss']:.2f}")
        print(f"  Return (%): {trade['Return (%)']:.2f}")
        print(f"  Equity Before: {trade['Equity Before']:.2f}")
        print(f"  Equity After: {trade['Equity After']:.2f}\n")

    total_return_percentage = (equity - starting_equity) / starting_equity * 100
    buy_and_hold_return_percentage = (ohlcv_data.iloc[-1]['Close'] - ohlcv_data.iloc[0]['Close']) / ohlcv_data.iloc[0]['Close'] * 100
    exposure_percentage = (len(trades_df) / len(ohlcv_data)) * 100
    equity_peak = max(equity_history)
    num_trading_days = (end_date - start_date).days

    # Calculate daily returns and related metrics
    equity_changes = np.diff(equity_history) / equity_history[:-1]
    daily_returns = pd.Series(equity_changes)

    # Annualized Return
    annualized_return = (equity / starting_equity) ** (365 / num_trading_days) - 1

    # Annualized Volatility
    annualized_volatility = daily_returns.std() * np.sqrt(365)

    # Risk-free rate (annualized)
    risk_free_rate = 0.0

    # Sharpe Ratio (with 5% risk-free rate)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Sortino Ratio (using only negative returns for downside risk)
    downside_risk = daily_returns[daily_returns < 0].std() * np.sqrt(365)
    sortino_ratio = (annualized_return - risk_free_rate) / downside_risk

    # Max Drawdown and Avg Drawdown
    drawdowns = []
    max_drawdown = 0
    equity_peak = equity_history[0]

    for equity_value in equity_history:
        equity_peak = max(equity_peak, equity_value)
        drawdown = (equity_peak - equity_value) / equity_peak
        drawdowns.append(drawdown)
        max_drawdown = max(max_drawdown, drawdown)

    max_drawdown_percentage = max_drawdown * 100
    avg_drawdown_percentage = np.mean(drawdowns) * 100

    # Calmar Ratio (annualized return divided by max drawdown)
    calmar_ratio = annualized_return / max_drawdown

    # Store max drawdown and avg drawdown (converted to percentage)
    #max_drawdown_percentage = max_drawdown
    #avg_drawdown_percentage = avg_drawdown

    # Calculate period return (final equity / starting equity)
    period_return = (equity / starting_equity) - 1

    # Final summary
    print(f"Number of Trades: {num_trades}")
    print(f"Number of Wins: {num_wins}")
    print(f"Number of Losses: {num_losses}")
    print(f"Win Rate: {num_wins / num_trades * 100:.2f}%")
    print(f"Starting Equity: {starting_equity:.2f}")
    print(f"Final Equity: {equity:.2f}")
    print(f"Total Return (%): {total_return_percentage:.2f}%")
    print(f"Buy and Hold Return (%): {buy_and_hold_return_percentage:.2f}%")
    print(f"Exposure (%): {exposure_percentage:.2f}%")
    print(f"Max Drawdown (%): {max_drawdown_percentage:.2f}%")
    print(f"Avg Drawdown (%): {avg_drawdown_percentage:.2f}%")
    print(f"Annualized Return (%): {annualized_return * 100:.2f}%")
    print(f"Annualized Volatility (%): {annualized_volatility * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Calmar Ratio: {calmar_ratio:.2f}")
    print(f"Equity Peak: {equity_peak:.2f}")
    print(f"Period Return: {period_return * 100:.2f}%")
    
    return trades_df, equity, equity_history, period_return

# Run backtest for specific period
start_date = pd.to_datetime('2023-02-02')
end_date = pd.to_datetime('2023-03-13')

trades_df, final_equity, equity_history, period_return = backtest_all_trades(start_date, end_date)
