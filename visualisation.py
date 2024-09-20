import pandas as pd
import plotly.graph_objects as go
from backtesting_script import fetch_ohlcv_data, backtest_all_trades

def plot_trades_with_plotly(ohlcv_data, trades_df):
    # Create a candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=ohlcv_data.index,
                                         open=ohlcv_data['Open'],
                                         high=ohlcv_data['High'],
                                         low=ohlcv_data['Low'],
                                         close=ohlcv_data['Close'],
                                         name="OHLCV")])

    # Add entry points for long and short trades
    long_trades = trades_df[trades_df['Order Type'] == 'long']
    short_trades = trades_df[trades_df['Order Type'] == 'short']

    # Add Long Entries trace
    fig.add_trace(go.Scatter(x=long_trades['Entry Time'],
                             y=long_trades['Entry Price'],
                             mode='markers',
                             marker=dict(symbol="triangle-up", size=10, color='green'),
                             name='Long Entries',
                             showlegend=True))

    # Add Short Entries trace
    fig.add_trace(go.Scatter(x=short_trades['Entry Time'],
                             y=short_trades['Entry Price'],
                             mode='markers',
                             marker=dict(symbol="triangle-down", size=10, color='red'),
                             name='Short Entries',
                             showlegend=True))

    # Add Exit Points trace
    fig.add_trace(go.Scatter(x=trades_df['Exit Time'],
                             y=trades_df['Exit Price'],
                             mode='markers',
                             marker=dict(symbol="x", size=10, color='yellow'),
                             name='Exits',
                             showlegend=True))

    # Add dotted lines from entry to exit, only add one line per trade
    for i, trade in trades_df.iterrows():
        fig.add_trace(go.Scatter(x=[trade['Entry Time'], trade['Exit Time']],
                                 y=[trade['Entry Price'], trade['Exit Price']],
                                 mode='lines',
                                 line=dict(color='yellow', dash='dot'),
                                 name=f'Trade Line {i}',
                                 showlegend=False))  # Hide trade lines from legend

    # Update layout
    fig.update_layout(title="Trades on BTCUSDT",
                      xaxis_title="Time",
                      yaxis_title="Price",
                      template="plotly_dark")
    fig.show()

# Example usage
if __name__ == "__main__":

    # Define the start and end dates for backtesting
    start_date = pd.to_datetime('2023-02-02')
    end_date = pd.to_datetime('2023-02-03')

    # Fetch OHLCV data
    ohlcv_data = fetch_ohlcv_data('BTCUSDT', '1m', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Run backtest to get both trades_df and equity_history
    trades_df, equity_history = backtest_all_trades(start_date, end_date)

    # Pass trades_df to the plotting function
    plot_trades_with_plotly(ohlcv_data, trades_df)

    # Optionally, you can plot the equity curve here if desired
    # plot_equity_curve(equity_history)
