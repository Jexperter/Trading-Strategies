import pandas as pd
import plotly.graph_objects as go
from backtesting_script import backtest_all_trades

def plot_equity_curve(equity_history):
    # Create a line chart for equity history
    fig = go.Figure(data=[go.Scatter(x=list(range(len(equity_history))),
                                     y=equity_history,
                                     mode='lines',
                                     name='Equity Curve',
                                     line=dict(color='blue'))])
    
    # Update layout
    fig.update_layout(title="Equity Curve",
                      xaxis_title="Trade Number",
                      yaxis_title="Equity",
                      template="plotly_dark")
    
    # Show plot
    fig.show()

# Main logic to fetch data and plot the equity curve
if __name__ == "__main__":
    import datetime
    
    # Define the start and end dates for the backtesting
    start_date = pd.to_datetime('2023-02-02')
    end_date = pd.to_datetime('2023-02-03')

    # Run backtest to get equity history
    trades_df, equity, equity_history, period_return = backtest_all_trades(start_date, end_date)

    # Plot the equity curve
    plot_equity_curve(equity_history)
