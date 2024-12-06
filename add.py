import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import os
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Constants for symbols (these are just examples)
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

# List of API endpoints
api_endpoints = [
    "https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies?apikey=YOUR_API_KEY",
    "https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs?apikey=YOUR_API_KEY",
    "https://financialmodelingprep.com/api/v3/symbol/available-indexes?apikey=YOUR_API_KEY",
    "https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey=YOUR_API_KEY"
]

# Function to fetch data from each endpoint
def fetch_data(endpoints):
    data = {}
    for url in endpoints:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data[url] = response.json()   # Store the response JSON
        except requests.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            data[url] = None  # Store None for failed requests
    return data

# Fetch and print data
if __name__ == "__main__":
    data = fetch_data(api_endpoints)
    for url, content in data.items():
        print(f"\nData from {url}:")
        if content is not None:
            print(content)
        else:
            print("Failed to retrieve data.")

# Initialize session state variables
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = {}

def calculate_signals(data):
    """Calculate technical indicators and generate signals."""
    data = data.copy()
    data.sort_index(inplace=True)

    # Add technical indicators
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    up_avg = up.rolling(window=14).mean()
    down_avg = down.rolling(window=14).mean()
    epsilon = 1e-6  # Small value to prevent division by zero
    data['RSI'] = 100 - (100 / (1 + (up_avg / (down_avg + epsilon))))

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    data['BB_Mid'] = data['Close'].rolling(window=20).mean()
    data['BB_Std'] = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['BB_Std'])

    # Feature Engineering
    data['Price_Change'] = data['Close'].pct_change()
    data['Volatility'] = data['Price_Change'].rolling(window=10).std()
    data['Momentum'] = data['Close'].diff(4)  # Adding momentum indicator

    # Drop NaN values
    data.dropna(inplace=True)

    return data

def prepare_ml_data(data):
    """Prepare data for machine learning."""
    data = data.copy()
    data['Future_Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD',
                'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility', 'Momentum']
    if not all(feature in data.columns for feature in features):
        return None, None  # Missing features

    X = data[features]
    y = data['Future_Close']  # Regression target

    return X, y

def train_ml_model(X, y):
    """Train the XGBoost model for regression with cross-validation."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)

    # Perform cross-validation with RMSE scoring
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    avg_rmse = np.sqrt(-scores.mean())

    # Fit the model
    model.fit(X, y)

    return model, avg_rmse

def execute_trade(symbol, action, price, amount):
    """Execute buy or sell orders."""
    if action == 'Buy':
        total_cost = amount * price
        if st.session_state.balance >= total_cost:
            st.session_state.balance -= total_cost
            if symbol in st.session_state.positions:
                st.session_state.positions[symbol]['quantity'] += amount
                st.session_state.positions[symbol]['cost_basis'] = price
            else:
                st.session_state.positions[symbol] = {
                    'quantity': amount, 'cost_basis': price}
            st.session_state.trade_history.append({
                'Date': datetime.now(),
                'Symbol': symbol,
                'Action': action,
                'Price': price,
                'Amount': amount
            })
            st.session_state.stop_loss[symbol] = price * (1 - st.session_state.user_stop_loss_pct)
            st.session_state.balance_history.append(
                {'Date': datetime.now(), 'Balance': st.session_state.balance})
            return True
        else:
            st.warning(f"Insufficient balance to execute trade for {symbol}.")
            return False
    elif action == 'Sell':
        if symbol in st.session_state.positions and st.session_state.positions[symbol]['quantity'] >= amount:
            st.session_state.balance += amount * price
            st.session_state.positions[symbol]['quantity'] -= amount
            if st.session_state.positions[symbol]['quantity'] == 0:
                del st.session_state.positions[symbol]
                del st.session_state.stop_loss[symbol]
            st.session_state.trade_history.append({
                'Date': datetime.now(),
                'Symbol': symbol,
                'Action': action,
                'Price': price,
                'Amount': amount
            })
            st.session_state.balance_history.append(
                {'Date': datetime.now(), 'Balance': st.session_state.balance})
            return True
        else:
            st.warning(f"No sufficient holdings to sell for {symbol}.")
            return False

def check_stop_loss(symbol, current_price):
    """Check if the stop-loss condition is met."""
    if symbol in st.session_state.stop_loss and current_price <= st.session_state.stop_loss[symbol]:
        amount = st.session_state.positions[symbol]['quantity']
        if execute_trade(symbol, 'Sell', current_price, amount):
            st.warning(
                f"Stop-loss triggered for {symbol}. Sold {amount} at ${current_price:.2f}")

def calculate_portfolio_metrics():
    """Calculate ROI and Max Drawdown."""
    total_value = st.session_state.balance
    for symbol, position in st.session_state.positions.items():
        data = yf.download(symbol, period='1d')
        if not data.empty:
            current_price = data['Close'].iloc[-1]
            total_value += position['quantity'] * current_price

    initial_balance = st.session_state.initial_balance
    roi = (total_value - initial_balance) / initial_balance * 100

    portfolio_series = pd.Series([entry['Balance'] for entry in st.session_state.balance_history])
    cumulative_max = portfolio_series.cummax()
    drawdowns = (cumulative_max - portfolio_series) / cumulative_max
    max_drawdown = drawdowns.max() * 100

    return roi, max_drawdown

def calculate_profit_loss(symbol, current_price):
    """Calculate profit or loss for a position."""
    if symbol in st.session_state.positions:
        position = st.session_state.positions[symbol]
        quantity = position['quantity']
        cost_basis = position['cost_basis']
        profit_loss = (current_price - cost_basis) * quantity
        return profit_loss
    else:
        return 0.0

def plot_balance_history():
    """Plot the account balance over time."""
    df = pd.DataFrame(st.session_state.balance_history)
    fig = px.line(df, x='Date', y='Balance', title='Account Balance History')
    return fig

def display_signals_table(signals):
    """Display a table with market signals."""
    if signals:
        signals_df = pd.DataFrame(signals)

        # Ensure all expected columns are present
        expected_columns = ['Symbol', 'Current Price', 'Buy at Price', 'Sell at Price', 'Stop Loss', 'Predicted Price Next 24h']
        for col in expected_columns:
            if col not in signals_df.columns:
                signals_df[col] = np.nan

        # Convert columns to numeric where appropriate
        numeric_columns = ['Current Price', 'Buy at Price', 'Sell at Price', 'Stop Loss', 'Predicted Price Next 24h']
        for col in numeric_columns:
            signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')

        # Apply styling
        st.subheader("Signals")
        st.dataframe(
            signals_df.style.format({
                'Current Price': '${:,.2f}',
                'Buy at Price': '${:,.2f}',
                'Sell at Price': '${:,.2f}',
                'Stop Loss': '${:,.2f}',
                'Predicted Price Next 24h': '${:,.2f}'
            }, na_rep='N/A')
        )
    else:
        st.write("No signals to display.")

def main():
    st.title("Enhanced Multi-Asset Trading Bot with Machine Learning")

    st.sidebar.header("Settings")

    # Configurable initial balance
    if 'initial_balance' not in st.session_state:
        initial_balance = st.sidebar.number_input(
            "Set Initial Balance", min_value=1000, value=100000, step=1000)
        st.session_state.initial_balance = initial_balance
        st.session_state.balance = initial_balance
        st.session_state.balance_history = [
            {'Date': datetime.now(), 'Balance': st.session_state.balance}]
    else:
        st.sidebar.write(
            f"Initial Balance: ${st.session_state.initial_balance:,.2f}")

    # Risk management settings
    st.session_state.user_stop_loss_pct = st.sidebar.slider(
        "Stop-Loss Percentage", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    asset_type = st.sidebar.selectbox(
        "Select Asset Type", ["Commodities", "Forex", "Crypto", "Indices"])

    if asset_type == "Commodities":
        symbols = COMMODITIES
    elif asset_type == "Forex":
        symbols = FOREX_SYMBOLS
    elif asset_type == "Crypto":
        symbols = CRYPTO_SYMBOLS
    elif asset_type == "Indices":
        symbols = INDICES_SYMBOLS

    start_date = st.sidebar.date_input(
        "Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())

    trade_amount = st.sidebar.number_input(
        "Trade Amount per Asset", min_value=0.01, value=1.0, step=0.01)

    if st.sidebar.button("Run Analysis"):
        all_data = {}
        signals_list = []
        model_scores = []

        progress_bar = st.progress(0)
        total_symbols = len(symbols)

        for idx, symbol in enumerate(symbols):
            try:
                data = get_data(symbol, start_date, end_date)
                if data.empty:
                    st.warning(f"No data retrieved for {symbol}.")
                    continue

                data = calculate_signals(data)
                X, y = prepare_ml_data(data)
                if X is None or y is None:
                    st.warning(f"Skipping {symbol} due to insufficient data or missing features.")
                    continue

                model, rmse = train_ml_model(X, y)
                model_scores.append(rmse)

                latest_data = data.iloc[-1]
                features = ['Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD',
                            'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility', 'Momentum']
                X_latest = latest_data[features].values.reshape(1, -1)
                predicted_price = model.predict(X_latest)[0]

                current_price = latest_data['Close']

                # Determine recommended action based on the predicted price movement
                if predicted_price > current_price:
                    action = 'Buy'
                else:
                    action = 'Sell'

                # Calculate stop loss price
                stop_loss_price = current_price * (1 - st.session_state.user_stop_loss_pct)

                # Execute trade automatically
                if action == 'Buy':
                    execute_trade(symbol, 'Buy', current_price, trade_amount)
                elif action == 'Sell':
                    if symbol in st.session_state.positions:
                        amount_to_sell = st.session_state.positions[symbol]['quantity']
                        execute_trade(symbol, 'Sell', current_price, amount_to_sell)

                # Prepare signal data
                signals_list.append({
                    'Symbol': symbol,
                    'Current Price': current_price,
                    'Buy at Price': current_price if action == 'Buy' else np.nan,
                    'Sell at Price': current_price if action == 'Sell' else np.nan,
                    'Stop Loss': stop_loss_price if action == 'Buy' else np.nan,
                    'Predicted Price Next 24h': predicted_price
                })

                # Check stop-loss
                check_stop_loss(symbol, current_price)

            except Exception as e:
                st.error(f"An error occurred while processing {symbol}: {e}")

            progress_bar.progress((idx + 1) / total_symbols)

        # Display signals in a table
        display_signals_table(signals_list)

        # Display current positions
        st.subheader("Current Positions")
        positions_list = []
        for symbol, position in st.session_state.positions.items():
            data = yf.download(symbol, period='1d')
            if not data.empty:
                current_price = data['Close'].iloc[-1]
                quantity = position['quantity']
                cost_basis = position['cost_basis']
                market_value = quantity * current_price
                profit_loss = calculate_profit_loss(symbol, current_price)
                positions_list.append({
                    'Symbol': symbol,
                    'Quantity': quantity,
                    'Cost Basis': cost_basis,
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Profit/Loss': profit_loss
                })
        if positions_list:
            positions_df = pd.DataFrame(positions_list)
            st.dataframe(positions_df.style.format({
                'Cost Basis': '${:,.2f}',
                'Current Price': '${:,.2f}',
                'Market Value': '${:,.2f}',
                'Profit/Loss': '${:,.2f}'
            }))
        else:
            st.write("No positions currently held.")

        # Display trade history
        st.subheader("Trade History")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(history_df)
        else:
            st.write("No trades executed yet.")

        # Display account information
        st.subheader("Account Information")
        st.write(f"Current Balance: ${st.session_state.balance:,.2f}")
        roi, max_drawdown = calculate_portfolio_metrics()
        st.write(f"ROI: {roi:.2f}%")
        st.write(f"Max Drawdown: {max_drawdown:.2f}%")
        if model_scores:
            avg_rmse = sum(model_scores) / len(model_scores)
            st.write(f"Average Model RMSE: ${avg_rmse:.2f}")

        # Display balance history chart
        st.subheader("Account Balance History")
        st.plotly_chart(plot_balance_history())

if __name__ == "__main__":
    main()
