import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Constants for symbols (for use elsewhere in your program)
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

# List of API endpoints
api_endpoints = {
    "cryptocurrencies": "https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies?apikey={}",
    "forex": "https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs?apikey={}",
    "indexes": "https://financialmodelingprep.com/api/v3/symbol/available-indexes?apikey={}",
    "commodities": "https://financialmodelingprep.com/api/v3/symbol/available-commodities?apikey={}"
}

def get_data(symbol, start_date, end_date):
    """Fetch historical data for a given symbol."""
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        st.error("API key not found in environment variables. Set 'FMP_API_KEY'.")
        return pd.DataFrame()

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Debugging - print the response
        st.write(f"Response for {symbol}: {data}")

        if 'historical' in data and len(data['historical']) > 0:
            return pd.DataFrame(data['historical'])
        else:
            st.error(f"No historical data found for {symbol}. Check if the symbol is correct.")
            return pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_data(endpoints):
    data = {}
    api_key = os.getenv("FMP_API_KEY")
    
    if not api_key:
        raise ValueError("API key not found in environment variables. Set 'FMP_API_KEY'.")

    for key, url in endpoints.items():
        try:
            full_url = url.format(api_key)
            response = requests.get(full_url, headers={"User-Agent": "MyApp/1.0"})
            response.raise_for_status()
            data[key] = response.json()
        except requests.RequestException as e:
            st.error(f"Error fetching data from {key}: {e}")
            data[key] = None

    return data

# Initialize session state variables if they don't exist
if 'positions' not in st.session_state:
    st.session_state.positions = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'stop_loss' not in st.session_state:
    st.session_state.stop_loss = {}
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = []

def calculate_signals(data):
    """Calculate technical indicators and generate signals."""
    data = data.copy()
    data.sort_index(inplace=True)

    # Add technical indicators
    data['SMA20'] = data['close'].rolling(window=20).mean()
    data['SMA50'] = data['close'].rolling(window=50).mean()
    data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()

    delta = data['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    up_avg = up.rolling(window=14).mean()
    down_avg = down.rolling(window=14).mean()
    epsilon = 1e-6
    data['RSI'] = 100 - (100 / (1 + (up_avg / (down_avg + epsilon))))

    exp1 = data['close'].ewm(span=12, adjust=False).mean()
    exp2 = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

    data['BB_Mid'] = data['close'].rolling(window=20).mean()
    data['BB_Std'] = data['close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Mid'] - (2 * data['BB_Std'])

    data['Price_Change'] = data['close'].pct_change()
    data['Volatility'] = data['Price_Change'].rolling(window=10).std()
    data['Momentum'] = data['close'].diff(4)

    data.dropna(inplace=True)

    return data

def prepare_ml_data(data):
    """Prepare data for machine learning."""
    data = data.copy()
    data['Future_Close'] = data['close'].shift(-1)
    data.dropna(inplace=True)

    features = ['close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility', 'Momentum']
    if not all(feature in data.columns for feature in features):
        return None, None

    X = data[features]
    y = data['Future_Close']

    return X, y

def train_ml_model(X, y):
    """Train the XGBoost model for regression with cross-validation."""
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    avg_rmse = np.sqrt(-scores.mean())
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
            else:
                st.session_state.positions[symbol] = {'quantity': amount, 'cost_basis': price}
            st.session_state.trade_history.append({'Date': datetime.now(), 'Symbol': symbol, 'Action': action, 'Price': price, 'Amount': amount})
            st.session_state.stop_loss[symbol] = price * (1 - st.session_state.user_stop_loss_pct)
            st.session_state.balance_history.append({'Date': datetime.now(), 'Balance': st.session_state.balance})
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
            st.session_state.trade_history.append({'Date': datetime.now(), 'Symbol': symbol, 'Action': action, 'Price': price, 'Amount': amount})
            st.session_state.balance_history.append({'Date': datetime.now(), 'Balance': st.session_state.balance})
            return True
        else:
            st.warning(f"No sufficient holdings to sell for {symbol}.")
            return False

def check_stop_loss(symbol, current_price):
    """Check if the stop-loss condition is met."""
    if symbol in st.session_state.stop_loss and current_price <= st.session_state.stop_loss[symbol]:
        amount = st.session_state.positions[symbol]['quantity']
        if execute_trade(symbol, 'Sell', current_price, amount):
            st.warning(f"Stop-loss triggered for {symbol}. Sold {amount} at ${current_price:.2f}")

def calculate_portfolio_metrics():
    """Calculate ROI and Max Drawdown."""
    total_value = st.session_state.balance
    for symbol, position in st.session_state.positions.items():
        data = get_data(symbol, '2020-01-01', datetime.now().strftime('%Y-%m-%d'))  # Use FMP for the latest price
        if not data.empty and 'close' in data.columns:
            current_price = data['close'].iloc[-1]
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
        expected_columns = ['Symbol', 'Current Price', 'Buy at Price', 'Sell at Price', 'Stop Loss', 'Predicted Price Next 24h']
        for col in expected_columns:
            if col not in signals_df.columns:
                signals_df[col] = np.nan

        numeric_columns = ['Current Price', 'Buy at Price', 'Sell at Price', 'Stop Loss', 'Predicted Price Next 24h']
        for col in numeric_columns:
            signals_df[col] = pd.to_numeric(signals_df[col], errors='coerce')

        st.subheader("Signals")
        st.dataframe(signals_df.style.format({
            'Current Price': '${:,.2f}',
            'Buy at Price': '${:,.2f}',
            'Sell at Price': '${:,.2f}',
            'Stop Loss': '${:,.2f}',
            'Predicted Price Next 24h': '${:,.2f}'
        }, na_rep='N/A'))
    else:
        st.write("No signals to display.")

def main():
    st.title("Enhanced Multi-Asset Trading Bot with Machine Learning")

    st.sidebar.header("Settings")

    if 'initial_balance' not in st.session_state:
        initial_balance = st.sidebar.number_input("Set Initial Balance", min_value=1000, value=100000, step=1000)
        st.session_state.initial_balance = initial_balance
        st.session_state.balance = initial_balance
        st.session_state.balance_history.append({'Date': datetime.now(), 'Balance': st.session_state.balance})
    else:
        st.sidebar.write(f"Initial Balance: ${st.session_state.initial_balance:,.2f}")

    st.session_state.user_stop_loss_pct = st.sidebar.slider("Stop-Loss Percentage", min_value=0.01, max_value=0.2, value=0.05, step=0.01)

    asset_type = st.sidebar.selectbox("Select Asset Type", ["Commodities", "Forex", "Crypto", "Indices"])
    symbols = {
        "Commodities": COMMODITIES,
        "Forex": FOREX_SYMBOLS,
        "Crypto": CRYPTO_SYMBOLS,
        "Indices": INDICES_SYMBOLS
    }[asset_type]

    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", datetime.now())
    trade_amount = st.sidebar.number_input("Trade Amount per Asset", min_value=0.01, value=1.0, step=0.01)

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

                # Check for 'close' column presence
                if 'close' not in data.columns:
                    st.warning(f"The 'close' column is missing in the data for {symbol}.")
                    continue

                data = calculate_signals(data)
                X, y = prepare_ml_data(data)
                if X is None or y is None:
                    st.warning(f"Skipping {symbol} due to insufficient data or missing features.")
                    continue

                model, rmse = train_ml_model(X, y)
                model_scores.append(rmse)

                latest_data = data.iloc[-1]
                features = ['close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Lower', 'Volatility', 'Momentum']
                X_latest = latest_data[features].values.reshape(1, -1)
                predicted_price = model.predict(X_latest)[0]

                current_price = latest_data['close']

                action = 'Buy' if predicted_price > current_price else 'Sell'
                stop_loss_price = current_price * (1 - st.session_state.user_stop_loss_pct)

                if action == 'Buy':
                    execute_trade(symbol, 'Buy', current_price, trade_amount)
                elif action == 'Sell' and symbol in st.session_state.positions:
                    amount_to_sell = st.session_state.positions[symbol]['quantity']
                    execute_trade(symbol, 'Sell', current_price, amount_to_sell)

                signals_list.append({
                    'Symbol': symbol,
                    'Current Price': current_price,
                    'Buy at Price': current_price if action == 'Buy' else np.nan,
                    'Sell at Price': current_price if action == 'Sell' else np.nan,
                    'Stop Loss': stop_loss_price if action == 'Buy' else np.nan,
                    'Predicted Price Next 24h': predicted_price
                })

                check_stop_loss(symbol, current_price)

            except Exception as e:
                st.error(f"An error occurred while processing {symbol}: {e}")

            progress_bar.progress((idx + 1) / total_symbols)

        display_signals_table(signals_list)

        st.subheader("Current Positions")
        positions_list = []
        for symbol, position in st.session_state.positions.items():
            data = get_data(symbol, '2020-01-01', datetime.now().strftime('%Y-%m-%d'))  # Get latest price
            if not data.empty and 'close' in data.columns:
                current_price = data['close'].iloc[-1]
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

        st.subheader("Trade History")
        if st.session_state.trade_history:
            history_df = pd.DataFrame(st.session_state.trade_history)
            st.dataframe(history_df)
        else:
            st.write("No trades executed yet.")

        st.subheader("Account Information")
        st.write(f"Current Balance: ${st.session_state.balance:,.2f}")
        roi, max_drawdown = calculate_portfolio_metrics()
        st.write(f"ROI: {roi:.2f}%")
        st.write(f"Max Drawdown: {max_drawdown:.2f}%")
        if model_scores:
            avg_rmse = sum(model_scores) / len(model_scores)
            st.write(f"Average Model RMSE: {avg_rmse:.2f}")

        st.subheader("Account Balance History")
        st.plotly_chart(plot_balance_history())

if __name__ == "__main__":
    main()
