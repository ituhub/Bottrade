import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

st.set_page_config(
    page_title="Advanced Trading Bot Dashboard with Enhanced Feature Engineering",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

if 'initial_balance' not in st.session_state:
    st.session_state.initial_balance = 10000
if 'balance' not in st.session_state:
    st.session_state.balance = st.session_state.initial_balance
if 'allocated_capital' not in st.session_state:
    st.session_state.allocated_capital = {}
if 'open_positions' not in st.session_state:
    st.session_state.open_positions = {}
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = []

st.title("🚀 Advanced Trading Bot Dashboard with Enhanced Feature Engineering")
st.markdown("""
This version includes:
- Advanced feature engineering for the XGBoost model:
  - Technical indicators (RSI, MACD, Bollinger Bands, Fibonacci Retracements)
  - Time-based cyclical features (hour_of_day, day_of_week)
  - Relative price difference features
- Integration of Bollinger Bands and Fibonacci Retracement strategies to complement RSI and MACD.
- LSTM-based forecasting to enhance detection of non-linear price patterns.
- The rest of the logic (Prophet predictions, signal generation, etc.) remains intact.
""")

st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Asset Class", ["Forex", "Commodities", "Indices", "Cryptocurrency"])

tickers = []
asset_class = None

if section == "Forex":
    st.header("💱 Top Forex Pairs")
    tickers = FOREX_SYMBOLS
    asset_class = 'Forex'
elif section == "Commodities":
    st.header("🌐 Top Commodities")
    tickers = COMMODITIES
    asset_class = 'Commodities'
elif section == "Indices":
    st.header("📊 Global Indices Overview")
    tickers = INDICES_SYMBOLS
    asset_class = 'Indices'
elif section == "Cryptocurrency":
    st.header("💎 Top Cryptocurrencies")
    tickers = CRYPTO_SYMBOLS
    asset_class = 'Cryptocurrency'
else:
    st.error("Invalid section selected.")
    st.stop()

if not tickers:
    st.error(f"No tickers defined for section: {section}")
    st.stop()

st.write(f"Selected Asset Class: {asset_class}")
st.write(f"Tickers: {tickers}")

num_tickers = len(tickers)
capital_per_ticker = st.session_state.balance / num_tickers

for ticker in tickers:
    if ticker not in st.session_state.allocated_capital:
        st.session_state.allocated_capital[ticker] = capital_per_ticker
    if ticker not in st.session_state.open_positions:
        st.session_state.open_positions[ticker] = None

def fetch_live_data(tickers, asset_class):
    data = {}
    api_key = os.getenv("FMP_API_KEY")

    if not api_key:
        st.error("API key not found in environment variables. Set 'FMP_API_KEY'.")
        return data

    for ticker in tickers:
        try:
            ticker_api = ticker.replace('/', '')
            url = f'https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}'
            response = requests.get(url)
            response.raise_for_status()
            data_json = response.json()

            if not data_json or len(data_json) < 1:
                st.warning(f"No data returned for {ticker}.")
                continue

            df = pd.DataFrame(data_json)
            if df.empty:
                st.warning(f"No data available for {ticker}.")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
            df.sort_index(inplace=True)

            df_hourly = df.resample('H').last().dropna(subset=['Close'])
            if df_hourly.empty:
                st.warning(f"No hourly data for {ticker}.")
                continue

            data[ticker] = df_hourly
        except Exception as e:
            st.warning(f"Failed to fetch data for {ticker}: {e}")
    return data

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

def compute_bollinger_bands(df, period=20, num_std=2):
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + num_std * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - num_std * df['BB_Std']
    return df

def compute_fibonacci_levels(df, lookback=100):
    if len(df) < lookback:
        lookback = len(df)
    recent_data = df.tail(lookback)
    swing_high = recent_data['High'].max()
    swing_low = recent_data['Low'].min()

    diff = swing_high - swing_low
    levels = {
        'Fibo_23.6': swing_high - 0.236 * diff,
        'Fibo_38.2': swing_high - 0.382 * diff,
        'Fibo_50.0': swing_high - 0.5 * diff,
        'Fibo_61.8': swing_high - 0.618 * diff,
    }
    for k,v in levels.items():
        df[k] = v
    return df

def compute_indicators(df, asset_class):
    df = df.copy()
    df['MA_Short'] = df['Close'].rolling(window=5).mean()
    df['MA_Long'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    df = compute_bollinger_bands(df, period=20, num_std=2)
    df = compute_fibonacci_levels(df, lookback=100)
    return df.dropna()

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1
    return df

def simulate_trades_live(data):
    for ticker in tickers:
        if ticker in data:
            df = compute_indicators(data[ticker], asset_class)
            if df.empty:
                continue
            df = generate_signals(df)
            allocated = st.session_state.allocated_capital[ticker]
            position = st.session_state.open_positions[ticker]

            current_time = df.index[-1]
            row = df.iloc[-1]
            signal = row['Signal']
            price = row['Close']

            if position is None:
                if signal == 1:
                    quantity = allocated / price
                    buy_price = price
                    position = {
                        'Buy_Time': current_time,
                        'Buy_Price': buy_price,
                        'Quantity': quantity
                    }
                    st.session_state.open_positions[ticker] = position
                    st.session_state.balance -= allocated
                    st.session_state.balance_history.append({'Time': current_time, 'Balance': st.session_state.balance})
                    st.success(f"✅ Bought {ticker} at ${buy_price:.2f} on {current_time}")
            else:
                if price >= position['Buy_Price'] * 1.10 or signal == -1:
                    sell_price = price
                    profit = (sell_price - position['Buy_Price']) * position['Quantity']
                    st.session_state.balance += allocated + profit
                    st.session_state.balance_history.append({'Time': current_time, 'Balance': st.session_state.balance})
                    st.session_state.trade_history.append({
                        'Ticker': ticker,
                        'Buy_Time': position['Buy_Time'],
                        'Buy_Price': position['Buy_Price'],
                        'Sell_Time': current_time,
                        'Sell_Price': sell_price,
                        'Profit/Loss': profit
                    })
                    st.success(f"✅ Sold {ticker} at ${sell_price:.2f} on {current_time} | Profit: ${profit:.2f}")
                    st.session_state.open_positions[ticker] = None

data = fetch_live_data(tickers, asset_class)
if not data:
    st.error("No data fetched.")
    st.stop()

simulate_trades_live(data)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.header("💰 Account Overview")
    st.metric("Initial Balance", f"${st.session_state.initial_balance:,.2f}")
    st.metric("Current Balance", f"${st.session_state.balance:,.2f}")
    if st.session_state.trade_history:
        total_profit = sum([trade['Profit/Loss'] for trade in st.session_state.trade_history])
        st.metric("Total Profit/Loss", f"${total_profit:,.2f}")
        num_trades = len(st.session_state.trade_history)
        winning_trades = sum(1 for trade in st.session_state.trade_history if trade['Profit/Loss'] > 0)
        win_rate = (winning_trades / num_trades) * 100 if num_trades > 0 else 0
        st.metric("Total Trades", f"{num_trades}")
        st.metric("Winning Percentage", f"{win_rate:.2f}%")
    else:
        st.metric("Total Profit/Loss", "$0.00")
        st.metric("Total Trades", "0")
        st.metric("Winning Percentage", "0.00%")

with col2:
    st.header("📈 Account Balance Over Time")
    if st.session_state.balance_history:
        balance_df = pd.DataFrame(st.session_state.balance_history).drop_duplicates(subset=['Time'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=balance_df['Time'], y=balance_df['Balance'], mode='lines', name='Balance'))
        fig.update_layout(xaxis_title='Time', yaxis_title='Balance ($)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No account activity to display.")

st.markdown("---")
st.header("📝 Trade History")
if st.session_state.trade_history:
    trades_df = pd.DataFrame(st.session_state.trade_history)
    trades_df['Buy_Time'] = trades_df['Buy_Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Sell_Time'] = trades_df['Sell_Time'].dt.strftime('%Y-%m-%d %H:%M')
    trades_df['Profit/Loss'] = trades_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
    trades_df_display = trades_df[['Ticker', 'Buy_Time', 'Buy_Price', 'Sell_Time', 'Sell_Price', 'Profit/Loss']]
    st.dataframe(trades_df_display.style.format({'Buy_Price': '${:,.2f}', 'Sell_Price': '${:,.2f}'}))
else:
    st.info("No trades executed yet.")

st.markdown("---")
st.header("📌 Current Open Positions")
if any(position is not None for position in st.session_state.open_positions.values()):
    open_positions_list = []
    for ticker, position in st.session_state.open_positions.items():
        if position and ticker in data and not data[ticker].empty:
            current_price = data[ticker]['Close'][-1]
            profit_loss = (current_price - position['Buy_Price']) * position['Quantity']
            open_positions_list.append({
                'Ticker': ticker,
                'Buy_Time': position['Buy_Time'].strftime('%Y-%m-%d %H:%M'),
                'Buy_Price': position['Buy_Price'],
                'Current_Price': current_price,
                'Profit/Loss': profit_loss
            })
    if open_positions_list:
        open_positions_df = pd.DataFrame(open_positions_list)
        open_positions_df['Profit/Loss'] = open_positions_df['Profit/Loss'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(open_positions_df.style.format({'Buy_Price': '${:,.2f}', 'Current_Price': '${:,.2f}'}))
    else:
        st.info("No open positions to display.")
else:
    st.info("No open positions.")

st.markdown("---")

#############################################
# Forecasting and Feature Engineering for Prophet and XGBoost
#############################################

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def train_test_prophet(df, test_hours=24):
    if len(df) < test_hours + 24:
        return None, None, None, None

    prophet_df = df.reset_index()[['date','Close']].rename(columns={'date':'ds','Close':'y'}).sort_values('ds')
    train_df = prophet_df.iloc[:-test_hours]
    test_df = prophet_df.iloc[-test_hours:]

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(train_df)

    future_test = m.make_future_dataframe(periods=test_hours, freq='H')
    forecast_test = m.predict(future_test)
    forecast_test = forecast_test.set_index('ds')
    common_times = forecast_test.index.intersection(test_df['ds'])
    if len(common_times) < 1:
        return None, None, None, None

    test_forecast = forecast_test.loc[common_times]
    test_df = test_df[test_df['ds'].isin(common_times)]

    if len(test_df) < 1:
        return None, None, None, None

    y_true = test_df['y'].values
    y_pred = test_forecast['yhat'].values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = max(0, 100 - mape)

    future_full = m.make_future_dataframe(periods=24, freq='H')
    forecast_full = m.predict(future_full).set_index('ds')
    last_date = prophet_df['ds'].iloc[-1]

    return m, forecast_full, accuracy, last_date

def multi_horizon_forecast_with_accuracy_prophet(df, horizons=[8,16,24]):
    if df.empty:
        last_close = df['Close'].iloc[-1] if not df.empty else 100.0
        return {h: last_close for h in horizons}, 0.0

    res = train_test_prophet(df, test_hours=24)
    if res[0] is None:
        last_close = df['Close'].iloc[-1]
        return {h: last_close for h in horizons}, 0.0

    m, forecast_full, accuracy, last_date = res
    pred = {}
    for h in horizons:
        target_date = last_date + timedelta(hours=h)
        if target_date in forecast_full.index:
            pred[h] = forecast_full.loc[target_date, 'yhat']
        else:
            pred[h] = forecast_full['yhat'].iloc[-1]
    return pred, accuracy

#############################################
# Enhanced Feature Engineering for XGBoost
#############################################
def create_xgb_features(df):
    dff = df.copy()

    # Time-based features
    dff['hour'] = dff.index.hour
    dff['day_of_week'] = dff.index.dayofweek
    # Cyclical encoding for hour_of_day (24h cycle)
    dff['hour_sin'] = np.sin(2 * np.pi * dff['hour'] / 24)
    dff['hour_cos'] = np.cos(2 * np.pi * dff['hour'] / 24)
    # Cyclical encoding for day_of_week (7-day cycle)
    dff['dow_sin'] = np.sin(2 * np.pi * dff['day_of_week'] / 7)
    dff['dow_cos'] = np.cos(2 * np.pi * dff['day_of_week'] / 7)

    # Lag features
    max_lag = 6
    for i in range(1, max_lag+1):
        dff[f'Close_lag_{i}'] = dff['Close'].shift(i)

    # Relative features
    if 'BB_Middle' in dff.columns:
        dff['Close_minus_BB_Mid'] = dff['Close'] - dff['BB_Middle']
    if 'Fibo_50.0' in dff.columns:
        dff['Close_minus_Fibo_50'] = dff['Close'] - dff['Fibo_50.0']

    feature_candidates = [
        'RSI','MACD','MACD_Signal',
        'BB_Middle','BB_Upper','BB_Lower',
        'Fibo_23.6','Fibo_38.2','Fibo_50.0','Fibo_61.8',
        'Close_minus_BB_Mid','Close_minus_Fibo_50'
    ]

    dff = dff.dropna(subset=['Close'] + [f'Close_lag_{i}' for i in range(1, max_lag+1)])
    return dff, feature_candidates

def xgb_forecast(df, horizons=[8,16,24]):
    if len(df) < 100:
        last_close = df['Close'].iloc[-1]
        return {h: last_close for h in horizons}

    dff, feature_candidates = create_xgb_features(df)

    valid_features = [c for c in feature_candidates if c in dff.columns]
    lag_features = [c for c in dff.columns if 'lag_' in c]
    time_features = ['hour_sin','hour_cos','dow_sin','dow_cos']
    all_features = valid_features + lag_features + time_features

    dff = dff.dropna(subset=all_features)
    if len(dff) < 50:
        last_close = df['Close'].iloc[-1]
        return {h: last_close for h in horizons}

    X = dff[all_features]
    y = dff['Close']

    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X, y)

    current_vals = dff.iloc[-1].copy()

    def shift_lags(vals, new_close):
        for i in range(6,0,-1):
            vals[f'Close_lag_{i}'] = vals[f'Close_lag_{i-1}'] if i>1 else new_close

    preds = {}
    current_dt = dff.index[-1]
    for h in horizons:
        steps = h
        temp_vals = current_vals.copy()
        for step in range(1, steps+1):
            future_dt = current_dt + timedelta(hours=step)
            future_hour = future_dt.hour
            future_dow = future_dt.dayofweek
            temp_vals['hour'] = future_hour
            temp_vals['day_of_week'] = future_dow
            temp_vals['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
            temp_vals['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)
            temp_vals['dow_sin'] = np.sin(2 * np.pi * future_dow / 7)
            temp_vals['dow_cos'] = np.cos(2 * np.pi * future_dow / 7)

            X_pred = temp_vals[all_features].values.reshape(1,-1)
            pred_close = model.predict(X_pred)[0]

            shift_lags(temp_vals, pred_close)

        preds[h] = pred_close

    return preds

#############################################
# LSTM-based Forecasting for Non-Linear Patterns
#############################################
def lstm_forecast(df, horizons=[8,16,24], lookback=24):
    # For simplicity, we train a simple LSTM on recent data to predict next step prices.
    # Then we iterate to predict further horizons.
    # This is a simplified LSTM approach due to environment constraints.

    if len(df) < lookback + 24:
        last_close = df['Close'].iloc[-1]
        return {h: last_close for h in horizons}

    dff = df[['Close']].copy()
    dff = dff.dropna()
    if len(dff) < lookback:
        last_close = df['Close'].iloc[-1]
        return {h: last_close for h in horizons}

    # Prepare data for LSTM
    data_vals = dff['Close'].values
    X_data, y_data = [], []
    for i in range(lookback, len(data_vals)):
        X_data.append(data_vals[i-lookback:i])
        y_data.append(data_vals[i])

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Train/test split
    train_size = int(len(X_data)*0.8)
    X_train, y_train = X_data[:train_size], y_data[:train_size]
    X_test, y_test = X_data[train_size:], y_data[train_size:]

    # Reshape for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = Sequential()
    model.add(LSTM(32, input_shape=(lookback,1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test,y_test), verbose=0, callbacks=[es, rl])

    # Start forecasting from the last known window
    last_window = data_vals[-lookback:].reshape(1,lookback,1)
    preds = {}
    current_vals = last_window.copy()

    for h in horizons:
        steps = h
        temp_vals = current_vals.copy()
        for step in range(1, steps+1):
            pred = model.predict(temp_vals, verbose=0)[0][0]
            # Shift window
            temp_data = temp_vals.reshape(lookback,1)
            temp_data = np.append(temp_data[1:], [[pred]], axis=0)
            temp_vals = temp_data.reshape(1,lookback,1)

        preds[h] = pred

    return preds

def classify_signal(df, position_open):
    prophet_preds, prophet_acc = multi_horizon_forecast_with_accuracy_prophet(df, horizons=[8,16,24])
    xgb_preds = xgb_forecast(df, horizons=[8,16,24])
    lstm_preds = lstm_forecast(df, horizons=[8,16,24])

    # Combine predictions by averaging them
    p8 = (prophet_preds[8] + xgb_preds[8] + lstm_preds[8]) / 3
    p16 = (prophet_preds[16] + xgb_preds[16] + lstm_preds[16]) / 3
    p24 = (prophet_preds[24] + xgb_preds[24] + lstm_preds[24]) / 3
    accuracy = prophet_acc

    lookback = min(len(df), 48)
    volatility = df['Close'].tail(lookback).std() if lookback > 1 else 1
    predicted_price = p8

    last_row = df.iloc[-1]
    signal = last_row['Signal']
    rsi = last_row['RSI']

    tp_factor = 2.0
    sl_factor = 2.0

    signal_strength = {
        "Buy": "",
        "Sell": "",
        "Close position": "",
        "Prediction (8h)": f"${p8:.2f}",
        "Prediction (16h)": f"${p16:.2f}",
        "Prediction (24h)": f"${p24:.2f}",
        "Model Accuracy": f"{accuracy:.2f}%",
        "Take Profit": "",
        "Stop Loss": ""
    }

    if signal == 1:  # Bullish
        if rsi < 30:
            signal_strength["Buy"] = "Strong"
        else:
            signal_strength["Buy"] = "Potential"
        tp = predicted_price + (volatility * tp_factor)
        sl = predicted_price - (volatility * sl_factor)
        signal_strength["Take Profit"] = f"${tp:.2f}"
        signal_strength["Stop Loss"] = f"${sl:.2f}"

    elif signal == -1:  # Bearish
        if rsi > 70:
            signal_strength["Sell"] = "Strong"
        else:
            signal_strength["Sell"] = "Potential"
        if position_open:
            signal_strength["Close position"] = "Close Position"
        tp = predicted_price - (volatility * tp_factor)
        sl = predicted_price + (volatility * sl_factor)
        signal_strength["Take Profit"] = f"${tp:.2f}"
        signal_strength["Stop Loss"] = f"${sl:.2f}"

    else:  # Neutral
        if position_open:
            signal_strength["Close position"] = "Consider Close"
            tp = predicted_price + (volatility * 1.0)
            sl = predicted_price - (volatility * 1.0)
            signal_strength["Take Profit"] = f"${tp:.2f}"
            signal_strength["Stop Loss"] = f"${sl:.2f}"

    return signal_strength

signals_list = []
for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker], asset_class)
        if df.empty:
            continue
        df = generate_signals(df)
        position_open = st.session_state.open_positions[ticker] is not None
        classification = classify_signal(df, position_open)
        signals_list.append({
            "Symbol": ticker,
            "Buy": classification["Buy"],
            "Sell": classification["Sell"],
            "Close position": classification["Close position"],
            "Prediction (8h)": classification["Prediction (8h)"],
            "Prediction (16h)": classification["Prediction (16h)"],
            "Prediction (24h)": classification["Prediction (24h)"],
            "Model Accuracy": classification["Model Accuracy"],
            "Take Profit": classification["Take Profit"],
            "Stop Loss": classification["Stop Loss"]
        })

if signals_list:
    signals_df = pd.DataFrame(signals_list)
    st.dataframe(signals_df)
else:
    st.info("No signals available to display.")

st.markdown("---")
st.header("🔍 Trade Signals and Price Charts")
for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker], asset_class)
        if df.empty:
            st.warning(f"No data to display for {ticker}.")
            continue
        df = generate_signals(df)

        trades = [trade for trade in st.session_state.trade_history if trade['Ticker'] == ticker]
        position = st.session_state.open_positions[ticker]

        st.subheader(f"{ticker} Price Chart with Indicators and Signals")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_Short'], line=dict(width=1), name='MA Short'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_Long'], line=dict(width=1), name='MA Long'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'], line=dict(width=1, color='red'), name='BB Upper'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Middle'], line=dict(width=1, color='blue'), name='BB Middle'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'], line=dict(width=1, color='red'), name='BB Lower'
        ))

        fib_cols = [c for c in df.columns if 'Fibo_' in c]
        for fib_c in fib_cols:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[fib_c], line=dict(width=1, dash='dot'), name=fib_c
            ))

        if trades:
            buy_times = [pd.to_datetime(trade['Buy_Time']) for trade in trades]
            buy_prices = [trade['Buy_Price'] for trade in trades]
            sell_times = [pd.to_datetime(trade['Sell_Time']) for trade in trades]
            sell_prices = [trade['Sell_Price'] for trade in trades]

            fig.add_trace(go.Scatter(
                x=buy_times, y=buy_prices, mode='markers', marker_symbol='triangle-up', marker_color='green',
                marker_size=12, name='Buy Signal'
            ))
            fig.add_trace(go.Scatter(
                x=sell_times, y=sell_prices, mode='markers', marker_symbol='triangle-down', marker_color='red',
                marker_size=12, name='Sell Signal'
            ))

        if position:
            fig.add_trace(go.Scatter(
                x=[position['Buy_Time']], y=[position['Buy_Price']], mode='markers',
                marker_symbol='star', marker_color='gold', marker_size=15, name='Open Position'
            ))

        fig.update_layout(
            xaxis_title='Date/Time',
            yaxis_title='Price',
            height=600,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {ticker}.")

st.markdown("---")
st.markdown("<div style='text-align:center;'>© 2023 Advanced Trading Bot Dashboard | Powered by Streamlit</div>", unsafe_allow_html=True)
