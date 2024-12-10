import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
from datetime import datetime, timedelta
from prophet import Prophet  # Using Prophet for advanced forecasting

# Set page configuration first as recommended by Streamlit
st.set_page_config(
    page_title="Automated Trading Bot Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants for symbols
COMMODITIES = ["GC=F", "SI=F", "NG=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "BCH-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI"]

# API endpoints if needed (not currently used)
api_endpoints = {
    "cryptocurrencies": "https://financialmodelingprep.com/api/v3/symbol/available-cryptocurrencies",
    "forex": "https://financialmodelingprep.com/api/v3/symbol/available-forex-currency-pairs",
    "indexes": "https://financialmodelingprep.com/api/v3/symbol/available-indexes",
    "commodities": "https://financialmodelingprep.com/api/v3/symbol/available-commodities"
}

# Initialize session state for balance and trades
if 'initial_balance' not in st.session_state:
    st.session_state.initial_balance = 10000  # $10,000
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

# Title and description
st.title("🚀 Automated Trading Bot Dashboard")
st.markdown("""
Welcome to the professional trading bot dashboard. This tool analyzes top-performing commodities, forex pairs, indices, and cryptocurrencies to provide trading signals and execute trades based on advanced strategies.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select Asset Class", ["Forex", "Commodities", "Indices", "Cryptocurrency"])

# Initialize tickers and asset_class to avoid reference errors
tickers = []
asset_class = None

# Define tickers based on selection
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

# Validate tickers
if not tickers:
    st.error(f"No tickers defined for section: {section}")
    st.stop()

st.write(f"Selected Asset Class: {asset_class}")
st.write(f"Tickers: {tickers}")

# Allocate capital per ticker
num_tickers = len(tickers)
capital_per_ticker = st.session_state.balance / num_tickers

# Initialize allocated capital and open positions
for ticker in tickers:
    if ticker not in st.session_state.allocated_capital:
        st.session_state.allocated_capital[ticker] = capital_per_ticker
    if ticker not in st.session_state.open_positions:
        st.session_state.open_positions[ticker] = None  # None means no open position

def fetch_live_data(tickers, asset_class):
    data = {}
    api_key = os.getenv("FMP_API_KEY")

    if not api_key:
        st.error("API key not found. Set 'FMP_API_KEY'.")
        return data

    for ticker in tickers:
        try:
            ticker_api = ticker.replace('/', '')

            # Different endpoints based on asset_class
            if asset_class == 'Indices':
                url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker_api}?timeseries=30&apikey={api_key}'
            else:
                url = f'https://financialmodelingprep.com/api/v3/historical-chart/5min/{ticker_api}?apikey={api_key}'

            response = requests.get(url)
            response.raise_for_status()
            data_json = response.json()

            if not data_json:
                st.warning(f"No data returned for {ticker}.")
                continue

            if asset_class == 'Indices':
                df = pd.DataFrame(data_json.get('historical', []))
            else:
                df = pd.DataFrame(data_json)

            if df.empty:
                st.warning(f"No data available for {ticker}.")
                continue

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
            df.sort_index(inplace=True)

            if asset_class == 'Indices':
                df = df[df.index >= (datetime.utcnow() - timedelta(days=30))]
            else:
                df = df[df.index >= (datetime.utcnow() - timedelta(days=2))]

            if df.empty:
                st.warning(f"No recent data for {ticker}.")
                continue

            data[ticker] = df
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

def compute_indicators(df, asset_class):
    df = df.copy()
    if asset_class == 'Indices':
        df['MA_Short'] = df['Close'].rolling(window=5).mean()
        df['MA_Long'] = df['Close'].rolling(window=20).mean()
    else:
        df['MA_Short'] = df['Close'].rolling(window=10).mean()
        df['MA_Long'] = df['Close'].rolling(window=30).mean()
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    return df

def generate_signals(df):
    df['Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'Signal'] = 1
    df.loc[df['MA_Short'] < df['MA_Long'], 'Signal'] = -1
    return df

def simulate_trades_live(data):
    for ticker in tickers:
        if ticker in data:
            df = compute_indicators(data[ticker], asset_class)
            df = df.dropna()

            if df.empty:
                st.warning(f"No data for {ticker}.")
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
                    # Execute Buy
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
                # Check if 10% profit or sell signal
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
        else:
            st.warning(f"No data for {ticker}.")
    return

# Fetch live data
data = fetch_live_data(tickers, asset_class)

if not data:
    st.error("No data fetched. Check tickers and API.")
    st.stop()

# Simulate trades
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
        win_rate = (winning_trades / num_trades) * 100
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
        if position:
            if ticker in data and not data[ticker].empty:
                current_price = data[ticker]['Close'][-1]
                profit_loss = (current_price - position['Buy_Price']) * position['Quantity']
                open_positions_list.append({
                    'Ticker': ticker,
                    'Buy_Time': position['Buy_Time'].strftime('%Y-%m-%d %H:%M'),
                    'Buy_Price': position['Buy_Price'],
                    'Current_Price': current_price,
                    'Profit/Loss': profit_loss
                })
            else:
                st.warning(f"No current price data available for {ticker}.")
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
# Adjusted SECTION: Advanced Forecasting + Revised TP/SL
#############################################
st.header("📊 Signals and Predictions")

def forecast_next_day_price(df):
    if df.empty or len(df) < 10:
        return df['Close'][-1] if not df.empty else None
    
    df_daily = df.resample('D').last().dropna(subset=['Close'])
    if len(df_daily) < 10:
        return df_daily['Close'][-1] if not df_daily.empty else None
    
    prophet_df = df_daily.reset_index()[['date','Close']]
    prophet_df = prophet_df.rename(columns={'date':'ds','Close':'y'})
    
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
    m.fit(prophet_df)
    
    future = m.make_future_dataframe(periods=1)
    forecast = m.predict(future)
    predicted_price = forecast.iloc[-1]['yhat']
    return predicted_price

def classify_signal(df, position_open):
    row = df.iloc[-1]
    signal = row['Signal']
    rsi = row['RSI']
    price = row['Close']
    
    predicted_price = forecast_next_day_price(df)
    if predicted_price is None:
        predicted_price = price  # fallback
    
    signal_strength = {
        "Buy": "",
        "Sell": "",
        "Close": "",
        "Prediction": f"${predicted_price:.2f}",
        "Take Profit": "",
        "Stop Loss": ""
    }

    # Define small, closer increments around the predicted price
    # For Buy: TP at predicted_price *1.02, SL at predicted_price*0.98
    # For Sell: TP at predicted_price *0.98, SL at predicted_price*1.02 (since we profit if price goes down)
    # For Neutral with position: narrower band around predicted price, e.g. ±1%

    if signal == 1:
        # Bullish
        if rsi < 30:
            signal_strength["Buy"] = "Strong"
        else:
            signal_strength["Buy"] = "Potential"
        # Set TP & SL around predicted price
        tp = predicted_price * 1.02
        sl = predicted_price * 0.98
        signal_strength["Take Profit"] = f"${tp:.2f}"
        signal_strength["Stop Loss"] = f"${sl:.2f}"

    elif signal == -1:
        # Bearish
        if rsi > 70:
            signal_strength["Sell"] = "Strong"
        else:
            signal_strength["Sell"] = "Potential"
        if position_open:
            signal_strength["Close"] = "Close Position"
        # For a sell, TP if price goes down (predicted_price *0.98), SL if price goes up (predicted_price*1.02)
        tp = predicted_price * 0.98
        sl = predicted_price * 1.02
        signal_strength["Take Profit"] = f"${tp:.2f}"
        signal_strength["Stop Loss"] = f"${sl:.2f}"

    else:
        # Neutral
        if position_open:
            signal_strength["Close"] = "Consider Close"
            # Narrow band around predicted price ±1%
            tp = predicted_price * 1.01
            sl = predicted_price * 0.99
            signal_strength["Take Profit"] = f"${tp:.2f}"
            signal_strength["Stop Loss"] = f"${sl:.2f}"
        else:
            # No position open, no strong signal
            signal_strength["Buy"] = ""
            signal_strength["Sell"] = ""
            signal_strength["Take Profit"] = ""
            signal_strength["Stop Loss"] = ""

    return signal_strength

signals_list = []
for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker], asset_class)
        df = df.dropna()
        if df.empty:
            continue
        df = generate_signals(df)
        position_open = st.session_state.open_positions[ticker] is not None
        classification = classify_signal(df, position_open)
        signals_list.append({
            "Symbol": ticker,
            "Buy": classification["Buy"],
            "Sell": classification["Sell"],
            "Close position": classification["Close"],
            "Prediction": classification["Prediction"],
            "Take Profit": classification["Take Profit"],
            "Stop Loss": classification["Stop Loss"]
        })

if signals_list:
    signals_df = pd.DataFrame(signals_list)
    st.dataframe(signals_df)
else:
    st.info("No signals available to display.")

#############################################
# END OF UPDATED SECTION
#############################################

st.markdown("---")

st.header("🔍 Trade Signals and Price Charts")
for ticker in tickers:
    if ticker in data:
        df = compute_indicators(data[ticker], asset_class)
        df = df.dropna()
        if df.empty:
            st.warning(f"No data to display for {ticker}.")
            continue
        df = generate_signals(df)

        # Find buy and sell points from history
        trades = [trade for trade in st.session_state.trade_history if trade['Ticker'] == ticker]
        position = st.session_state.open_positions[ticker]

        st.subheader(f"{ticker} Price Chart with Trade Signals")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_Short'], line=dict(width=1), name='MA Short'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MA_Long'], line=dict(width=1), name='MA Long'
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
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {ticker}.")

st.markdown("---")
st.markdown("<div style='text-align:center;'>© 2023 Trading Bot Dashboard | Powered by Streamlit</div>", unsafe_allow_html=True)
