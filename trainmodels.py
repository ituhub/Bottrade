import os
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime
from prophet import Prophet
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Define your tickers
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

ALL_TICKERS = COMMODITIES + FOREX_SYMBOLS + CRYPTO_SYMBOLS + INDICES_SYMBOLS

# Function to sanitize ticker symbols
def sanitize_ticker(ticker):
    return ticker.replace('=', '_').replace('/', '_').replace('^', '').replace('.', '_')

# Ensure the models directory exists
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Fetch data function
def fetch_data(ticker, api_key):
    try:
        ticker_api = ticker.replace('/', '')
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/15min/{ticker_api}?apikey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data_json = response.json()

        if not data_json or len(data_json) < 1:
            print(f"No data returned for {ticker}.")
            return pd.DataFrame()

        df = pd.DataFrame(data_json)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)

        df_hourly = df.resample('H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
        df_hourly.dropna(subset=['Close'], inplace=True)
        return df_hourly
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()

# Function to compute RSI
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to compute MACD
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
    for k, v in levels.items():
        df[k] = v
    return df

# Function to compute indicators
def compute_indicators(df):
    df['RSI'] = compute_RSI(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_MACD(df['Close'])
    df = compute_bollinger_bands(df)
    df = compute_fibonacci_levels(df)
    return df

# Function to create features for XGBoost
def create_xgb_features(df):
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    for lag in range(1, 7):
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    if 'BB_Middle' in df.columns:
        df['Close_minus_BB_Mid'] = df['Close'] - df['BB_Middle']
    if 'Fibo_50.0' in df.columns:
        df['Close_minus_Fibo_50'] = df['Close'] - df['Fibo_50.0']

    df.dropna(inplace=True)
    return df

# Main training loop
api_key = os.getenv("FMP_API_KEY")

if not api_key:
    print("API key not found. Please set the 'FMP_API_KEY' environment variable.")
    exit(1)

for ticker in ALL_TICKERS:
    print(f"Processing {ticker}...")
    sanitized_ticker = sanitize_ticker(ticker)
    try:
        df = fetch_data(ticker, api_key)
        if df.empty:
            print(f"No data fetched for {ticker}. Skipping.")
            continue

        df = compute_indicators(df)
        df.dropna(inplace=True)
        if df.empty:
            print(f"Not enough data after computing indicators for {ticker}. Skipping.")
            continue

        # Prophet Model
        m = Prophet()
        df_prophet = df.reset_index()[['date', 'Close']].rename(columns={'date': 'ds', 'Close': 'y'})
        m.fit(df_prophet)

        with open(os.path.join(models_dir, f'prophet_model_{sanitized_ticker}.pkl'), 'wb') as f:
            pickle.dump(m, f)

        # XGBoost Model
        df_xgb = create_xgb_features(df)
        features = [col for col in df_xgb.columns if col not in ['Open', 'High', 'Low', 'Close']]
        X = df_xgb[features]
        y = df_xgb['Close']

        if len(X) < 50:
            print(f"Not enough data for XGBoost model for {ticker}. Skipping.")
            continue

        xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        xgb_model.fit(X, y)

        with open(os.path.join(models_dir, f'xgb_model_{sanitized_ticker}.pkl'), 'wb') as f:
            pickle.dump(xgb_model, f)

        # LSTM Model
        data = df['Close'].values
        lookback = 24

        if len(data) < lookback + 10:
            print(f"Not enough data for LSTM model for {ticker}. Skipping.")
            continue

        X_lstm = []
        y_lstm = []
        for i in range(lookback, len(data)):
            X_lstm.append(data[i - lookback:i])
            y_lstm.append(data[i])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        X_lstm = X_lstm.reshape(X_lstm.shape[0], X_lstm.shape[1], 1)

        lstm_model = Sequential()
        lstm_model.add(LSTM(64, input_shape=(X_lstm.shape[1], 1)))
        lstm_model.add(Dense(1))
        lstm_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stopping = EarlyStopping(monitor='loss', patience=5)
        lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, verbose=0, callbacks=[early_stopping])

        lstm_model.save(os.path.join(models_dir, f'lstm_model_{sanitized_ticker}.h5'))

        print(f"{ticker} models trained and saved successfully!")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

print("All tickers processed.")
