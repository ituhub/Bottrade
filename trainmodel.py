import os
import requests
import pandas as pd
import numpy as np
from datetime import timedelta
from prophet import Prophet
from xgboost import XGBRegressor
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error

#############################################
# Define Asset Classes and Tickers
#############################################
COMMODITIES = ["GC=F", "SI=F", "NG=F", "KC=F"]
FOREX_SYMBOLS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "DOT-USD", "LTC-USD"]
INDICES_SYMBOLS = ["^GSPC", "^GDAXI", "^HSI", "000300.SS"]

# Combine all tickers into one list
ALL_TICKERS = FOREX_SYMBOLS + COMMODITIES + CRYPTO_SYMBOLS + INDICES_SYMBOLS

#############################################
# Retrieve API Key from Environment Variable
#############################################
api_key = os.getenv("FMP_API_KEY")

if not api_key:
    raise ValueError("No API key found. Please set the FMP_API_KEY environment variable before running the script.")

#############################################
# Helper Functions
#############################################
def fetch_data(ticker, api_key):
    # Replace "/" in the ticker if any
    ticker_api = ticker.replace("/", "")
    url = f"https://financialmodelingprep.com/api/v3/historical-chart/1hour/{ticker_api}?apikey={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {response.text}")
    data_json = response.json()
    if not data_json or len(data_json) < 50:
        raise RuntimeError(f"Not enough data for {ticker}")

    df = pd.DataFrame(data_json)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
    df.sort_index(inplace=True)
    df = df.dropna()
    return df

def create_xgb_features(data, lags=6):
    dff = data.copy()
    for i in range(1, lags+1):
        dff[f'Close_lag_{i}'] = dff['Close'].shift(i)
    dff = dff.dropna()
    feature_cols = [f'Close_lag_{i}' for i in range(1, lags+1)]
    X = dff[feature_cols]
    y = dff['Close']
    return X, y

def create_lstm_dataset(series, lookback=24):
    values = series.values
    X_data, y_data = [], []
    for i in range(lookback, len(values)):
        X_data.append(values[i-lookback:i])
        y_data.append(values[i])
    return np.array(X_data), np.array(y_data)

#############################################
# Training Loop for All Tickers
#############################################
for ticker in ALL_TICKERS:
    print(f"Processing {ticker}...")
    try:
        # Fetch data
        df = fetch_data(ticker, api_key)
        if len(df) < 200:
            print(f"Skipping {ticker}: not enough data after cleaning.")
            continue

        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in df.columns for col in required_cols):
            print(f"Skipping {ticker}: Missing one of the required columns (Open, High, Low, Close).")
            continue

        # Prophet Model
        prophet_df = df[['Close']].reset_index().rename(columns={'date':'ds','Close':'y'}).sort_values('ds')
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(prophet_df)
        with open(f'prophet_model_{ticker}.pkl', 'wb') as f:
            pickle.dump(m, f)

        # XGBoost Model
        X, y = create_xgb_features(df, lags=6)
        if len(X) < 100:
            print(f"Skipping XGBoost for {ticker}: Not enough data for features.")
        else:
            train_size = int(len(X)*0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
            print(f"{ticker} XGB MAE: {xgb_mae}")
            with open(f'xgb_model_{ticker}.pkl', 'wb') as f:
                pickle.dump(xgb_model, f)

        # LSTM Model
        lookback = 24
        X_lstm_full, y_lstm_full = create_lstm_dataset(df['Close'], lookback=lookback)
        if len(X_lstm_full) < 100:
            print(f"Skipping LSTM for {ticker}: Not enough data for LSTM.")
        else:
            lstm_train_size = int(len(X_lstm_full)*0.8)
            X_lstm_train, X_lstm_test = X_lstm_full[:lstm_train_size], X_lstm_full[lstm_train_size:]
            y_lstm_train, y_lstm_test = y_lstm_full[:lstm_train_size], y_lstm_full[lstm_train_size:]

            X_lstm_train = X_lstm_train.reshape(X_lstm_train.shape[0], lookback, 1)
            X_lstm_test = X_lstm_test.reshape(X_lstm_test.shape[0], lookback, 1)

            lstm_model = Sequential()
            lstm_model.add(LSTM(32, input_shape=(lookback,1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))

            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

            lstm_model.fit(X_lstm_train, y_lstm_train, epochs=20, batch_size=16, 
                           validation_data=(X_lstm_test,y_lstm_test),
                           callbacks=[es, rl], verbose=0)

            y_pred_lstm = lstm_model.predict(X_lstm_test)
            lstm_mae = mean_absolute_error(y_lstm_test, y_pred_lstm)
            print(f"{ticker} LSTM MAE: {lstm_mae}")

            lstm_model.save(f'lstm_model_{ticker}.h5')

        print(f"{ticker} models trained and saved successfully!")

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        continue

print("All tickers processed.")
