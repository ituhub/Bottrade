# train_models.py
import pandas as pd
import pickle
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load your historical data (e.g. from CSV)
df = pd.read_csv('historical_data.csv', parse_dates=True, index_col='date')

# ... do feature engineering ...
# X, y = prepare_features(df)

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
xgb_model.fit(X, y)
with open('xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Train LSTM
lookback = 24
# X_lstm, y_lstm = prepare_lstm_features(df, lookback)

model = Sequential()
model.add(LSTM(32, input_shape=(lookback,1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, validation_split=0.2)
model.save('lstm_model.h5')
