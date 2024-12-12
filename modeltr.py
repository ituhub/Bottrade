import os
from prophet import Prophet
from prophet.serialize import model_from_json
from xgboost import XGBRegressor
from tensorflow.keras.models import load_model

# Step 1: Load all models from the folder
models_folder = "models"
models_dict = {}

for file_name in os.listdir(models_folder):
    model_path = os.path.join(models_folder, file_name)
    if file_name.startswith('prophet_model_') and file_name.endswith('.json'):
        ticker = file_name[len('prophet_model_'):-len('.json')]
        with open(model_path, 'r') as f:
            model = model_from_json(f.read())
        models_dict[f'prophet_{ticker}'] = model
    elif file_name.startswith('xgb_model_') and file_name.endswith('.json'):
        ticker = file_name[len('xgb_model_'):-len('.json')]
        xgb_model = XGBRegressor()
        xgb_model.load_model(model_path)
        models_dict[f'xgb_{ticker}'] = xgb_model
    elif file_name.startswith('lstm_model_') and file_name.endswith('.h5'):
        ticker = file_name[len('lstm_model_'):-len('.h5')]
        lstm_model = load_model(model_path)
        models_dict[f'lstm_{ticker}'] = lstm_model

print("Loaded models:", models_dict.keys())

# You can now use `models_dict` as needed in your application
