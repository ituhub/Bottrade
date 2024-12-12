import os
import joblib

# Step 1: Load all models from the folder
models_folder = "models"
models_dict = {}

for file_name in os.listdir(models_folder):
    if file_name.endswith(".pkl"):  # Only process .pkl files
        model_path = os.path.join(models_folder, file_name)
        model_name = os.path.splitext(file_name)[0]  # Use file name (without extension) as key
        models_dict[model_name] = joblib.load(model_path)

print("Loaded models:", models_dict.keys())

# Step 2: Save all models into one file
combined_file = "combined_models.pkl"
joblib.dump(models_dict, combined_file)
print(f"All models saved into {combined_file}")

# Step 3: Load the combined models later
loaded_models = joblib.load(combined_file)
print("Loaded models from combined file:", loaded_models.keys())
