import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from prep_pred_data import prepare_new_data


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "PYTORCH_MODEL1"

SCALER_PATH = "artifacts/scaler.joblib"
FEATURE_COLUMNS_PATH = "artifacts/feature_columns.json"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# CONNECT TO MLFLOW
# ------------------------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)


# ------------------------------------------------------------
# LOAD THE CHAMPION MODEL
# ------------------------------------------------------------

model_uri = (
    f"models:/{REGISTERED_MODEL_NAME}@champion"
)

loaded_model = mlflow.pytorch.load_model(
    model_uri,
    map_location=device
)

loaded_model.eval()


# ------------------------------------------------------------
# LOAD NEW DATA
# ------------------------------------------------------------

# Example: CSV containing new customers
new_data = pd.read_csv(
    "data/new_customers.csv"
)

print("Raw new data:")
print(new_data.head())


# ------------------------------------------------------------
# APPLY TRAINING PREPROCESSING
# ------------------------------------------------------------

input_tensor = prepare_new_data(
    new_data=new_data,
    scaler_path=SCALER_PATH,
    feature_column_path=FEATURE_COLUMNS_PATH,
    irrelevant_column=["ID"]
).to(device)


# ------------------------------------------------------------
# MAKE PREDICTIONS
# ------------------------------------------------------------

with torch.inference_mode():

    logits = loaded_model(
        input_tensor
    )

    # Handle model output [N, 1]
    logits = logits.squeeze(dim=1)

    probabilities = torch.sigmoid(
        logits
    )

    predicted_classes = (
        probabilities >= 0.5
    ).int()


# ------------------------------------------------------------
# ADD RESULTS TO THE DATAFRAME
# ------------------------------------------------------------

prediction_output = new_data.copy()

prediction_output["fraud_probability"] = (
    probabilities
    .detach()
    .cpu()
    .numpy()
)

prediction_output["predicted_class"] = (
    predicted_classes
    .detach()
    .cpu()
    .numpy()
)

print("\nPredictions:")
print(prediction_output.head())


# Optional: save predictions
prediction_output.to_csv(
    "data/predictions.csv",
    index=False
)