Fraud Detection — Binary Classification with PyTorch

A modular PyTorch pipeline for detecting fraudulent transactions, built to practice the full ML lifecycle: data cleaning → preprocessing → class-imbalance handling → training with hyperparameter search → experiment tracking → model registry → inference on new data.

This project is a learning exercise focused on doing the pipeline properly, not just training a model in a notebook — with a real train/serve split, no data leakage, and full experiment tracking via MLflow.

Problem

Binary classification: predict whether a transaction/customer record is fraudulent (Y target column). The dataset is imbalanced (fraud cases are the minority class), which shapes several of the design choices below.

Project structure
going_modular/
├── datasetup.py        # Download data, data-quality checks, train/test split + scaling + balancing
├── imbalance.py         # Oversampling / undersampling of the training set only
├── dataloader.py         # Wraps train/test tensors into PyTorch DataLoaders
├── model.py             # Simple feed-forward binary classifier (nn.Module)
├── engine.py            # Train/eval loop, tracks best epoch by test accuracy
├── train.py              # End-to-end training run: grid search + MLflow tracking + model registry
├── prep_pred_data.py    # Applies the *exact* training preprocessing to new data
├── predict.py             # Loads the registered "champion" model and scores new records
├── trian_raytunr.py     # WIP: Ray Tune hyperparameter search (run separately in Colab — see below)
├── artifacts/            # Saved scaler + feature column order (for consistent inference)
└── data/                 # Fraud.csv, new_customers.csv, predictions.csv
How it works
datasetup.py downloads the dataset, runs basic data-quality checks (missing values, duplicates, class distribution), drops irrelevant columns, splits into train/test before any scaling or balancing (to avoid leakage), then fits a StandardScaler on the training set only.
imbalance.py oversamples the minority class — applied only to the training set, never the test set.
dataloader.py converts the processed train/test data into PyTorch DataLoaders.
model.py defines a small feed-forward network (Linear → ReLU → Linear) for binary classification.
train.py runs a grid search over learning rate and hidden layer size, logs every trial (params + metrics) to MLflow as a nested run, tracks the best-performing model by test accuracy, and registers the winning model in the MLflow Model Registry under a champion alias.
prep_pred_data.py + predict.py load the registered champion model and the same scaler/feature-column order used in training, so new data is transformed identically to how the model was trained — avoiding train/serve skew.
Experiment tracking

All runs are tracked with MLflow (mlflow.start_run, nested runs per grid search trial, mlflow.log_params/log_metrics, and model registration with set_registered_model_alias). To view runs locally:

bash
mlflow ui --backend-store-uri <your tracking uri>
Setup
bash
pip install -r requirements.txt

Run training:

bash
cd going_modular
python train.py

Run inference on new data:

bash
python predict.py
Hyperparameter search — current status

train.py currently does an exhaustive grid search over a small space (learning_rate, hidden_features). I'm extending this with Ray Tune for a larger search space and early-stopping (ASHA scheduler) — Ray Tune doesn't install cleanly on native Windows, so that part is being developed and run separately in Google Colab (trian_raytunr.py is the work-in-progress local stub for this; the actual search runs in Colab).

Deployment (in progress)

Currently, inference only happens via predict.py run against a local CSV. Planned deployment layer:

FastAPI — a /predict endpoint that loads the registered champion model from MLflow and scores incoming transaction records in real time, reusing the existing prep_pred_data.py preprocessing so the API applies identical transformations to what the model was trained on.
Streamlit — a simple front end on top of the FastAPI service for uploading a CSV (or entering a single record) and viewing fraud probability/predicted class, so the model is usable without touching code.
Known limitations / next steps
 Build the FastAPI inference service (/predict endpoint).
 Build the Streamlit front end on top of the API.
 Add precision, recall, F1, and ROC-AUC — accuracy alone is a weak metric for an imbalanced fraud dataset and is currently the only metric tracked.
 Add unit tests for the preprocessing and inference functions.
 Move the hardcoded local MLflow tracking URI into a config/env variable.
 Compare the neural network against a gradient-boosted tree baseline (e.g. XGBoost), which is the standard approach for tabular fraud data.
 Finish the Ray Tune hyperparameter search (currently in progress in Colab).
Dataset

UCI Machine Learning Repository — fraud detection dataset (downloaded automatically by datasetup.py).
