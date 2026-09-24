Credit Card Default Prediction with PyTorch and MLflow

A binary classification pipeline in PyTorch that predicts whether a credit card client will default on their next payment, with experiment tracking and model registry in MLflow.

Dataset: Default of Credit Card Clients, UCI Machine Learning Repository. 30,000 clients, 23 features. Downloaded automatically.

Highlights
No data leakage: data is split first, the scaler is fitted on training data only, and oversampling is applied to the training set only.
Same preprocessing for training and prediction: the fitted scaler and feature order are saved and reused at inference time.
Best-epoch checkpointing: each trial keeps its best weights, not just the last epoch.
Grid search with MLflow: one parent run with a nested run per trial, logging parameters, metrics and artifacts.
Model registry: the best model is registered with a signature and loaded for prediction by its champion alias.
Model

Linear(23 → hidden) → ReLU → Linear(hidden → 1), trained with BCEWithLogitsLoss and Adam.

How to run
bash
pip install -r requirements.txt
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
cd going_modular
python train.py      # train, tune and register the best model
python predict.py    # predict on data/new_customers.csv
Next steps

Add a validation split, report precision, recall, F1 and ROC-AUC, and serve the model with FastAPI.
