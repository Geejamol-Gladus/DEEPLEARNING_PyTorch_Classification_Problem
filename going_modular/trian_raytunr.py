import pandas as pd
from datasetup import data_download
from datasetup import check_data_quality
from datasetup import split_balance
import torch
from torch import nn
from model import Classgitmodel
from dataloader import dataloader
from engine import train
import mlflow 
from mlflow.models  import infer_signature
import itertools
from mlflow import MlflowClient


data_path = data_download(
    url="https://archive.ics.uci.edu/static/public/350/data.csv",
    data_dir="data/",
    filename="Fraud.csv"
)
data =pd.read_csv(data_path )
print(data.head())

cleaned_dataframe = check_data_quality(
    dataframe=data,
    target_column="Y",
    irrelevant_columns=["ID"]
)
print(cleaned_dataframe.head())


# 3. Split the data and balance only the training set
X_train, X_test, y_train, y_test = split_balance(
    cleaned_dataframe=cleaned_dataframe,target ="Y")
print(f"the shape of X_train :{X_train.shape}")
print(f"the shape of y_train: {y_train.shape}")
print(f"the shape of X-test:{X_test.shape}")
print(f"the shape of X-test:{y_test.shape}")

#-------------------SET TRACKING INFORMATION -------------------------------------------#

TRACKING_URI ="http://127.0.0.1:5000"
EXPERIMENT_NAME ="RAY_TUNE_EXPERIEMNT"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)