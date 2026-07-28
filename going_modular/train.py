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

train_dataloader,test_dataloader ,input_feature=dataloader(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
X_batch, y_batch = next(iter(train_dataloader))

print("Features:", X_batch.shape)
print("Targets:", y_batch.shape)
# call the model 
#input_feature =X_train_tensor.shape[1]
hidden_feature =32
output_feature =1
model = Classgitmodel(input_features=input_feature,
                      hidden_features=hidden_feature,
                      output_feature=output_feature)
print(f"model :{model.state_dict()}")
loss_fn = nn.BCEWithLogitsLoss()
LEARNING_RATE =0.01
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Binary Classification ")
EPO=100
mlflow.enable_system_metrics_logging()
with mlflow.start_run():
    # lof the parameters
    mlflow.log_params({
        "EPOCHS":EPO,
        "OPTIMIZER ":optimizer.__class__.__name__,
        "LOSS_FUNCTION ":loss_fn.__class__.__name__,
        "MODEL_NAME":model.__class__.__name__,
        "LEARNING RATE":LEARNING_RATE,
        "INPUT_FEATURE ":input_feature,
        "HIDDEN_FEATURE":hidden_feature,
        "OUTPUT_FEATURE":output_feature,

    })
    result =train(model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  loss_fn=loss_fn,
                  optimizer =optimizer,epoch=EPO)
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="BINARY CLASSIFIER",  
        input_example=X_batch[:1].numpy(), 
        serialization_format="pickle", 
    )