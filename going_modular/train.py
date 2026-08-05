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


#----------------------------------------------------------------------------
#                    CONFIGURATION 
#----------------------------------------------------------------------------
TRACKING_URI ="http://127.0.0.1:5000"
EXPERIMENT_NAME ="BINARY _CLASSIFICATION(FRAUD_DETECTION )"
REGISTERED_MODEL_NAME ="PYTORCH_MODEL1"
device ="cuda" if torch.cuda.is_available() else "cpu"

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.enable_system_metrics_logging()
client =MlflowClient()
#----------------------------------------------------------------------------
#                    HYPERPARAMETER TUNING 
# PSEUDOCODE
# DEFINE THE SEARCH SPACE (LETS START SMALL WITH TWO )
# import itertool 
# THE ITERTOOLS.PRODUCT FOR THE SEARCH SPACE
#----------------------------------------------------------------------------
# DEFINING THE SEARCH SPACE
search_space =({
      "learning_rate":[0.01,0.1],
      "hidden_features":[16,32]
})
hyper_combo =itertools.product(search_space["learning_rate"],search_space["hidden_features"]
                               )

#----------------------------------------------------------------------------
#                   DEFINE THE VARIABLES FOR THE BEST MODEL 
#----------------------------------------------------------------------------

best_accuracy =float("-inf")
best_parameters =None
best_model_state=None
best_results=None
best_run_id=None
best_epoch=None
#----------------------------------------------------------------------------
#                DEFINING THE PARENT RUN MLFLOW
#set the tags,
#calculate number of trials and 
#log teh param 
#----------------------------------------------------------------------------
with mlflow.start_run(run_name ="Binary_classification_GRID_SEARCH",log_system_metrics=True) as parent_run:
      num_of_trials =(len(search_space["learning_rate"])*len(search_space["hidden_features"]))
      mlflow.set_tags({
        "run_type":"hyper_parameter tuning",
        "search_method":"grid search",
        "framework":"PyTorch",
        "problem_type":"binary_classification"
      })
      mlflow.log_param("NUMBER_OF_TRIALS",num_of_trials)   
      #-----------------------------------------------------------    -----------------
      #                   loop through every combinations 

      #----------------------------------------------------------------------------
      for trial_no,(learning_rate,hidden_feature) in enumerate(hyper_combo,start =1):
            with mlflow.start_run(run_name=f"trial_{trial_no}",nested =True) as child_run:
                   #-----------------------------------------------------------    -----------------
                   #                 Create new dataloader 
                  
                   #----------------------------------------------------------------------------
                  train_dataloader,test_dataloader ,input_feature=dataloader(X_train=X_train,
                                                                             y_train=y_train,X_test=X_test,
                                                                             y_test=y_test)
                  X_batch, y_batch = next(iter(train_dataloader))
                  print("Features:", X_batch.shape)
                  print("Targets:", y_batch.shape)
                  #-----------------------------------------------------------    -----------------
                  #                Create new model
                                    
                  #----------------------------------------------------------------------------
                  output_feature =1
                  model = Classgitmodel(input_features=input_feature,
                      hidden_features=hidden_feature,
                      output_feature=output_feature).to(device)
                  #print(f"model :{model.state_dict()}")
                  loss_fn = nn.BCEWithLogitsLoss()
                  optimizer = torch.optim.Adam(    model.parameters(),    lr=learning_rate)
                  #-----------------------------------------------------------    -----------------
                  #                                   log the params
                  #---------------------------------------------------------------------------
                  EPO =100
                  mlflow.log_params({
                        "TRAIL_NUMBER":trial_no,
                        "EPOCHS":EPO,
                        "OPTIMIZER ":optimizer.__class__.__name__,
                        "LOSS_FUNCTION ":loss_fn.__class__.__name__,
                        "MODEL_NAME":model.__class__.__name__,
                        "LEARNING RATE":learning_rate,
                        "INPUT_FEATURE ":input_feature,
                        "HIDDEN_FEATURE":hidden_feature,
                        "OUTPUT_FEATURE":output_feature,
                        })
                  trial_results = train(
                                model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                epoch=EPO,
                                device=device
                                
                                )
                  ## log in the final values but the last one 
                  trial_best_accuracy =trial_results["best_test_accuracy"]
                  trial_best_loss =trial_results["best_test_loss"]
                  trial_best_epoch=trial_results["best_epoch"]
                  #-----------------------------------------------------------
                  # LOG THE RECIVED METRICS
                  #------------------------------------------------------------


                  mlflow.log_params({"learning_rate": learning_rate,
                                     "hidden_features": hidden_feature
                                     })
                  mlflow.log_metrics({
                          "best_test_accuracy":trial_best_accuracy ,
                          "best_test_loss": trial_best_loss,
                          "best_epoch":trial_best_epoch
                          })

                  
                  if trial_best_accuracy > best_accuracy:
                       best_accuracy = trial_best_accuracy
                       best_run_id = child_run.info.run_id
                       best_model_state=trial_results["best_model_state"]
                       best_parameters = {
                             "learning_rate": learning_rate,
                             "hidden_features": hidden_feature,
                             "input_features":input_feature,
                             "output_features":output_feature
                               }
                       best_epoch =trial_results["best_epoch"]
    
    # ----------------------------------------------------------------
    # LOG OVERALL BEST GRID-SEARCH RESULT TO THE PARENT RUN
    # ----------------------------------------------------------------

      mlflow.log_metric(
        "best_validation_accuracy",
        best_accuracy
    )

      mlflow.log_metric(
        "best_epoch",
        best_epoch
    )

      mlflow.log_params({
        "best_learning_rate":
            best_parameters["learning_rate"],

        "best_hidden_features":
            best_parameters["hidden_features"],

        "best_child_run_id":
            best_run_id
    })

    # ----------------------------------------------------------------
    # REBUILD THE BEST MODEL
    # ----------------------------------------------------------------

      best_model = Classgitmodel(
        input_features=best_parameters["input_features"],
        hidden_features=best_parameters["hidden_features"],
        output_feature=best_parameters["output_features"]
    ).to(device)

      best_model.load_state_dict(
        best_model_state
    )
      best_model.eval()

    # ----------------------------------------------------------------
    # CREATE SIGNATURE USING THE ACTUAL BEST MODEL
    # ----------------------------------------------------------------

      input_tensor = X_batch[:4].float().to(device)
      with torch.inference_mode():
                prediction_tensor = best_model(
                    input_tensor
                )

      input_signature_array = (
                input_tensor
                .detach()
                .cpu()
                .numpy()
            )

      output_signature_array = (
                prediction_tensor
                .detach()
                .cpu()
                .numpy()
            )

      signature = infer_signature(
                model_input=input_signature_array,
                model_output=output_signature_array
            )

    # ----------------------------------------------------------------
    # LOG THE ACTUAL BEST MODEL
    # ----------------------------------------------------------------

      model_info = mlflow.pytorch.log_model(
        pytorch_model=best_model,
        name="binary_classifier",
        input_example=input_signature_array[:1],
        signature=signature,
        serialization_format="pickle",
        registered_model_name=REGISTERED_MODEL_NAME
    )
      client.set_registered_model_alias(   name=REGISTERED_MODEL_NAME ,
    alias="champion",
    version=model_info.registered_model_version,
)

      print("Best accuracy:", best_accuracy)
      print("Best epoch:", best_epoch)
      print("Best parameters:", best_parameters)
      print("Best child run ID:", best_run_id)
      print("BEST modelURI:",f"models/{REGISTERED_MODEL_NAME}@champion")




                  
                  
               
                        



                  

    

















  

  


