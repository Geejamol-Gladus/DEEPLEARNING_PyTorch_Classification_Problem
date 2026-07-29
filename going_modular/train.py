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
hyper_combo =itertools.product(search_space["hidden_features"],
                               search_space["learning_rate"])

#----------------------------------------------------------------------------
#                   DEFINE THE VARIABLES FOR THE BEST MODEL 
#----------------------------------------------------------------------------

best_accuracy =float("-inf")
best_parameters =None
best_model_state=None
best_results=None
best_run_id=None
#----------------------------------------------------------------------------
#                DEFINING THE PARENT RUN MLFLOW
#set the tags,
#calculate number of trials and 
#log teh param 
#----------------------------------------------------------------------------
with mlflow.start_run(run_name ="Binary_classification_GRID_SEARCH",log_system_metrics=True) as parent_run:
      num_of_trials =(len(search_space["hidden_features"])*len(search_space["learning_rate"]))
      mlflow.set_tags({
        "run_type":"hyper_parameter tuning",
        "search_method":"grid search",
        "framework":"PyTorch",
        "problem_type":"binary_classification"
      })
      mlflow.log_param("NUMBER _OF_TRIALS",num_of_trials)   
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
                      output_feature=output_feature)
                  #print(f"model :{model.state_dict()}")
                  loss_fn = nn.BCEWithLogitsLoss()
                  optimizer = torch.optim.Adam(    model.parameters(),    lr=learning_rate)
                  #-----------------------------------------------------------    -----------------
                  #                                   log the params
                  #---------------------------------------------------------------------------
                  EPO =100
                  mlflow.log_params({
                        "EPOCHS":EPO,
                        "OPTIMIZER ":optimizer.__class__.__name__,
                        "LOSS_FUNCTION ":loss_fn.__class__.__name__,
                        "MODEL_NAME":model.__class__.__name__,
                        "LEARNING RATE":learning_rate,""
                        "INPUT_FEATURE ":input_feature,
                        "HIDDEN_FEATURE":hidden_feature,
                        "OUTPUT_FEATURE":output_feature,
                        })
                  results = train(
                                model=model,
                                train_dataloader=train_dataloader,
                                test_dataloader=test_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                epochs=EPO,
                                device=device,
                                log_to_mlflow=True
                                )
                  ## log in the final values but the last one 
                  final_train_loss =results["train_loss"][-1]

                  final_train_acc =results["train_acc"][-1]

                  final_test_acc =results["test_acc"][-1]

                  final_test_loss =results["test_loss"][-1]
                  mlflow.log_params({"learning_rate": learning_rate,
                                     "hidden_features": hidden_feature
                                     })
                  mlflow.log_metric("final_test_accuracy",final_test_acc)
                  if final_test_acc > best_accuracy:
                       best_accuracy = final_test_acc
                       best_run_id = child_run.info.run_id
                       best_params = {
                             "learning_rate": learning_rate,
                             "hidden_features": hidden_feature
                               }

mlflow.log_metric("best_validation_accuracy", best_accuracy)
mlflow.log_params({
        "best_learning_rate": best_params["learning_rate"],
        "best_hidden_features": best_params["hidden_features"],
        "best_child_run_id": best_run_id
    })
     # adding signature after training and inside model.eval()
model.eval()
input_tensor =X_batch[:4].float().to(device)
with torch.inference_mode():              
            pred_tensor =model(input_tensor)
    # mlflow signature uses CPU nympy array
input_sig =input_tensor.detach().cpu().numpy()
pred =pred_tensor.detach().cpu().numpy()
signature = infer_signature(input_sig,pred )



mlflow.pytorch.log_model(
        pytorch_model=model,
        name="BINARY_CLASSIFIER",  
        input_example=input_sig[:1] ,
        signature =signature ,
        serialization_format="pickle", 
    )



                  
                  
               
                        



                  

    

















  

  


