import torch
from torch import nn
import mlflow 
import copy 

def train_setup(model,train_dataloader,loss_fn,optimizer):
    model.train()
    train_loss=0
    total_sample=0
    total_correct =0
    for X_batch,y_batch in train_dataloader:
        # send them to target device 
        # do the forward pass
         
        logits =model(X_batch).squeeze(1)

        # calaulate teh loss

        loss =loss_fn(logits,y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # get teh number of samples in current batch

        batch_size =X_batch.size(0)

        #  calulate the average 
        train_loss+=loss.item()*batch_size

        #caluclate  the accuracy since binary classification
        pred =(torch.sigmoid(logits)>=0.5).float()
        total_correct+=(pred==y_batch).sum().item()

        # count all samples 
        total_sample +=batch_size

    # find teh average loss for the enitre data set 
    average_loss =train_loss/total_sample 
    average_acc =total_correct/total_sample

    return  average_loss,average_acc
    






def test_setup(model,test_dataloader,loss_fn):
    model.eval()
    test_loss =0
    test_correct =0
    total_sample =0
    with torch.inference_mode():
        for X_batch,y_batch in test_dataloader:
             logits =model(X_batch).squeeze(1)

             loss =loss_fn(logits,y_batch)
             # calulate  sample per batch
             batch_size =X_batch.size(0)

             test_loss +=loss.item()*batch_size
             pred =(torch.sigmoid(logits)>=0.5).float()
             test_correct+=(pred==y_batch).sum().item()
             total_sample+=batch_size
    aver_test_loss =test_loss/total_sample
    aver_test_acc =test_correct/total_sample
    return aver_test_loss,aver_test_acc

def train(model,train_dataloader,test_dataloader,loss_fn,optimizer,epoch):
    epochs =epoch
    #-----------------------------------------
    #TO FIND OUT THE BEST EPOCH AND BEST WEIGHT 
    #-----------------------------------------

    best_epoch =0
    best_test_accuracy =float("-inf")
    best_test_loss =float("-inf")
    best_state_mold =None

    #---------------------------------------------
    #store metrics
    result = {
        "train_loss" :[],
        "test_loss":[],
        "train_acc":[],
        "test_acc":[]   }
    for epoch in range(epochs):
        train_loss,train_acc =train_setup(model=model,train_dataloader=train_dataloader,loss_fn =loss_fn,optimizer=optimizer)

        test_loss,test_acc =test_setup(model =model,test_dataloader=test_dataloader,loss_fn=loss_fn)

        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["test_loss"].append(test_loss)
        result["test_acc"].append(test_acc)
        #-----------------------------------------------------------
        # TEH BEST EPOCH
        #-----------------------------------------------------------
        if test_acc>best_test_accuracy:
            best_test_accuracy=test_acc
            best_test_loss=test_loss
            best_epoch=epoch+1
            best_model_state =copy.deepcopy(model.state_dict())


        # log all epoch metric to mlflow 
        """mlflow.log_metrics ({
            "train_loss":train_loss,
            "test_loss":test_loss,
            "train_accuracy":train_acc,
            "test_accuracy ":test_acc},step =epoch
        )"""



        print(f"the number of epochs are:{epoch}")
        print(f"the train accuracry is :{train_acc}")
        print(f"the test accuracy is :{test_acc}")
        print(f"the train loss is {train_loss}")
        print(f"the test loss is{test_loss}")
    return {
        "history" :result,
        "best_test_accuracy":best_test_accuracy,
        "best_test_loss": best_test_loss,
        "best_model_state":best_model_state,
        "best_epoch":best_epoch
    }