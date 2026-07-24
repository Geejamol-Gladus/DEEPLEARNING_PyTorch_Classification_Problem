# convert eh tesnors to data loader
import torch
import pandas as pd
from torch.utils.data import DataLoader ,TensorDataset
def dataloader(X_train:pd.DataFrame,
               y_train:pd.DataFrame,
               X_test:pd.DataFrame,
               y_test:pd.DataFrame,batch_size:int =32):
    
    # conver teh data frame to tensor 

    X_train_tensor =torch.tensor(X_train.to_numpy(),dtype = torch.float32)

    input_feature =X_train_tensor.shape[1]

    X_test_tensor =torch.tensor(X_test.to_numpy(),dtype =torch.float32)

    y_train_tensor =torch.tensor(y_train.to_numpy(),dtype =torch.float32)

    y_test_tensor =torch.tensor(y_test.to_numpy(),dtype =torch.float32)
    # pair each row  with its target
    train_dataset =TensorDataset(X_train_tensor,y_train_tensor)
    test_dataset =TensorDataset(X_test_tensor,y_test_tensor)

    #create dataloader
    train_dataloader =DataLoader(dataset=train_dataset,
                                 batch_size=32,
                                 shuffle=True)
    test_dataloader =DataLoader(dataset=test_dataset,
                                batch_size=32,shuffle=False)
    return train_dataloader,test_dataloader,input_feature
  

    
