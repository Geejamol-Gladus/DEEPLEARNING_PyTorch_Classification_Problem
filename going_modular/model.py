
import torch
from torch import nn

class Classgitmodel(nn.Module):

    def __init__(self,input_features,hidden_features,output_feature):
        super().__init__()
        self.Layer1 =nn.Sequential(
            nn.Linear(in_features=input_features,out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features,out_features=output_feature)
        )
    def forward(self,x):
        return self.Layer1(x)