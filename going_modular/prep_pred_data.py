import jsonimport joblib
import pandas as pd
import torch

def prepare_new_data(
        new_data:pd.DataFrame,
        scaler_path:str,
        feature_column_path:str,
        irrelevant_column =None
)->torch.Tensor:
    df =new_data.copy()
    if irrelevant_column is None:
        irrelevant_column=[]


    # Remove columns that were not used during training
    columns_to_remove =[
        column 
        for column in irrelevant_column
        if column in df.columns
    ]
    if columns_to_remove:
        df =df.drop(columns=columns_to_remove)

    #New prediciton data should not contain the target 
    if "Y" in df.columns:
        df= df.drop(columns =["Y"])

    # load the exact feature order used for training 
    with open(feature_column_path,"r") as file:
        feature_columns=json.load(file)

    # check for missing features 
    missing_column =[
        column
        for column in feature_columns
        if column not in df.columns

    ]

    if missing_columns:
        raise ValueError(
            f"New data is missing Columns:{missing_columns}"
        )
    #remove unexpected columns and restore training order
    df =df[feature_columns]

    # oad the scaler fitted on training data 
    scaler =joblib.load(scaler_path)

    # transform only -never fit again
    scaled_data =scaler.transform(df)
    # conver tt o tensor 
    input_tensor =torch.as_tensor(scaled_data ,dtype=torch.float32)

    return input_tensor 
