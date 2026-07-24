from pathlib import Path
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from imbalance import balance_training_data
from sklearn.preprocessing import StandardScaler
def data_download(url:str, data_dir:str,filename:str):
    data_path =Path(data_dir)
    data_path.mkdir(parents=True,exist_ok=True)

    file_path =data_path/filename

    if file_path.exists():
        print(f"file already exist {file_path}")
    else:
        print(f"downloading data from {url}")
        respond= requests.get(url)
        respond.raise_for_status()
        with open(file_path,"wb") as f:
            f.write(respond.content)
        print(f"data saved to path{file_path}")
    return file_path
import pandas as pd


def check_data_quality(
    dataframe,
    target_column,
    irrelevant_columns=None
):
    """
    Performs basic data-quality checks and removes selected columns.
    """

    # Work on a copy to protect the original dataset
    df = dataframe.copy()

    # If no columns are provided, use an empty list
    if irrelevant_columns is None:
        irrelevant_columns = []

    # Check missing values
    missing_values = df.isnull().sum()

    # Check duplicate rows
    duplicate_rows = df.duplicated().sum()

    # Check class distribution
    class_distribution = df[target_column].value_counts()

    # Remove irrelevant columns
    df = df.drop(columns=irrelevant_columns)

    print("Missing values:")
    print(missing_values)

    print("\nDuplicate rows:")
    print(duplicate_rows)

    print("\nClass distribution:")
    print(class_distribution)

    print("\nRemoved columns:")
    print(irrelevant_columns)

    return df


def split_balance(cleaned_dataframe:pd.DataFrame,target:str):
     # Separate features and target
    X = cleaned_dataframe.drop(columns=target)
    y = cleaned_dataframe[target]
    
    # Split first
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
   # Save names before StandardScaler returns NumPy arrays
    feature_columns = X_train.columns

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=feature_columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=feature_columns,
        index=X_test.index
    )

    # Balance only the training data
    X_train, y_train = balance_training_data(
        X_train=X_train_scaled,
        y_train=y_train.reset_index(drop =True),
        method="oversampling"
    )

    return X_train,X_test_scaled,y_train,y_test




