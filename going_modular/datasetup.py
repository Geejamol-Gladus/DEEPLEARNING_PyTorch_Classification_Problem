from pathlib import Path
import requests
import pandas as pd
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
    