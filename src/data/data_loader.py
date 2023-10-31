from ..paths import *
import pandas as pd
import os

def get_nested_csv(dir_name: str, file_name: str):
    """
    Returns a joint pandas dataframe from the files matching file_name
    in all the different import dates subdirectories of the directory
    specified by dir_name
    
    :param dir_name: The name of the directory (e.g. "seantis")
    :param file_name: The name of the csv file to be read. (e.g. "diagnoses.csv")
    """
    list_dfs = []

    for root, dirs, files in os.walk(os.path.join(DATA_PATH_RAW, dir_name)):
        
        if root.split(os.sep)[-1].startswith("imported_"):
            try:
                _df = pd.read_csv(os.path.join(root, file_name))
                list_dfs.append(_df)
            except FileNotFoundError:
                print(f"File not found in: {root}")
                continue
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError in: {root}")
                continue

    df = pd.concat(list_dfs)
    return df