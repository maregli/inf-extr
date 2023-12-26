
from datasets import DatasetDict, load_dataset

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths

import pandas as pd



def load_data()->DatasetDict:
    """Loads the data for MS-Diag task and returns the dataset dictionary
    
    Returns:
        DatasetDict: Returns the dataset dictionary
    """

    data_files = {"train": "ms-diag_clean_train.csv", "validation": "ms-diag_clean_val.csv", "test": "ms-diag_clean_test.csv"}

    df = pd.read_csv(os.path.join(paths.DATA_PATH_PREPROCESSED,'ms-diag/ms-diag_clean_train.csv'))
    
    return df

def main():
    print("Trying to load df")

    # Load Data
    df = load_data()

    print(df)

    return

if __name__ == "__main__":
    main()


