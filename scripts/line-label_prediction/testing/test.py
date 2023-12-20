import numpy as np

import pandas as pd

import time

import torch

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src import paths

print("Numpy version: ", np.__version__)

a = np.array([1, 2, 3])
print("a: ", a)

# Check if GPU is available
print("GPU available: ", torch.cuda.is_available())
print("GPU device name: ", torch.cuda.get_device_name(0))
print("Number of GPUs: ", torch.cuda.device_count())
print("GPU memory per device: ", torch.cuda.get_device_properties("cuda:0"))

# Save a list of numbers to a file
b = ["blabla", "blabla2", "blabla3"]
pd.Series(b).to_csv(paths.RESULTS_PATH/"test.csv", index=False, header=False)

# Count to 30 with 1 second delay
for i in range(30):
    print(i)
    time.sleep(1)