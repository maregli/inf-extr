import numpy as np
import time
import torch

print("Numpy version: ", np.__version__)

a = np.array([1, 2, 3])
print("a: ", a)

# Check if GPU is available
print("GPU available: ", torch.cuda.is_available())
print("GPU device name: ", torch.cuda.get_device_name(0))
print("Number of GPUs: ", torch.cuda.device_count())
print("GPU memory per device: ", torch.cuda.get_device_properties("cuda:0"))

# Count to 50 with 1 second delay
for i in range(50):
    print(i)
    time.sleep(1)